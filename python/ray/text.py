import logging
from abc import abstractmethod

import ray
import torch
import torch.nn as nn

from ..tensor_parallel import VocabParallelEmbedding
from ..tensor_parallel.cross_entropy import vocab_parallel_causal_cross_entropy
from .tensor_transfer import (
    TensorTransferRequest,
    prepare_tensor_for_transfer,
    receive_tensor,
)
from .trainer import Trainer
from .utils import get_physical_gpu_id, init_distributed_comm

logger = logging.getLogger(__name__)


class BaseTextTrainer(Trainer):
    """Base class for text trainers with common functionality."""

    def __init__(self, config, rank: int):
        super().__init__(config, rank)
        self.receiver_gpu_ids = None  # GPU IDs of vision trainers (for sending gradients back)
        self.use_ipc = False  # Whether to use CUDA IPC for this actor
        self._vision_grad_owner_rank = rank  # Which rank should receive gradients from this actor
        logger.debug(f"[r{self.rank}] {self.__class__.__name__} initialized")

    @abstractmethod
    def _create_model_and_lm_head(self, model_config):
        """
        Create model and lm_head instances.
        Returns: (model, lm_head) tuple.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _get_embedding_module(self, model):
        """
        Get the embedding module from the model.
        Returns: embedding module (e.g., embed_tokens).
        Must be implemented by subclasses.
        """
        pass

    def _replace_embedding_module(self, model, new_embedding):
        """
        Replace the embedding module in the model with a new one.
        Handles different model hierarchies (Qwen vs InternVL).

        Args:
            model: The model whose embedding should be replaced
            new_embedding: The new embedding module
        """
        # Try direct replacement for Qwen-style models
        if hasattr(model, "embed_tokens"):
            model.embed_tokens = new_embedding
            return
        if hasattr(model, "tok_embeddings"):
            model.tok_embeddings = new_embedding
            return

        # For InternVL-style models, search through hierarchy
        if hasattr(model, "model"):
            self._replace_embedding_module(model.model, new_embedding)
            return
        if hasattr(model, "decoder"):
            self._replace_embedding_module(model.decoder, new_embedding)
            return

        raise ValueError("Could not find embedding module to replace in model")

    def _build_model_architecture(
        self, config, model_name, device, torch_dtype, parallelism, load_pretrained_path=None
    ):
        """
        Common model building logic - template method pattern.
        Subclasses implement specific abstract methods for customization.

        Args:
            load_pretrained_path: Optional path to pretrained checkpoint directory

        Returns:
            tuple: (model, lm_head, tp_group)
        """
        # Load model config
        logger.debug(f"[r{self.rank}] Loading text model config...")
        model_config = self._load_model_config(model_name)

        # Configure attention backend
        attention_backend = config["attention_backend"]
        self._configure_attention_backend(model_config, attention_backend, "text_config")

        # Handle DeepSpeed parallelism
        if parallelism == "deepspeed":
            import torch.distributed as dist

            from .utils import init_distributed_comm

            if not dist.is_initialized():
                init_distributed_comm(backend="nccl", use_deepspeed=True)
            return self._build_model_with_deepspeed(config, model_config, device, torch_dtype, load_pretrained_path)

        if parallelism == "autotp":
            import os

            import torch.distributed as dist

            from .utils import init_distributed_comm

            if not dist.is_initialized():
                world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
                if world_size_env > 1:
                    init_distributed_comm(backend="nccl", use_deepspeed=True)
            return self._build_model_with_autotp(config, model_config, device, torch_dtype, load_pretrained_path)

        # Validate tensor parallelism compatibility
        if parallelism == "tensor":
            self._validate_tensor_parallelism(model_config)

        # Set default device
        torch.set_default_device(device)

        logger.debug(f"[r{self.rank}] Creating text model...")
        model, lm_head = self._create_model_and_lm_head(model_config)

        # Reset default device
        torch.set_default_device("cpu")
        model.to(torch_dtype)
        lm_head.to(torch_dtype)

        # Load pretrained weights BEFORE tensor parallelism
        if load_pretrained_path:
            import os

            text_checkpoint_path = os.path.join(load_pretrained_path, "text_model.pt")
            # Temporarily assign model for loading
            self.model = model
            self.load_pretrained_weights(text_checkpoint_path)
            model = self.model

        # Activation checkpointing
        activation_checkpointing = config["activation_checkpointing"]
        self._apply_activation_checkpointing(model, activation_checkpointing, "text model")

        # Apply tensor parallelism if needed (AFTER loading pretrained weights)
        tp_group = None
        if parallelism == "tensor":
            tp_group = self._apply_tensor_parallelism(model, lm_head, model_config, device)

        model.to(device=device, dtype=torch_dtype)
        lm_head.to(device=device, dtype=torch_dtype)

        # Tie weights between lm_head and embeddings
        embed_module = self._get_embedding_module(model)
        if embed_module is None:
            raise ValueError("Unable to locate embedding module for tying the LM head.")
        self._tie_lm_head_to_embeddings(lm_head, embed_module)

        model.train()
        lm_head.train()

        return model, lm_head, tp_group

    def _validate_tensor_parallelism(self, model_config, tp_world_size=None):
        """Validate that tensor parallelism is compatible with model config."""
        if tp_world_size is None:
            init_distributed_comm(backend="nccl")
            import torch.distributed as dist

            if not dist.is_initialized():
                return
            tp_world_size = dist.get_world_size()

        # Get num_key_value_heads from text_config
        text_config = getattr(model_config, "text_config", model_config)
        num_kv_heads = text_config.num_key_value_heads
        if num_kv_heads % tp_world_size != 0:
            raise ValueError(
                f"Tensor parallel world size must divide num_key_value_heads. "
                f"Got world size {tp_world_size} and num_key_value_heads {num_kv_heads}."
            )

    def _build_model_with_deepspeed(self, config, model_config, device, torch_dtype, load_pretrained_path=None):
        """
        Build model with DeepSpeed ZeRO optimization.

        Args:
            config: Training config dict
            model_config: Model configuration
            device: Target device
            torch_dtype: Target dtype
            load_pretrained_path: Optional path to pretrained checkpoint directory

        Returns:
            (model_engine, lm_head, tp_group) tuple where model_engine is DeepSpeed engine
        """
        # Build model on target device
        torch.set_default_device(device)
        logger.debug(f"[r{self.rank}] Creating text model for DeepSpeed...")
        model, lm_head = self._create_model_and_lm_head(model_config)
        torch.set_default_device("cpu")

        model.to(torch_dtype)
        lm_head.to(torch_dtype)

        # Load pretrained weights BEFORE DeepSpeed wrapping
        if load_pretrained_path:
            import os

            text_checkpoint_path = os.path.join(load_pretrained_path, "text_model.pt")
            # Temporarily assign model for loading
            self.model = model
            self.load_pretrained_weights(text_checkpoint_path)
            model = self.model

        # Apply activation checkpointing
        activation_checkpointing = config["activation_checkpointing"]
        self._apply_activation_checkpointing(model, activation_checkpointing, "text model")

        # Tie weights between lm_head and embeddings
        embed_module = self._get_embedding_module(model)
        if embed_module is None:
            raise ValueError("Unable to locate embedding module for tying the LM head.")
        self._tie_lm_head_to_embeddings(lm_head, embed_module)

        # Attach lm_head to model so DeepSpeed can manage both
        model.lm_head = lm_head

        # Collect parameters from model and lm_head
        # Note: After tying weights, embed_tokens.weight and lm_head.weight are the same object
        # The helper method will deduplicate to avoid passing the same parameter twice to DeepSpeed
        params = self._collect_parameters_from_modules(model, lm_head)

        # Initialize with DeepSpeed using base class method
        model_engine, optimizer, _, _ = self._initialize_deepspeed(
            model=model,
            params=params,
            config=config,
            torch_dtype=torch_dtype,
        )

        # Return the engine in place of model (lm_head stays separate, tp_group is None)
        return model_engine, lm_head, None

    def _build_model_with_autotp(self, config, model_config, device, torch_dtype, load_pretrained_path=None):
        """
        Build model with DeepSpeed AutoTP tensor parallelism.

        Args:
            config: Training config dict
            model_config: Model configuration
            device: Target device
            torch_dtype: Target dtype
            load_pretrained_path: Optional path to pretrained checkpoint directory

        Returns:
            (model_engine, lm_head, tp_group) tuple where model_engine is DeepSpeed engine
        """
        import torch.distributed as dist
        from deepspeed.module_inject.layers import set_autotp_mode
        from deepspeed.utils import groups

        # Enable AutoTP instrumentation before model creation
        set_autotp_mode(training=True)

        # Build model on target device
        torch.set_default_device(device)
        logger.debug(f"[r{self.rank}] Creating text model for AutoTP...")
        model, lm_head = self._create_model_and_lm_head(model_config)
        torch.set_default_device("cpu")

        model.to(torch_dtype)
        lm_head.to(torch_dtype)

        # Load pretrained weights BEFORE DeepSpeed wrapping
        if load_pretrained_path:
            import os

            text_checkpoint_path = os.path.join(load_pretrained_path, "text_model.pt")
            # Temporarily assign model for loading
            self.model = model
            self.load_pretrained_weights(text_checkpoint_path)
            model = self.model

        # Apply activation checkpointing
        activation_checkpointing = config["activation_checkpointing"]
        self._apply_activation_checkpointing(model, activation_checkpointing, "text model")

        # Tie weights between lm_head and embeddings
        embed_module = self._get_embedding_module(model)
        if embed_module is None:
            raise ValueError("Unable to locate embedding module for tying the LM head.")
        self._tie_lm_head_to_embeddings(lm_head, embed_module)

        # Attach lm_head to model so DeepSpeed can manage both
        model.lm_head = lm_head

        # Collect parameters from model and lm_head
        params = self._collect_parameters_from_modules(model, lm_head)

        # Determine AutoTP size (defaults to world size if not provided)
        autotp_size = config.get("autotp_size") or config.get("tensor_parallel_size")
        if autotp_size is None:
            autotp_size = dist.get_world_size() if dist.is_initialized() else 1
        autotp_size = int(autotp_size)
        if autotp_size <= 0:
            raise ValueError(f"autotp_size must be > 0, got {autotp_size}")
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if autotp_size > world_size or world_size % autotp_size != 0:
                raise ValueError(
                    f"Invalid autotp_size {autotp_size} for world size {world_size}: must divide world size"
                )
            data_parallel_size = world_size // autotp_size
            logger.debug(
                f"[r{self.rank}] AutoTP sizes -> tp={autotp_size}, data_parallel={data_parallel_size}, world={world_size}"
            )
        elif autotp_size > 1:
            raise ValueError("autotp_size > 1 requires torch.distributed to be initialized")

        # Validate tensor parallelism compatibility with chosen AutoTP size
        self._validate_tensor_parallelism(model_config, tp_world_size=autotp_size)

        # Count parameters before TP sharding for logging
        params_before_tp = sum(p.numel() for p in model.parameters())
        logger.debug(f"[r{self.rank}] Parameters BEFORE TP sharding: {params_before_tp:,}")

        # Apply tensor parallelism with deepspeed.tp_model_init()
        # This actually shards the model parameters across TP ranks
        # NOTE: set_autotp_mode(training=True) alone only enables tracking but does NOT shard weights.
        # We must call tp_model_init() to perform the actual module injection and weight sharding.
        import deepspeed

        logger.debug(f"[r{self.rank}] Applying TP sharding with deepspeed.tp_model_init (tp_size={autotp_size})...")
        model = deepspeed.tp_model_init(model, tp_size=autotp_size, dtype=torch_dtype)

        # Count parameters after TP sharding
        params_after_tp = sum(p.numel() for p in model.parameters())
        reduction_pct = 100 * (params_before_tp - params_after_tp) / params_before_tp
        logger.debug(f"[r{self.rank}] Parameters AFTER TP sharding: {params_after_tp:,} ({reduction_pct:.1f}% reduction)")

        # Get TP group early for vocab parallel embedding
        tp_group = None
        try:
            tp_group = groups.get_tensor_model_parallel_group()
        except Exception:
            tp_group = None

        # Apply vocabulary-parallel embedding for proper parallel loss computation
        # DeepSpeed AutoTP doesn't partition embeddings/lm_head by vocabulary dimension,
        # so we replace them with VocabParallelEmbedding for correct loss computation
        if tp_group is not None and config.get("use_vocab_parallel", True):
            tp_rank = groups.get_tensor_model_parallel_rank()
            tp_world_size = groups.get_tensor_model_parallel_world_size()

            original_embedding = self._get_embedding_module(model)
            if original_embedding is not None:
                logger.debug(
                    f"[r{self.rank}] Replacing embedding with VocabParallelEmbedding "
                    f"(vocab_size={original_embedding.num_embeddings}, tp_size={tp_world_size})"
                )

                # Create VocabParallelEmbedding
                vocab_parallel_embedding = VocabParallelEmbedding(
                    num_embeddings=original_embedding.num_embeddings,
                    embedding_dim=original_embedding.embedding_dim,
                    padding_idx=original_embedding.padding_idx,
                    tp_group=tp_group,
                    dtype=original_embedding.weight.dtype,
                    device=device,
                )

                # Copy the appropriate partition of weights from original embedding
                with torch.no_grad():
                    start_idx = vocab_parallel_embedding.vocab_start_index
                    end_idx = vocab_parallel_embedding.vocab_end_index
                    vocab_parallel_embedding.weight.data.copy_(original_embedding.weight.data[start_idx:end_idx])

                # Replace the embedding in the model
                self._replace_embedding_module(model, vocab_parallel_embedding)

                # Update lm_head to use the partitioned embedding weights (tied weights)
                # The lm_head output dimension should now match the partitioned vocab size
                lm_head = model.lm_head
                lm_head.weight = vocab_parallel_embedding.weight

                logger.debug(
                    f"[r{self.rank}] VocabParallelEmbedding applied: vocab_range=[{start_idx}, {end_idx}), "
                    f"partition_size={end_idx - start_idx}"
                )

        # Re-collect parameters after TP sharding (shapes have changed)
        # The lm_head is attached to model, so we get it from there
        lm_head = model.lm_head
        params = self._collect_parameters_from_modules(model, lm_head)

        tp_overlap_comm = config.get("tp_overlap_comm", None)
        tensor_parallel_cfg = {"autotp_size": autotp_size}
        if tp_overlap_comm is not None:
            tensor_parallel_cfg["tp_overlap_comm"] = bool(tp_overlap_comm)

        # Initialize DeepSpeed with AutoTP configuration
        model_engine, optimizer, _, _ = self._initialize_deepspeed(
            model=model,
            params=params,
            config=config,
            torch_dtype=torch_dtype,
            tensor_parallel_config=tensor_parallel_cfg,
        )

        return model_engine, lm_head, tp_group

    def _apply_tensor_parallelism(self, model, lm_head, model_config, device):
        """Apply tensor parallelism to model, lm_head, and return tp_group."""
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import parallelize_module

        assert dist.is_initialized(), "Distributed must be initialized for tensor parallelism."

        tp_world_size = dist.get_world_size()
        tp_mesh = init_device_mesh("cuda", (tp_world_size,))

        logger.debug(f"[r{self.rank}] Applying tensor parallelism to text model...")

        # Replace embedding with VocabParallelEmbedding for correct vocabulary parallelism
        logger.debug(f"[r{self.rank}] Replacing embedding layer with VocabParallelEmbedding...")
        original_embedding = self._get_embedding_module(model)
        if original_embedding is None:
            raise ValueError("Unable to locate embedding module for tensor parallelism.")

        vocab_parallel_embedding = VocabParallelEmbedding(
            num_embeddings=original_embedding.num_embeddings,
            embedding_dim=original_embedding.embedding_dim,
            padding_idx=original_embedding.padding_idx,
            tp_group=tp_mesh.get_group(),
            tp_mesh=tp_mesh,
            dtype=original_embedding.weight.dtype,
            device=device,
        )

        # Copy the appropriate partition of weights from original embedding
        with torch.no_grad():
            start_idx = vocab_parallel_embedding.vocab_start_index
            end_idx = vocab_parallel_embedding.vocab_end_index
            vocab_parallel_embedding.weight.data.copy_(original_embedding.weight.data[start_idx:end_idx])

        # Replace the embedding layer in the model
        # For Qwen models, this is model.embed_tokens
        # For InternVL, we need to search through the model hierarchy
        self._replace_embedding_module(model, vocab_parallel_embedding)
        logger.debug(
            f"[r{self.rank}] Embedding replaced: vocab_range=[{start_idx}, {end_idx}), "
            f"partition_size={end_idx - start_idx}"
        )

        # Parallelize transformer layers
        layers = self._get_transformer_layers(model)
        if layers is None:
            raise ValueError("Unable to locate decoder layers for tensor parallelism.")

        tp_mapping = self._get_tensor_parallel_mapping()
        for layer in layers:
            parallelize_module(layer, tp_mesh, tp_mapping, src_data_rank=0)

        # NOTE: Do NOT parallelize lm_head here because its weights will be
        # tied to the VocabParallelEmbedding, which already has the correct sharding

        return tp_mesh.get_group()

    def load_pretrained_weights(self, checkpoint_path: str):
        """
        Load pretrained weights from a split checkpoint.

        Args:
            checkpoint_path: Path to the text_model.pt checkpoint file
        """
        import os

        if not os.path.exists(checkpoint_path):
            logger.warning(f"[r{self.rank}] Pretrained checkpoint not found: {checkpoint_path}")
            return False

        logger.debug(f"[r{self.rank}] Loading pretrained text weights from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)

            # Remove 'language_model.' prefix if present (HuggingFace full model vs text-only component)
            # Also handle 'model.' prefix
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Remove 'language_model.' prefix
                if new_key.startswith("language_model."):
                    new_key = new_key[len("language_model.") :]
                # Remove 'model.' prefix if it's still there
                elif new_key.startswith("model."):
                    new_key = new_key[len("model.") :]
                cleaned_state_dict[new_key] = value

            # Load into model
            if self.use_deepspeed and self.deepspeed_engine is not None:
                # For DeepSpeed, load into the wrapped module
                missing_keys, unexpected_keys = self.deepspeed_engine.module.load_state_dict(
                    cleaned_state_dict, strict=False
                )
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)

            if missing_keys:
                logger.warning(f"[r{self.rank}] Missing keys when loading text weights: {missing_keys[:10]}")
            if unexpected_keys:
                logger.warning(f"[r{self.rank}] Unexpected keys when loading text weights: {unexpected_keys[:10]}")

            logger.info(
                f"[r{self.rank}] Successfully loaded pretrained text weights ({len(cleaned_state_dict)} parameters)"
            )
            return True

        except Exception as e:
            logger.error(f"[r{self.rank}] Failed to load pretrained weights from {checkpoint_path}: {e}")
            return False

    def build_model(self):
        """Build text model based on the model type specified in config."""
        model_name = self.config["model_name"]
        parallelism = self.config["parallelism"]
        dtype_str = self.config["dtype"]

        torch_dtype = self._get_torch_dtype(dtype_str)
        device = self._get_device()

        # Check if we need to load pretrained weights
        pretrained_checkpoint_dir = self.config.get("pretrained_checkpoint_dir", None)

        # Build model architecture (subclass-specific)
        # Note: For tensor parallelism, we need to load weights BEFORE applying TP
        model, lm_head, tp_group = self._build_model_architecture(
            self.config, model_name, device, torch_dtype, parallelism, load_pretrained_path=pretrained_checkpoint_dir
        )

        # Make sure to free up the memory used before TP partitioning
        torch.cuda.empty_cache()
        logger.debug(f"[r{self.rank}] Text model built successfully")
        self.model = model

        # Store model config and lm_head for later use
        # When using DeepSpeed, model is the engine, so get config and lm_head from the wrapped module
        if self.use_deepspeed:
            self.model_config = self.deepspeed_engine.module.config
            self.lm_head = self.deepspeed_engine.module.lm_head
        else:
            self.model_config = model.config
            self.lm_head = lm_head

        self.model.tp_group = tp_group

        # Build optimizer (skip if using DeepSpeed)
        if not self.use_deepspeed:
            self._build_optimizer(self._collect_optimizer_parameters())
        else:
            logger.debug(f"[r{self.rank}] Using DeepSpeed optimizer, skipping manual optimizer creation")

        # Build LR scheduler for both DeepSpeed and non-DeepSpeed modes
        self._build_scheduler(self.config["num_iterations"])

        # Initialize loss function (CrossEntropyLoss with ignore_index=-100 for padding)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def _tie_lm_head_to_embeddings(self, lm_head: nn.Linear, embed_tokens) -> None:
        """Share weights between the LM head projection and token embeddings.

        Args:
            lm_head: The language model head (nn.Linear)
            embed_tokens: Either nn.Embedding or VocabParallelEmbedding
        """
        if lm_head is None:
            return

        # Both VocabParallelEmbedding and nn.Embedding have .weight
        lm_head.weight = embed_tokens.weight

    def _get_actual_model(self):
        """Get the actual model, unwrapping DeepSpeed engine if needed."""
        if self.use_deepspeed:
            return self.deepspeed_engine.module
        return self.model

    def _collect_optimizer_parameters(self):
        """Gather parameters that should be optimized."""
        params = []
        if hasattr(self, "model") and self.model is not None:
            params.extend(list(self.model.parameters()))
        if hasattr(self, "lm_head") and self.lm_head is not None:
            params.extend(list(self.lm_head.parameters()))
        return params

    def initialize_trainer(self):
        """Initialize the trainer with real dataset via build_dataloader (text)."""
        logger.debug(f"[r{self.rank}] {self.__class__.__name__} initialize_trainer called")

        # Build dataloader using common helper method
        # Use modality="all" to get full sequence with <image> placeholder tokens
        self.dataloader = self._build_dataloader(modality="all")
        self.data_iterator = iter(self.dataloader)

        batch_size = self.config["batch_size"]
        dataset_len = len(self.dataloader.dataset)
        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__} initialized with {dataset_len} samples, batch_size={batch_size}"
        )

    def set_receiver_info(self, receiver_gpu_ids: list[str], use_ipc: bool):
        """Set receiver GPU IDs and whether to use CUDA IPC.

        Args:
            receiver_gpu_ids: List of physical GPU IDs of vision trainers that will receive gradients from this text trainer
            use_ipc: Whether to use CUDA IPC for transfer
        """
        self.receiver_gpu_ids = receiver_gpu_ids
        self.use_ipc = use_ipc
        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__}: receiver_gpu_ids={receiver_gpu_ids}, use_ipc={use_ipc}"
        )

    def forward_step(self, vision_embeddings_ref, iteration: int = -1):
        """
        Run one forward pass on the text model with vision embeddings.

        Args:
            vision_embeddings_ref: Ray object reference, TensorTransferRequest dict, or tensor from VisionTrainer
                                   [batch_size, num_image_tokens, hidden_size]
            iteration: Training iteration number for synchronization verification

        Returns:
            Loss value (scalar tensor)
        """
        # Store iteration for verification
        self._current_iteration = iteration
        logger.debug(f"[r{self.rank}] Text forward_step: iteration={iteration}")

        # If we received an object reference, call ray.get() to retrieve the actual data
        if isinstance(vision_embeddings_ref, ray.ObjectRef):
            vision_data = ray.get(vision_embeddings_ref)
        else:
            vision_data = vision_embeddings_ref

        # Vision trainer returns dict with embeddings and metadata
        if not isinstance(vision_data, dict):
            raise RuntimeError(
                f"[r{self.rank}] Expected vision_data to be dict with 'vision_embeddings', 'sample_index', 'iteration', "
                f"but got type {type(vision_data)}"
            )

        vision_embeddings_data = vision_data.get("vision_embeddings")
        vision_sample_index = vision_data.get("sample_index")
        vision_iteration = vision_data.get("iteration")

        # Guard: Verify iteration matches
        if vision_iteration != iteration:
            raise RuntimeError(
                f"[r{self.rank}] Iteration mismatch! Vision trainer iteration={vision_iteration}, "
                f"text trainer iteration={iteration}. Trainers are out of sync!"
            )

        logger.debug(
            f"[r{self.rank}] Received from vision: iteration={vision_iteration}, sample_index={vision_sample_index}"
        )

        # Check if we received a TensorTransferRequest (dict with 'use_ipc')
        if isinstance(vision_embeddings_data, dict) and "use_ipc" in vision_embeddings_data:
            transfer_request = TensorTransferRequest.from_dict(vision_embeddings_data)
            receiver_gpu_id = get_physical_gpu_id()
            vision_embeddings = receive_tensor(transfer_request, receiver_gpu_id)
            logger.debug(
                f"[r{self.rank}] Received embeddings via {'CUDA IPC' if transfer_request.use_ipc else 'Ray transport'}"
            )
        else:
            # Regular tensor
            vision_embeddings = vision_embeddings_data

        # Each rank owns gradients for its own microbatch
        self._vision_grad_owner_rank = self.rank

        return self._forward_step_impl(vision_embeddings, vision_sample_index, iteration)

    def _forward_step_impl(self, vision_embeddings, vision_sample_index=None, iteration=-1):
        """
        Implementation of forward pass. Used by both forward_step and forward_step_with_broadcast.

        Args:
            vision_embeddings: Vision embeddings tensor [batch_size, num_image_tokens, hidden_size]
            vision_sample_index: Sample index from vision trainer for synchronization verification
            iteration: Training iteration number

        Returns:
            Loss value (scalar tensor)
        """
        # Get next batch, reset iterator if exhausted
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            batch = next(self.data_iterator)

        # Guard: Verify sample_index matches between vision and text
        if "sample_index" in batch:
            text_sample_index = batch["sample_index"]
            if isinstance(text_sample_index, torch.Tensor):
                text_sample_index = (
                    text_sample_index.item() if text_sample_index.numel() == 1 else text_sample_index.tolist()
                )

            if vision_sample_index is not None and text_sample_index != vision_sample_index:
                raise RuntimeError(
                    f"[r{self.rank}] Sample index mismatch at iteration {iteration}! "
                    f"Vision sample_index={vision_sample_index}, text sample_index={text_sample_index}. "
                    f"Vision and text dataloaders are out of sync!"
                )

            logger.debug(
                f"[r{self.rank}] Text trainer iteration {iteration}: sample_index={text_sample_index} "
                f"(matches vision: {text_sample_index == vision_sample_index})"
            )

        # Move tensors to CUDA for real data path
        device = torch.device("cuda:0")
        if isinstance(batch, dict):
            for key in ("input_ids", "attention_mask", "labels", "position_ids"):
                if key in batch and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        position_ids = batch.get("position_ids")  # Get pre-computed position_ids from dataloader

        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        # Guard 3: Verify position_ids exist and have correct shape
        if position_ids is None:
            raise RuntimeError(
                f"[r{self.rank}] position_ids is None! Dataloader should provide pre-computed position_ids."
            )
        if position_ids.dim() != 3 or position_ids.shape[0] != 3:
            raise RuntimeError(
                f"[r{self.rank}] position_ids should have shape [3, batch_size, seq_len] for 3D RoPE, "
                f"got shape {position_ids.shape}"
            )
        if position_ids.shape[1:] != input_ids.shape:
            raise RuntimeError(
                f"[r{self.rank}] position_ids shape {position_ids.shape} doesn't match input_ids shape {input_ids.shape}. "
                f"Expected position_ids shape: [3, {input_ids.shape[0]}, {input_ids.shape[1]}]"
            )

        # Match original behavior: pass attention_mask=None to let language model handle it
        # The language model will create its own causal mask internally
        attention_mask = None

        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__} forward_step: input_ids shape={input_ids.shape}, "
            f"vision_embeddings shape={vision_embeddings.shape}"
        )

        if vision_embeddings.device != device:
            vision_embeddings = vision_embeddings.to(device, non_blocking=True)

        # Handle vision embeddings shape
        if vision_embeddings.dim() == 2:
            # [num_tokens, hidden_size] -> add batch dimension
            vision_embeddings = vision_embeddings.unsqueeze(0)
        # Now vision_embeddings is [batch_size, num_vision_tokens, hidden_size] or [1, num_tokens, hidden]

        # Enable gradient computation for vision embeddings (needed for backward pass)
        # Detach first so the new tensor is a leaf and gets .grad populated.
        # If this came via CUDA IPC, this also avoids keeping the shared memory alive.
        vision_embeddings = vision_embeddings.detach().clone().requires_grad_(True)
        vision_embeddings.retain_grad()

        # Save for backward pass
        self.vision_embeddings = vision_embeddings

        # Get actual model (unwrap DeepSpeed if needed)
        actual_model = self._get_actual_model()

        # Get text embeddings from input_ids (which contains <image> placeholder tokens)
        inputs_embeds = actual_model.embed_tokens(input_ids)
        # Shape: [batch_size, seq_len, hidden_size]

        # Replace <image> placeholder positions with vision embeddings
        IMAGE_TOKEN_ID = 151655

        # Create mask for image token positions: [batch_size, seq_len, hidden_size]
        image_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)

        # Guard 1: Verify <image> tokens exist in input_ids
        num_image_tokens_in_input = (input_ids == IMAGE_TOKEN_ID).sum().item()
        if num_image_tokens_in_input == 0:
            raise RuntimeError(
                f"[r{self.rank}] No <image> tokens (ID={IMAGE_TOKEN_ID}) found in input_ids! "
                f"This likely means modality filter is wrong. input_ids shape: {input_ids.shape}"
            )

        # Flatten vision embeddings to [total_vision_tokens, hidden_size] for scattering
        vision_embeds_flat = vision_embeddings.reshape(-1, vision_embeddings.shape[-1])

        # Guard 2: Verify vision embeddings count matches image token count
        num_vision_embeds = vision_embeds_flat.shape[0]
        if num_vision_embeds != num_image_tokens_in_input:
            raise RuntimeError(
                f"[r{self.rank}] Mismatch: {num_vision_embeds} vision embeddings but "
                f"{num_image_tokens_in_input} <image> tokens in input_ids! "
                f"Vision and text trainers are out of sync."
            )

        # Guard 4: Verify labels have -100 at image token positions
        image_positions = input_ids == IMAGE_TOKEN_ID
        labels_at_image_positions = labels[image_positions]
        if not torch.all(labels_at_image_positions == -100):
            num_wrong = (labels_at_image_positions != -100).sum().item()
            raise RuntimeError(
                f"[r{self.rank}] Labels should be -100 at <image> positions, but found {num_wrong} "
                f"positions with other values. This indicates incorrect label preprocessing."
            )

        logger.debug(f"[r{self.rank}] Replacing {num_image_tokens_in_input} <image> tokens with vision embeddings")

        # Use masked_scatter to replace image placeholder positions with vision embeddings
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, vision_embeds_flat)

        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__}: inputs_embeds shape={inputs_embeds.shape}, "
            f"position_ids shape={position_ids.shape if position_ids is not None else None}"
        )

        # Get autocast context for mixed precision
        autocast_context = self._get_autocast_context()

        # Forward pass through text model with combined embeddings and position_ids with autocast
        with autocast_context:
            text_outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # Get logits from lm_head
            logits = self.lm_head(text_outputs.last_hidden_state)

        logger.debug(f"[r{self.rank}] {self.__class__.__name__} forward_step: logits shape={logits.shape}")

        # Compute loss for next-token prediction
        # Use vocab_parallel_causal_cross_entropy for tensor parallelism (DTensor) or AutoTP with vocab_parallel
        parallelism = self.config.get("parallelism")
        tp_group = getattr(self.model, "tp_group", None)
        use_tp_loss = tp_group is not None and (
            parallelism == "tensor" or (parallelism == "autotp" and self.config.get("use_vocab_parallel", True))
        )
        if use_tp_loss:
            import torch.distributed as dist

            tp_rank = dist.get_rank(self.model.tp_group)
            tp_world_size = dist.get_world_size(self.model.tp_group)
            loss = vocab_parallel_causal_cross_entropy(
                logits,
                labels,
                self.model.tp_group,
                tp_rank,
                tp_world_size,
                ignore_index=-100,
            )
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Save loss for backward pass
        self.loss = loss

        logger.debug(f"[r{self.rank}] {self.__class__.__name__} forward_step: loss={loss.item():.4f}")

        return loss

    def backward_step(self):
        """
        Run backward pass on the text model.

        Returns:
            Gradients w.r.t. vision embeddings to be passed back to VisionTrainer.
            If CUDA IPC is enabled, returns a TensorTransferRequest dict.
            Otherwise returns the gradient tensor directly.
        """
        logger.debug(f"[r{self.rank}] {self.__class__.__name__} backward_step: computing gradients")

        # Backward pass: compute gradients
        if self.use_deepspeed and self.deepspeed_engine is not None:
            self.deepspeed_engine.backward(self.loss)
        else:
            self.loss.backward()

        # Extract gradients from vision_embeddings leaf tensor
        if self.vision_embeddings is None or self.vision_embeddings.grad is None:
            raise RuntimeError(
                f"[r{self.rank}] vision_embeddings or its gradient is None! "
                f"This should not happen - check that vision_embeddings.requires_grad=True in forward."
            )

        # vision_embeddings has shape [batch_size, num_vision_tokens, hidden_size] or [1, num_tokens, hidden_size]
        # Flatten to [num_vision_tokens, hidden_size] to match what vision trainer expects
        vision_grad = self.vision_embeddings.grad
        if vision_grad.dim() == 3 and vision_grad.shape[0] == 1:
            vision_grad = vision_grad.squeeze(0)  # Remove batch dimension if it's 1

        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__} backward_step: " f"vision_grad shape={vision_grad.shape}"
        )

        # Detach and clone for transfer
        grad_to_send = vision_grad.detach().clone()

        # Clear saved tensors to release memory
        self.loss = None
        self.vision_embeddings = None

        if grad_to_send is None:
            return None

        # Use CUDA IPC if configured and receiver info is available
        if self.use_ipc and self.receiver_gpu_ids is not None:
            sender_gpu_id = get_physical_gpu_id()
            # Detach the gradient tensor before creating IPC handle
            # Gradients are leaf tensors but may still have autograd metadata
            # The VisionTrainer will use it directly in backward() which doesn't need grad tracking
            transfer_request = prepare_tensor_for_transfer(
                grad_to_send,
                receiver_gpu_ids=self.receiver_gpu_ids,
                sender_gpu_id=sender_gpu_id,
                use_ipc_if_same_gpu=True,
            )
            logger.debug(
                f"[r{self.rank}] {self.__class__.__name__}: Created gradient transfer request (use_ipc={transfer_request.use_ipc})"
            )
            return transfer_request.to_dict()
        else:
            # Return gradients directly (Ray's default transport)
            return grad_to_send

    def zero_grad(self):
        """Zero out model and lm_head gradients to free memory."""
        super().zero_grad()
        # Also zero lm_head gradients if optimizer doesn't exist
        if self.optimizer is None and hasattr(self, "lm_head") and self.lm_head is not None:
            for param in self.lm_head.parameters():
                if param.grad is not None:
                    param.grad = None

    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Save text model checkpoint to disk.

        Args:
            checkpoint_dir: Base directory for checkpoints
            epoch: Current epoch number

        Returns:
            Path to saved checkpoint
        """
        import os

        # Create checkpoint directory for this epoch
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}", "text")
        os.makedirs(epoch_dir, exist_ok=True)

        # Get model state dict (handles DeepSpeed engine vs regular model)
        model_state_dict = self._get_model_state_dict_for_checkpoint()

        # Save model checkpoint for this rank
        checkpoint_path = os.path.join(epoch_dir, f"rank_{self.rank}.pt")
        checkpoint_data = {"model": model_state_dict}

        # Note: We don't save lm_head separately because its weights are tied to embeddings
        # The weight tying will be automatically restored when we load the model

        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"[r{self.rank}] Saved text checkpoint to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Load text model checkpoint from disk.

        Args:
            checkpoint_dir: Base directory for checkpoints
            epoch: Epoch number to load from

        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        import os

        # Build checkpoint path for this rank
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}", "text")
        checkpoint_path = os.path.join(epoch_dir, f"rank_{self.rank}.pt")

        if not os.path.exists(checkpoint_path):
            logger.warning(f"[r{self.rank}] Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

            # Load model state dict
            if "model" in checkpoint_data:
                if self.use_deepspeed and self.deepspeed_engine is not None:
                    # Load into DeepSpeed engine's module
                    self.deepspeed_engine.module.load_state_dict(checkpoint_data["model"])
                elif self.model is not None:
                    self.model.load_state_dict(checkpoint_data["model"])

            # Note: Weight tying will automatically reflect in lm_head
            # No need to load lm_head separately

            logger.debug(f"[r{self.rank}] Loaded text checkpoint from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"[r{self.rank}] Failed to load checkpoint from {checkpoint_path}: {e}")
            return False


class QwenTextMixin:
    """Qwen2.5-VL text model helpers shared by Ray and standalone trainers."""

    def _load_model_config(self, model_name):
        """Load Qwen2.5-VL model config."""
        from ..models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

        return Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)

    def _create_model_and_lm_head(self, model_config):
        """Create Qwen2.5-VL text model and lm_head."""
        from ..models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

        model = Qwen2_5_VLTextModel._from_config(model_config.text_config)
        lm_head = nn.Linear(model_config.text_config.hidden_size, model_config.text_config.vocab_size, bias=False)
        return model, lm_head

    def _get_embedding_module(self, model):
        """Get embedding module for Qwen."""
        return model.embed_tokens

    def _get_transformer_layers(self, model):
        """Get transformer layers for Qwen."""
        return model.layers

    def _get_tensor_parallel_mapping(self):
        """Get tensor parallel mapping for Qwen."""
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        return {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=6)
class QwenTextTrainer(QwenTextMixin, BaseTextTrainer):
    """Qwen2.5-VL text trainer."""

    pass
