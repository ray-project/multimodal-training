import logging
import time
from abc import abstractmethod
from collections import deque

import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from .tensor_transfer import (
    TensorTransferRequest,
    prepare_tensor_for_transfer,
    receive_tensor,
)
from .trainer import Trainer
from .utils import get_physical_gpu_id, init_distributed_comm

logger = logging.getLogger(__name__)

# Import DeepSpeed sequence parallel module (will be used when parallelism == "sequence")
try:
    import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
except ImportError:
    mpu = None


class BaseVisionTrainer(Trainer):
    """Base class for vision trainers with common functionality."""

    def __init__(self, config, rank: int):
        super().__init__(config, rank)
        self.receiver_gpu_ids = None
        self.use_ipc = False
        self.model = None
        self.projector = None
        self._pending_outputs: deque[torch.Tensor] = deque()
        self.sp_group = None  # Sequence parallel group (initialized when parallelism == "sequence")
        logger.debug(f"[r{self.rank}] {self.__class__.__name__} initialized")

    @abstractmethod
    def _create_model_instance(self, model_config):
        """
        Create model and optional projector instances.
        Returns: (model, projector) tuple where projector can be None.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _get_projector_or_merger(self, model, projector):
        """
        Get the projector or merger module for parallelization.
        Returns: module to parallelize (or None if not applicable).
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _parallelize_projector_or_merger(self, model, projector, tp_mesh):
        """
        Apply tensor parallelism to projector/merger.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _setup_sequence_parallel(self, model, sp_group):
        """
        Set up sequence parallelism group assignment.
        Must be implemented by subclasses.

        Args:
            model: The vision model instance
            sp_group: The DeepSpeed sequence parallel process group
        """
        pass

    @abstractmethod
    def _get_vision_config(self, model_name):
        """Get vision config for dataset creation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _model_forward(self, batch):
        """Forward pass through model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _zero_padded_weights_after_init(self, model, projector):
        """
        Zero out padded weights after initialize_weights() is called.
        This is needed when using tensor parallelism with meta device initialization.
        Must be implemented by subclasses.
        """
        pass

    def _build_model_architecture(
        self, config, model_name, device, torch_dtype, parallelism, load_pretrained_path=None
    ):
        """
        Common model building logic - template method pattern.
        Subclasses implement specific abstract methods for customization.

        Args:
            load_pretrained_path: Optional path to pretrained checkpoint directory
        """
        # Load model config
        logger.debug(f"[r{self.rank}] Loading vision model config...")
        model_config = self._load_model_config(model_name)

        # Configure attention backend
        attention_backend = config["attention_backend"]
        self._configure_attention_backend(model_config, attention_backend, "vision_config")

        # Handle sequence parallelism
        # Note: Sequence parallelism requires DeepSpeed engine for proper gradient reduction
        # even without ZeRO, because parameter gradients must be reduced across all SP ranks
        if parallelism == "sequence":
            self._configure_sequence_parallel_mode(model_config, True)
            if not dist.is_initialized():
                init_distributed_comm(backend="nccl", use_deepspeed=True)

            # Initialize DeepSpeed sequence parallel groups
            if mpu is None:
                raise ImportError(
                    "DeepSpeed is required for sequence parallelism. " "Please install DeepSpeed: pip install deepspeed"
                )
            world_size = dist.get_world_size()
            sequence_parallel_size = config.get("sequence_parallel_size")
            if sequence_parallel_size is None:
                sequence_parallel_size = world_size
            sequence_parallel_size = int(sequence_parallel_size)
            if sequence_parallel_size <= 1:
                raise ValueError(
                    f"Sequence parallelism requires sequence_parallel_size > 1 (got {sequence_parallel_size})"
                )
            if world_size % sequence_parallel_size != 0:
                raise ValueError(f"sequence_parallel_size {sequence_parallel_size} must divide world size {world_size}")
            data_parallel_size = world_size // sequence_parallel_size
            logger.debug(
                f"[r{self.rank}] Initializing DeepSpeed sequence parallel:"
                f" seq={sequence_parallel_size} data={data_parallel_size} world={world_size}"
            )
            mpu.initialize_sequence_parallel(sequence_parallel_size=sequence_parallel_size)
            self.sp_group = mpu.get_sequence_parallel_group()
            logger.debug(f"[r{self.rank}] Sequence parallel group initialized: {self.sp_group}")

            # Build with DeepSpeed engine to enable gradient reduction and optional ZeRO
            # Pass sp_group so it can be set up after model creation
            return self._build_model_with_deepspeed(
                config,
                model_config,
                device,
                torch_dtype,
                sp_group=self.sp_group,
                mpu_module=mpu,
                load_pretrained_path=load_pretrained_path,
            )

        # Handle DeepSpeed parallelism (ZeRO without sequence parallelism)
        if parallelism == "deepspeed":
            if not dist.is_initialized():
                init_distributed_comm(backend="nccl", use_deepspeed=True)
            return self._build_model_with_deepspeed(
                config, model_config, device, torch_dtype, load_pretrained_path=load_pretrained_path
            )

        # Build model
        torch.set_default_device(device)

        logger.debug(f"[r{self.rank}] Creating vision model...")
        model, projector = self._create_model_instance(model_config)

        torch.set_default_device("cpu")
        model.to(torch_dtype)
        if projector is not None:
            projector.to(torch_dtype)

        # Load pretrained weights BEFORE tensor parallelism (for non-DeepSpeed parallelism)
        if load_pretrained_path and parallelism not in ["sequence", "deepspeed"]:
            import os

            vision_checkpoint_path = os.path.join(load_pretrained_path, "vision_model.pt")
            # Temporarily assign model for loading
            self.model = model
            self.load_pretrained_weights(vision_checkpoint_path)
            model = self.model

        # Activation checkpointing
        activation_checkpointing = config["activation_checkpointing"]
        self._apply_activation_checkpointing(model, activation_checkpointing, "vision model")

        # Tensor parallelism
        if parallelism == "tensor":
            init_distributed_comm(backend="nccl")
            assert dist.is_initialized(), "Failed to initialize distributed communication for tensor parallelism"

            tp_world_size = dist.get_world_size()
            tp_mesh = init_device_mesh("cuda", (tp_world_size,))

            logger.debug(f"[r{self.rank}] Applying tensor parallelism...")

            # Parallelize transformer blocks
            layers = self._get_transformer_layers(model)
            tp_mapping = self._get_tensor_parallel_mapping()
            for layer in layers:
                parallelize_module(layer, tp_mesh, tp_mapping, src_data_rank=None)

            # Parallelize projector/merger
            self._parallelize_projector_or_merger(model, projector, tp_mesh)

            # Zero out padded attention weights (if any)
            self._zero_padded_weights_after_init(model, projector)

        # Final device setup
        model.to(device=device, dtype=torch_dtype)
        if projector is not None:
            projector.to(device=device, dtype=torch_dtype)

        return model, projector

    def _build_model_with_deepspeed(
        self, config, model_config, device, torch_dtype, sp_group=None, mpu_module=None, load_pretrained_path=None
    ):
        """
        Build model with DeepSpeed ZeRO optimization.

        Args:
            config: Training config dict
            model_config: Model configuration
            device: Target device
            torch_dtype: Target dtype
            sp_group: Optional sequence parallel process group (for SP + ZeRO)
            mpu_module: Optional parallel state module to hand off to DeepSpeed
            load_pretrained_path: Optional path to pretrained checkpoint directory

        Returns:
            (model_engine, projector) tuple where model_engine is DeepSpeed engine
        """
        # Build model on target device
        torch.set_default_device(device)
        logger.debug(f"[r{self.rank}] Creating vision model for DeepSpeed...")
        model, projector = self._create_model_instance(model_config)
        torch.set_default_device("cpu")

        model.to(torch_dtype)
        if projector is not None:
            projector.to(torch_dtype)

        # Load pretrained weights BEFORE DeepSpeed wrapping
        if load_pretrained_path:
            import os

            vision_checkpoint_path = os.path.join(load_pretrained_path, "vision_model.pt")
            # Temporarily assign model for loading
            self.model = model
            self.load_pretrained_weights(vision_checkpoint_path)
            model = self.model

        # Apply activation checkpointing
        activation_checkpointing = config["activation_checkpointing"]
        self._apply_activation_checkpointing(model, activation_checkpointing, "vision model")

        # Set up sequence parallelism group if provided
        if sp_group is not None:
            logger.debug(f"[r{self.rank}] Setting sp_group for sequence parallelism in DeepSpeed model")
            self._setup_sequence_parallel(model, sp_group)

        # Collect parameters from model and projector
        params = self._collect_parameters_from_modules(model, projector)

        # Create the optimizer externally to ensure consistency with DTensor path
        # This uses the same _build_optimizer logic (Adam with proper weight decay groups)
        self._build_optimizer(params)
        external_optimizer = self.optimizer

        # Initialize with DeepSpeed using base class method and external optimizer
        model_engine, optimizer, _, _ = self._initialize_deepspeed(
            model=model,
            params=params,
            config=config,
            torch_dtype=torch_dtype,
            mpu=mpu_module,
            optimizer=external_optimizer,
        )

        # Return the engine in place of model (projector stays separate)
        return model_engine, projector

    def _configure_sequence_parallel_mode(self, model_config, enabled):
        """Configure sequence parallel mode - can be overridden by subclasses."""
        if hasattr(model_config, "vision_config"):
            model_config.vision_config.sequence_parallel = enabled
        if hasattr(model_config, "sequence_parallel"):
            model_config.sequence_parallel = enabled

    def load_pretrained_weights(self, checkpoint_path: str):
        """
        Load pretrained weights from a split checkpoint.

        Args:
            checkpoint_path: Path to the vision_model.pt checkpoint file
        """
        import os

        if not os.path.exists(checkpoint_path):
            logger.warning(f"[r{self.rank}] Pretrained checkpoint not found: {checkpoint_path}")
            return False

        logger.debug(f"[r{self.rank}] Loading pretrained vision weights from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)

            # Handle fused QKV weights: HuggingFace checkpoints have fused qkv.weight and qkv.bias
            # but our model expects separate q_proj, k_proj, v_proj
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if "attn.qkv.weight" in key or "attn.qkv.bias" in key:
                    # Split fused QKV into separate Q, K, V projections
                    # qkv shape: [3*hidden_size, ...] where hidden_size is per head
                    is_weight = "weight" in key
                    prefix = key.replace(".qkv.weight", "").replace(".qkv.bias", "")

                    if is_weight:
                        # Shape: [3*hidden, hidden] -> split into 3 parts
                        q, k, v = value.chunk(3, dim=0)
                        cleaned_state_dict[f"{prefix}.q_proj.weight"] = q
                        cleaned_state_dict[f"{prefix}.k_proj.weight"] = k
                        cleaned_state_dict[f"{prefix}.v_proj.weight"] = v
                    else:
                        # Shape: [3*hidden] -> split into 3 parts
                        q, k, v = value.chunk(3, dim=0)
                        cleaned_state_dict[f"{prefix}.q_proj.bias"] = q
                        cleaned_state_dict[f"{prefix}.k_proj.bias"] = k
                        cleaned_state_dict[f"{prefix}.v_proj.bias"] = v
                else:
                    cleaned_state_dict[key] = value

            # Load into model
            if self.use_deepspeed and self.deepspeed_engine is not None:
                # For DeepSpeed, load into the wrapped module
                missing_keys, unexpected_keys = self.deepspeed_engine.module.load_state_dict(
                    cleaned_state_dict, strict=False
                )
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)

            if missing_keys:
                logger.warning(f"[r{self.rank}] Missing keys when loading vision weights: {missing_keys[:10]}")
            if unexpected_keys:
                logger.warning(f"[r{self.rank}] Unexpected keys when loading vision weights: {unexpected_keys[:10]}")

            logger.info(
                f"[r{self.rank}] Successfully loaded pretrained vision weights ({len(cleaned_state_dict)} parameters)"
            )
            return True

        except Exception as e:
            logger.error(f"[r{self.rank}] Failed to load pretrained weights from {checkpoint_path}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def build_model(self):
        """Build vision model - calls subclass-specific architecture builder."""
        model_name = self.config["model_name"]
        parallelism = self.config["parallelism"]
        dtype_str = self.config["dtype"]

        torch_dtype = self._get_torch_dtype(dtype_str)
        device = self._get_device()

        # Check if we need to load pretrained weights
        pretrained_checkpoint_dir = self.config.get("pretrained_checkpoint_dir", None)

        # Call subclass-specific builder (handles pretrained loading internally)
        self.model, self.projector = self._build_model_architecture(
            self.config, model_name, device, torch_dtype, parallelism, load_pretrained_path=pretrained_checkpoint_dir
        )

        # Common finalization
        self.model.train()
        if self.projector is not None:
            self.projector.train()

        logger.debug(f"[r{self.rank}] Vision model built successfully")

        # Build optimizer with both model and projector parameters
        # Skip if using DeepSpeed - optimizer is created in build methods
        if not self.use_deepspeed:
            params = list(self.model.parameters())
            if self.projector is not None:
                params.extend(list(self.projector.parameters()))
            self._build_optimizer(params)
        else:
            logger.debug(f"[r{self.rank}] Using external optimizer with DeepSpeed (already created)")

        # Build LR scheduler for both DeepSpeed and non-DeepSpeed modes
        self._build_scheduler(self.config["num_iterations"])

    def initialize_trainer(self):
        """Initialize the trainer with real dataset via build_dataloader."""
        logger.debug(f"[r{self.rank}] {self.__class__.__name__} initialize_trainer called")

        # Build dataloader using common helper method
        self.dataloader = self._build_dataloader(modality="image")
        self.data_iterator = iter(self.dataloader)

        batch_size = self.config["batch_size"]
        dataset_len = len(self.dataloader.dataset)
        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__} initialized with {dataset_len} samples, batch_size={batch_size}"
        )
        super().initialize_trainer()

    def set_receiver_info(self, receiver_gpu_ids: list[str], use_ipc: bool):
        """Set receiver GPU IDs and whether to use CUDA IPC."""
        self.receiver_gpu_ids = receiver_gpu_ids
        self.use_ipc = use_ipc
        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__}: receiver_gpu_ids={receiver_gpu_ids}, use_ipc={use_ipc}"
        )

    def forward_step(self, iteration: int = -1):
        """Run one forward pass on the vision model.

        Args:
            iteration: Training iteration number for synchronization verification
        """
        profile_time = self.config.get("profile_time", False)
        if profile_time:
            forward_start = time.perf_counter()

        # Store iteration for verification
        self._current_iteration = iteration

        # Get next batch
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            batch = next(self.data_iterator)

        logger.debug(f"[r{self.rank}] Vision forward_step: iteration={iteration}")

        # Extract sample_index for synchronization verification
        sample_index = batch.get("sample_index", None)
        if sample_index is not None and isinstance(sample_index, torch.Tensor):
            sample_index = sample_index.item() if sample_index.numel() == 1 else sample_index.tolist()

        logger.debug(f"[r{self.rank}] Vision forward_step: iteration={iteration}, sample_index={sample_index}")

        # Move tensors to CUDA
        device = torch.device("cuda:0")
        batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
        batch["image_grid_thw"] = batch["image_grid_thw"].to(device, non_blocking=True)

        # Call subclass-specific forward
        vision_outputs = self._model_forward(batch)

        # Synchronize and measure timing only when profiling is enabled
        forward_time_ms = 0.0
        if profile_time:
            torch.cuda.synchronize()
            forward_time_ms = (time.perf_counter() - forward_start) * 1000

        # Enqueue outputs for backward pass (supports multiple outstanding microbatches)
        self._pending_outputs.append(vision_outputs)

        # Return both vision embeddings and metadata for synchronization verification
        result = {
            "vision_embeddings": vision_outputs,
            "sample_index": sample_index,
            "iteration": iteration,
            "forward_time_ms": forward_time_ms,
        }

        # Use CUDA IPC if configured
        if self.use_ipc and self.receiver_gpu_ids is not None:
            sender_gpu_id = get_physical_gpu_id()
            transfer_request = prepare_tensor_for_transfer(
                vision_outputs.detach(),
                receiver_gpu_ids=self.receiver_gpu_ids,
                sender_gpu_id=sender_gpu_id,
                use_ipc_if_same_gpu=True,
            )
            # Wrap transfer request with metadata
            result["vision_embeddings"] = transfer_request.to_dict()
            return result
        else:
            return result

    def _retrieve_gradient_tensor(self, vision_grad_ref):
        """Retrieve gradient tensor from Ray reference or transfer request."""
        if vision_grad_ref is None:
            return None

        if isinstance(vision_grad_ref, ray.ObjectRef):
            vision_grad_data = ray.get(vision_grad_ref)
        else:
            vision_grad_data = vision_grad_ref

        # Handle new dict format from text backward_step: {"grad": ..., "backward_time_ms": ...}
        if isinstance(vision_grad_data, dict) and "grad" in vision_grad_data:
            vision_grad_data = vision_grad_data["grad"]

        if vision_grad_data is None:
            return None

        if isinstance(vision_grad_data, dict) and "use_ipc" in vision_grad_data:
            transfer_request = TensorTransferRequest.from_dict(vision_grad_data)
            receiver_gpu_id = get_physical_gpu_id()
            vision_grad = receive_tensor(transfer_request, receiver_gpu_id)
        else:
            vision_grad = vision_grad_data

        return vision_grad

    def _apply_vision_backward(self, vision_grad: torch.Tensor | None):
        """Apply backward pass on stored outputs using provided gradient tensor."""
        if vision_grad is None:
            raise ValueError(f"[r{self.rank}] No gradient provided for backward pass")

        if not self._pending_outputs:
            raise RuntimeError(
                f"[r{self.rank}] No pending vision outputs available for backward pass. "
                "Ensure forward_step was called before backward."
            )

        vision_outputs = self._pending_outputs.popleft()

        # Match dimensions: handle case where text trainer squeezed the batch dimension
        # vision_outputs is always 3D: [batch_size, num_tokens, hidden_size]
        # vision_grad may be 2D: [num_tokens, hidden_size] if batch_size=1 was squeezed
        if vision_grad.dim() == 2 and vision_outputs.dim() == 3:
            if vision_outputs.shape[0] == 1:
                # Add batch dimension back to match
                vision_grad = vision_grad.unsqueeze(0)
            else:
                raise RuntimeError(
                    f"[r{self.rank}] Dimension mismatch: vision_grad is 2D {vision_grad.shape} "
                    f"but vision_outputs has batch_size={vision_outputs.shape[0]} > 1"
                )
        elif vision_grad.dim() == 3 and vision_outputs.dim() == 2:
            # Legacy case: if somehow output is 2D and grad is 3D
            vision_grad = vision_grad.squeeze(0)

        # Use explicit backward to backpropagate gradients through vision model
        # This is equivalent to vision_outputs.backward(gradient=vision_grad)
        # but more explicit about what we're doing
        # torch.autograd.backward(tensors=[vision_outputs], grad_tensors=[vision_grad], retain_graph=False)
        vision_outputs.backward(gradient=vision_grad, retain_graph=False)

        return None

    def backward_step(self, vision_grad_ref):
        """Run backward pass on the vision model."""
        profile_time = self.config.get("profile_time", False)
        if profile_time:
            backward_start = time.perf_counter()

        vision_grad = self._retrieve_gradient_tensor(vision_grad_ref)
        self._apply_vision_backward(vision_grad)

        # Synchronize and measure timing only when profiling is enabled
        backward_time_ms = 0.0
        if profile_time:
            torch.cuda.synchronize()
            backward_time_ms = (time.perf_counter() - backward_start) * 1000

        return {"backward_time_ms": backward_time_ms}

    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Save vision model checkpoint to disk.

        Args:
            checkpoint_dir: Base directory for checkpoints
            epoch: Current epoch number

        Returns:
            Path to saved checkpoint
        """
        import os

        # Create checkpoint directory for this epoch
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}", "vision")
        os.makedirs(epoch_dir, exist_ok=True)

        # Get model state dict (handles DeepSpeed engine vs regular model)
        model_state_dict = self._get_model_state_dict_for_checkpoint()

        # Save model checkpoint for this rank
        checkpoint_path = os.path.join(epoch_dir, f"rank_{self.rank}.pt")
        checkpoint_data = {"model": model_state_dict}

        # If projector exists, save it (only for rank 0 since it's not sharded)
        if self.projector is not None and self.rank == 0:
            checkpoint_data["projector"] = self.projector.state_dict()

        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"[r{self.rank}] Saved vision checkpoint to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Load vision model checkpoint from disk.

        Args:
            checkpoint_dir: Base directory for checkpoints
            epoch: Epoch number to load from

        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        import os

        # Build checkpoint path for this rank
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}", "vision")
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

            # Load projector if it exists (only for rank 0)
            if "projector" in checkpoint_data and self.projector is not None and self.rank == 0:
                self.projector.load_state_dict(checkpoint_data["projector"])

            logger.debug(f"[r{self.rank}] Loaded vision checkpoint from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"[r{self.rank}] Failed to load checkpoint from {checkpoint_path}: {e}")
            return False


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=6)
class QwenVisionTrainer(BaseVisionTrainer):
    """Qwen2.5-VL vision trainer."""

    def _load_model_config(self, model_name):
        """Load Qwen2.5-VL model config."""
        from ..models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

        return Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)

    def _create_model_instance(self, model_config):
        """Create Qwen2.5-VL vision model instance."""
        from ..models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        model = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
        return model, None  # No projector for Qwen

    def _get_transformer_layers(self, model):
        """Get transformer blocks for Qwen."""
        return model.blocks

    def _get_projector_or_merger(self, model, projector):
        """Get merger module for Qwen."""
        return model.merger

    def _get_tensor_parallel_mapping(self):
        """Get tensor parallel mapping for Qwen transformer layers."""
        return {
            "attn.q_proj": ColwiseParallel(),
            "attn.k_proj": ColwiseParallel(),
            "attn.v_proj": ColwiseParallel(),
            "attn.proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

    def _parallelize_projector_or_merger(self, model, projector, tp_mesh):
        """Parallelize Qwen merger MLP."""
        parallelize_module(model.merger.mlp[0], tp_mesh, ColwiseParallel(), src_data_rank=None)
        parallelize_module(model.merger.mlp[2], tp_mesh, RowwiseParallel(), src_data_rank=None)

    def _setup_sequence_parallel(self, model, sp_group):
        """Set up sequence parallelism for Qwen.

        Args:
            model: The Qwen vision model instance
            sp_group: The DeepSpeed sequence parallel process group
        """
        for m in model.modules():
            if m.__class__.__name__ in ["Qwen2_5_VLVisionAttention", "Qwen2_5_VisionTransformerPretrainedModel"]:
                m.sp_group = sp_group

    def _get_vision_config(self, model_name):
        """Get Qwen2.5-VL vision config."""
        from ..models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

        config = Qwen2_5_VLConfig.from_pretrained(model_name, trust_remote_code=True)
        return config.vision_config

    def _model_forward(self, batch):
        """Forward pass for Qwen2.5-VL."""
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]

        # Get autocast context for mixed precision
        autocast_context = self._get_autocast_context()

        # Handle batched inputs - process entire batch at once
        # DataLoader returns [batch_size, C, T, H, W] when batch_size > 1
        if pixel_values.dim() == 5:  # [batch_size, C, T, H, W]
            batch_size = pixel_values.shape[0]

            # Ensure image_grid_thw has shape [batch_size, 3]
            if image_grid_thw.dim() == 1:
                # Single grid_thw broadcasted to all samples in batch
                image_grid_thw = image_grid_thw.unsqueeze(0).expand(batch_size, -1)
            elif image_grid_thw.dim() == 2 and image_grid_thw.shape[0] != batch_size:
                # Handle edge case where dimensions don't match
                raise ValueError(
                    f"image_grid_thw batch dimension {image_grid_thw.shape[0]} doesn't match pixel_values batch dimension {batch_size}"
                )

            logger.debug(
                f"[r{self.rank}] Qwen forward batched: pixel_values={pixel_values.shape}, grid_thw={image_grid_thw.shape}"
            )

            # Process entire batch at once with autocast
            # NOTE: Qwen model concatenates all images along sequence dimension
            # Output shape: [batch_size * num_tokens_per_image, hidden_size]
            with autocast_context:
                vision_outputs = self.model(hidden_states=pixel_values, grid_thw=image_grid_thw)

            # Reshape to [batch_size, num_tokens_per_image, hidden_size]
            # The model processes all images as a single sequence, so we need to split back into batches
            total_tokens = vision_outputs.shape[0]
            num_tokens_per_image = total_tokens // batch_size
            vision_outputs = vision_outputs.reshape(batch_size, num_tokens_per_image, -1)

            logger.debug(
                f"[r{self.rank}] Qwen forward batched output shape={vision_outputs.shape} (reshaped from {total_tokens} total tokens)"
            )
        else:
            # Single sample case (batch_size=1 or unbatched)
            # pixel_values is [C, T, H, W], need grid_thw to be [1, 3]
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)

            logger.debug(
                f"[r{self.rank}] Qwen forward single: pixel_values={pixel_values.shape}, grid_thw={image_grid_thw.shape}"
            )

            # Model returns [num_tokens, hidden_size] for single image with autocast
            with autocast_context:
                vision_outputs = self.model(hidden_states=pixel_values, grid_thw=image_grid_thw)

            # Add batch dimension for consistency: [num_tokens, hidden_size] -> [1, num_tokens, hidden_size]
            vision_outputs = vision_outputs.unsqueeze(0)
            logger.debug(f"[r{self.rank}] Qwen forward single output shape={vision_outputs.shape}")

        return vision_outputs

    def _zero_padded_weights_after_init(self, model, projector):
        """Qwen does not use padded attention heads, so this is a no-op."""
        pass


# Keep VisionTrainer as alias to QwenVisionTrainer for backwards compatibility
VisionTrainer = QwenVisionTrainer
