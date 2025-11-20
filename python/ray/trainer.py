import logging
from abc import abstractmethod
from contextlib import nullcontext

import torch
from transformers import set_seed

from .actor import RayActor

logger = logging.getLogger(__name__)


class Trainer(RayActor):

    optimizer = None
    use_deepspeed = False
    deepspeed_engine = None

    def __init__(self, config, rank: int):
        RayActor.__init__(self, rank)
        self.config = config

        seed = config["seed"]
        set_seed(seed)

    def initialize_trainer(self):
        logger.debug(f"Initializing trainer {self.__class__.__name__}")

    def _build_dataloader(self, modality: str):
        """
        Build dataloader with common logic shared between vision and text trainers.

        Args:
            modality: Modality filter for the dataloader ("image", "all", etc.)
        """
        from transformers import AutoProcessor

        from ..data.config import DataConfig
        from ..data.dataset import build_dataloader

        model_name = self.config["model_name"]

        # Build processor
        min_pixels = self.config["min_pixels"]
        max_pixels = self.config["max_pixels"]
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Build DataConfig
        datasets = self.config["datasets"]
        force_fixed_size = self.config["force_fixed_size"]
        data_registry = self.config["data_registry"]
        data_cfg = DataConfig(
            datasets=datasets,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            force_fixed_size=force_fixed_size,
            data_registry=data_registry,
        )

        # Get training parameters
        parallelism = self.config["parallelism"]
        batch_size = self.config["batch_size"]
        seed = self.config["seed"]

        # Fixed rank/world_size for now
        rank = 0
        world_size = 1

        # Setup reproducibility for tensor/sequence parallelism
        generator = None
        worker_init_fn = None
        if parallelism in ["tensor", "sequence", "autotp"]:
            import random as _rnd

            import numpy as _np

            def _worker_init_fn(worker_id: int):
                s = seed + worker_id
                _np.random.seed(s)
                _rnd.seed(s)
                torch.manual_seed(s)

            generator = torch.Generator(device="cpu").manual_seed(seed)
            worker_init_fn = _worker_init_fn
        elif parallelism not in ["deepspeed"]:
            raise ValueError(f"[r{self.rank}] Unsupported parallelism mode: {parallelism}")

        # Build dataloader
        dataloader = build_dataloader(
            processor=processor,
            data_config=data_cfg,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            generator=generator,
            worker_init_fn=worker_init_fn,
            drop_last=True,
            modality=modality,
            seed=seed,
        )

        return dataloader

    def build_model(self):
        pass

    def forward_step(
        self,
        first_microbatch: bool = True,
    ):
        pass

    def backward_step(
        self,
        zero_grad: bool = True,
    ):
        pass

    def _clip_gradients(self, global_norm: float, max_norm: float) -> float:
        """
        Clip gradients using the global gradient norm.

        Args:
            global_norm: Global gradient norm across all parameters in all actors
            max_norm: Maximum allowed gradient norm

        Returns:
            Clipping coefficient applied (1.0 if no clipping)
        """
        clip_coeff = max_norm / (global_norm + 1e-6)
        clip_coeff = min(1.0, clip_coeff)

        if clip_coeff < 1.0:
            # Scale all parameter gradients by clipping coefficient
            params_to_clip = []

            if self.use_deepspeed:
                params_to_clip.extend(self.deepspeed_engine.parameters())
            else:
                if hasattr(self, "model") and self.model is not None:
                    params_to_clip.extend(self.model.parameters())
                # Also clip projector/lm_head if present
                if hasattr(self, "projector") and self.projector is not None:
                    params_to_clip.extend(self.projector.parameters())
                if hasattr(self, "lm_head") and self.lm_head is not None:
                    params_to_clip.extend(self.lm_head.parameters())

            for param in params_to_clip:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coeff)

        return clip_coeff

    def optimizer_step(
        self,
        global_grad_norm: float | None = None,
    ) -> float:
        """
        Step the optimizer with optional gradient clipping, then step the LR scheduler.

        Args:
            global_grad_norm: Global gradient norm for clipping (computed across all actors)
                            If provided, gradients will be clipped before optimizer step.

        Returns:
            Clipping coefficient (1.0 if no clipping applied)
        """
        clip_coeff = 1.0

        # Apply gradient clipping if global norm is provided
        if global_grad_norm is not None:
            max_norm = self.config.get("max_grad_norm", 1.0)
            clip_coeff = self._clip_gradients(global_grad_norm, max_norm)

        # Step optimizer
        if self.use_deepspeed:
            self.deepspeed_engine.step()
        elif self.optimizer is not None:
            self.optimizer.step()

        # Step LR scheduler after optimizer
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()

        return clip_coeff

    def train(self):
        # mock training loop
        for i in range(3):
            loss = self.forward_step()
            self.backward_step(loss)
            self.optimizer_step()

    # Common utility methods
    def _get_torch_dtype(self, dtype_str: str = "bfloat16") -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _get_autocast_context(self):
        """Get autocast context manager for mixed precision training.

        Reads dtype and autocast settings from self.config and returns the appropriate
        context manager.

        Returns:
            torch.autocast context manager if mixed precision is enabled,
            nullcontext() otherwise

        Note:
            When using DeepSpeed, autocast should be disabled as DeepSpeed handles
            mixed precision internally via fp16/bf16 config.
        """
        # Don't apply autocast when using DeepSpeed - it handles mixed precision internally
        if self.use_deepspeed:
            return nullcontext()

        # Get config values
        dtype_str = self.config["dtype"]
        autocast_enabled = self.config.get("autocast", True)
        torch_dtype = self._get_torch_dtype(dtype_str)

        # Apply autocast for non-DeepSpeed training with mixed precision if enabled
        if autocast_enabled and torch_dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=True)
        return nullcontext()

    def _get_device(self) -> torch.device:
        """Get the device for this actor (Ray sets CUDA_VISIBLE_DEVICES)."""
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device

    def _configure_attention_backend(self, config, attention_backend: str, config_attr: str = "vision_config"):
        """Configure attention backend for a model config.

        Args:
            config: Model configuration object
            attention_backend: Attention implementation to use (e.g., "sdpa", "flash_attention_2")
            config_attr: Name of the config attribute containing the subconfig (e.g., "vision_config", "text_config")
        """
        if hasattr(config, config_attr) and getattr(config, config_attr) is not None:
            subconfig = getattr(config, config_attr)
            if hasattr(subconfig, "attn_implementation"):
                subconfig.attn_implementation = attention_backend
            if hasattr(subconfig, "_attn_implementation"):
                subconfig._attn_implementation = attention_backend

    def _apply_activation_checkpointing(self, model, activation_checkpointing: bool, model_name: str = "model"):
        """Enable activation checkpointing on a model if configured.

        Args:
            model: The model to apply checkpointing to
            activation_checkpointing: Whether to enable activation checkpointing
            model_name: Name of the model for logging
        """
        if activation_checkpointing:
            model.gradient_checkpointing_enable()
            logger.debug(f"[r{self.rank}] Activation checkpointing enabled for {model_name}")

    def _reset_initialized_flag(self, model):
        """Reset HuggingFace _is_hf_initialized flag for weight initialization."""

        def _reset_flag(mod):
            if hasattr(mod, "_is_hf_initialized"):
                mod._is_hf_initialized = False

        model.apply(_reset_flag)

    def zero_grad(self):
        """Zero out model gradients to free memory. Override in subclasses if needed."""
        if self.use_deepspeed:
            # DeepSpeed handles gradient zeroing internally
            return

        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            return

        if hasattr(self, "model") and self.model is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None

    def _compute_grad_norm_squared(self, params):
        """
        Compute sum of squared gradient norms for a list of parameters.

        Args:
            params: Iterable of parameters

        Returns:
            Sum of ||grad||^2 for all parameters with gradients
        """
        norm_sq = 0.0
        for param in params:
            if param.grad is not None:
                param_norm = param.grad.data.float().norm(2)
                norm_sq += param_norm.item() ** 2
        return norm_sq

    def _is_sharded_parameter(self, param):
        """
        Check if a parameter is sharded across tensor parallel ranks using DTensor placement.

        Args:
            param: Parameter to check

        Returns:
            True if parameter is sharded, False otherwise
        """
        try:
            from torch.distributed.tensor import DTensor, Shard

            if isinstance(param, DTensor):
                # Check if any dimension has Shard placement
                for placement in param.placements:
                    if isinstance(placement, Shard):
                        return True
        except ImportError:
            # DTensor not available, assume not sharded
            pass

        return False

    def compute_grad_norm_contribution(self):
        """
        Compute this actor's contribution to the global gradient norm.

        Returns a dict with the local contribution to global gradient norm squared.
        The format depends on the parallelism strategy:

        - For sequence parallelism: {"type": "sequence", "norm_sq": X}
          All actors have the same gradient after backward, so only count once.

        - For tensor parallelism: {"type": "tensor", "replicated_norm_sq": X, "sharded_norm_sq": Y}
          Separate replicated (same across ranks) and sharded (different per rank) parameters.

        - For DeepSpeed with sequence parallelism: same as sequence parallelism
        - For DeepSpeed without sequence parallelism: same as tensor parallelism

        Returns:
            dict: Gradient norm contribution
        """
        parallelism = self.config.get("parallelism", "none")

        # Collect all parameters from model and any auxiliary modules
        all_params = []
        if self.use_deepspeed and self.deepspeed_engine is not None:
            all_params = list(self.deepspeed_engine.parameters())
        else:
            if hasattr(self, "model") and self.model is not None:
                all_params.extend(self.model.parameters())
            if hasattr(self, "projector") and self.projector is not None:
                all_params.extend(self.projector.parameters())
            if hasattr(self, "lm_head") and self.lm_head is not None:
                all_params.extend(self.lm_head.parameters())

        if parallelism == "sequence":
            # Sequence parallelism: gradients are synchronized via DeepSpeed
            # All ranks have the same gradient, so just return local norm
            norm_sq = self._compute_grad_norm_squared(all_params)
            return {"type": "sequence", "norm_sq": norm_sq}

        elif parallelism == "tensor":
            # Tensor parallelism: separate sharded and replicated parameters
            replicated_params = []
            sharded_params = []

            for param in all_params:
                if self._is_sharded_parameter(param):
                    sharded_params.append(param)
                else:
                    replicated_params.append(param)

            replicated_norm_sq = self._compute_grad_norm_squared(replicated_params)
            sharded_norm_sq_local = self._compute_grad_norm_squared(sharded_params)

            # For sharded parameters, we need to allreduce norm_sq across TP group
            import torch.distributed as dist

            if dist.is_initialized():
                # Get TP group if available
                tp_group = getattr(self.model, "tp_group", None)
                if tp_group is not None and dist.get_world_size(tp_group) > 1:
                    # Allreduce sharded norm squared across TP group
                    sharded_tensor = torch.tensor([sharded_norm_sq_local], dtype=torch.float32, device="cuda")
                    dist.all_reduce(sharded_tensor, group=tp_group)
                    sharded_norm_sq = sharded_tensor.item()
                else:
                    sharded_norm_sq = sharded_norm_sq_local
            else:
                sharded_norm_sq = sharded_norm_sq_local

            return {
                "type": "tensor",
                "replicated_norm_sq": replicated_norm_sq,
                "sharded_norm_sq": sharded_norm_sq,
            }

        elif parallelism == "autotp":
            # AutoTP uses DeepSpeed-managed tensor parallel groups; treat like DeepSpeed for clipping
            norm_sq = self._compute_grad_norm_squared(all_params)
            return {"type": "deepspeed", "norm_sq": norm_sq}

        elif parallelism == "deepspeed":
            # DeepSpeed: check if sequence parallelism is enabled
            # If sequence parallelism is enabled, treat as sequence parallelism
            # Otherwise, we need to aggregate across all ranks

            # Check if this is DeepSpeed with sequence parallelism
            if hasattr(self, "sp_group") and self.sp_group is not None:
                # DeepSpeed with sequence parallelism
                norm_sq = self._compute_grad_norm_squared(all_params)
                return {"type": "sequence", "norm_sq": norm_sq}
            else:
                # DeepSpeed without sequence parallelism (just ZeRO)
                # For ZeRO, gradients might be sharded depending on stage
                # For simplicity, compute local norm and let main trainer aggregate
                norm_sq = self._compute_grad_norm_squared(all_params)
                return {"type": "deepspeed", "norm_sq": norm_sq}

        else:
            # No parallelism or unknown: just compute local norm
            norm_sq = self._compute_grad_norm_squared(all_params)
            return {"type": "none", "norm_sq": norm_sq}

    def _collect_parameters_from_modules(self, *modules):
        """
        Collect unique parameters from multiple modules.

        Args:
            *modules: Variable number of modules (can be None)

        Returns:
            list: List of unique parameters from all modules
        """
        params = []
        seen = set()

        for module in modules:
            if module is not None:
                for param in module.parameters():
                    if id(param) not in seen:
                        params.append(param)
                        seen.add(id(param))

        return params

    def _get_decay_parameter_names(self, params):
        """
        Get names of parameters that should have weight decay applied.
        Excludes LayerNorm, bias, and other parameters that typically don't benefit from weight decay.

        Args:
            params: Iterable of (name, parameter) tuples

        Returns:
            Set of parameter names that should have weight decay
        """
        decay_params = set()
        for name, param in params:
            if not param.requires_grad:
                continue
            # Exclude bias, LayerNorm, RMSNorm, and similar parameters
            if any(exclude in name.lower() for exclude in ["bias", "norm", "ln_"]):
                continue
            decay_params.add(name)
        return decay_params

    def _build_optimizer(self, params, learning_rate: float | None = None):
        """
        Create an Adam optimizer with parameter grouping for weight decay.

        Args:
            params: Iterable of parameters to optimize
            learning_rate: Learning rate (defaults to config value if None)
        """
        # Convert params to list and deduplicate
        param_list = list(params)
        if not param_list:
            self.optimizer = None
            return

        # Get learning rate from config if not provided
        if learning_rate is None:
            learning_rate = self.config["learning_rate"]

        # Get weight decay from config
        weight_decay = self.config["weight_decay"]

        # If weight_decay is 0, use simple optimizer without parameter grouping
        if weight_decay == 0.0:
            unique_params = list({id(p): p for p in param_list}.values())
            self.optimizer = torch.optim.Adam(
                unique_params,
                lr=learning_rate,
                foreach=False,
            )
            return

        # For weight decay grouping, we need to get named parameters from modules
        # Collect named parameters from model, projector, and lm_head attributes
        named_params = []
        seen_ids = set()

        # Collect from available modules (model, projector, lm_head)
        for module_name in ["model", "projector", "lm_head"]:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    for name, param in module.named_parameters():
                        if id(param) not in seen_ids:
                            # Prefix with module name for clarity
                            full_name = f"{module_name}.{name}"
                            named_params.append((full_name, param))
                            seen_ids.add(id(param))

        # If we couldn't get named parameters (shouldn't happen), fall back to simple optimizer
        if not named_params:
            logger.warning(
                f"[r{self.rank}] Could not extract named parameters for weight decay grouping, "
                "using simple optimizer"
            )
            unique_params = list({id(p): p for p in param_list}.values())
            self.optimizer = torch.optim.Adam(
                unique_params,
                lr=learning_rate,
                foreach=False,
            )
            return

        # Get parameter names that should have decay
        decay_param_names = self._get_decay_parameter_names(named_params)

        # Split parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            if name in decay_param_names:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        # Create optimizer with parameter groups
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            foreach=False,
        )

        logger.debug(
            f"[r{self.rank}] Optimizer created: {len(decay_params)} params with decay, "
            f"{len(no_decay_params)} params without decay (weight_decay={weight_decay})"
        )

    def _build_scheduler(self, num_training_steps: int):
        """
        Create learning rate scheduler using transformers.get_scheduler().
        Works with both regular optimizer and DeepSpeed optimizer.

        Args:
            num_training_steps: Total number of training steps
        """
        # Get the optimizer - either from self.optimizer or from DeepSpeed engine
        if self.use_deepspeed and self.deepspeed_engine is not None:
            # DeepSpeed's optimizer might be wrapped, access the underlying optimizer
            ds_optimizer = self.deepspeed_engine.optimizer
            # If the DeepSpeed optimizer has an 'optimizer' attribute, use it (unwrap)
            optimizer = getattr(ds_optimizer, "optimizer", ds_optimizer)
        elif self.optimizer is not None:
            optimizer = self.optimizer
        else:
            self.scheduler = None
            logger.debug(f"[r{self.rank}] No optimizer, skipping scheduler creation")
            return

        from transformers import get_scheduler

        # Calculate warmup steps - warmup_ratio takes precedence over warmup_steps
        warmup_ratio = self.config["warmup_ratio"]
        if warmup_ratio > 0:
            num_warmup_steps = int(num_training_steps * warmup_ratio)
        else:
            num_warmup_steps = self.config["warmup_steps"]

        # Get scheduler type from config
        lr_scheduler_type = self.config["lr_scheduler_type"]

        # Create scheduler
        self.scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.debug(
            f"[r{self.rank}] LR Scheduler created: type={lr_scheduler_type}, "
            f"warmup_steps={num_warmup_steps}, total_steps={num_training_steps}"
        )

    def _initialize_deepspeed(self, model, params, config, torch_dtype, mpu=None, tensor_parallel_config=None):
        """
        Initialize DeepSpeed engine with the given model and parameters.

        Args:
            model: The model to wrap with DeepSpeed
            params: List of parameters to optimize
            config: Training config dict
            torch_dtype: Target dtype (torch.float16 or torch.bfloat16)
            mpu: Optional model/sequence parallel state module to pass into DeepSpeed
            tensor_parallel_config: Optional tensor parallel configuration (e.g., AutoTP settings)

        Returns:
            tuple: (model_engine, optimizer, _, _) from deepspeed.initialize
        """
        import deepspeed
        import torch.distributed as dist

        # Get DeepSpeed config parameters
        zero_stage = config["zero_stage"]
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        autocast_enabled = config.get("autocast", True)

        # All-reduce happens over the data-parallel group. Track total, DP, and SP sizes separately.
        if dist.is_initialized():
            total_world_size = dist.get_world_size()
        else:
            total_world_size = 1

        sequence_parallel_world_size = 1
        data_parallel_size = total_world_size

        if mpu is not None and hasattr(mpu, "get_sequence_parallel_world_size"):
            sequence_parallel_world_size = mpu.get_sequence_parallel_world_size()
            if sequence_parallel_world_size <= 0:
                raise ValueError("Invalid sequence_parallel_world_size reported by DeepSpeed mpu module")
            if total_world_size % sequence_parallel_world_size != 0:
                raise ValueError(
                    f"Total world size {total_world_size} must be divisible by sequence_parallel_size "
                    f"{sequence_parallel_world_size}"
                )
            data_parallel_size = max(1, total_world_size // sequence_parallel_world_size)
        else:
            sequence_parallel_world_size = 1
            data_parallel_size = total_world_size

        # Handle tensor parallel (AutoTP) world size for correct DP sizing
        tp_world_size = None
        if tensor_parallel_config and "autotp_size" in tensor_parallel_config:
            tp_world_size = int(tensor_parallel_config["autotp_size"])
            if tp_world_size <= 0:
                raise ValueError(f"autotp_size must be > 0 (got {tp_world_size})")
            if total_world_size % tp_world_size != 0:
                raise ValueError(
                    f"Total world size {total_world_size} must be divisible by autotp_size {tp_world_size}"
                )
            # AutoTP splits the remaining ranks into data parallel groups
            data_parallel_size = max(1, total_world_size // tp_world_size)
            if sequence_parallel_world_size > 1:
                raise ValueError("Sequence parallelism together with AutoTP is not supported in this trainer")

        # Calculate global batch size
        gradient_accumulation_steps = config["gradient_accumulation_steps"]
        if tensor_parallel_config:
            # DeepSpeed expects train_batch_size = micro_batch * grad_acc * total_world_size
            train_batch_size = batch_size * total_world_size * gradient_accumulation_steps
        else:
            train_batch_size = batch_size * data_parallel_size * gradient_accumulation_steps

        # Get reduce_bucket_size from config
        reduce_bucket_size = config["reduce_bucket_size"]

        # Configure DeepSpeed fp16/bf16/autocast based on dtype and autocast setting
        # When autocast is enabled, we use torch_autocast and disable fp16/bf16
        # When autocast is disabled, we use DeepSpeed's native fp16/bf16
        use_torch_autocast = autocast_enabled and torch_dtype != torch.float32

        ds_config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_bucket_size": reduce_bucket_size,
            },
            "fp16": {
                "enabled": torch_dtype == torch.float16,
            },
            "bf16": {
                "enabled": torch_dtype == torch.bfloat16,
                "bf16_master_weights_and_grads": True,
                "bf16_optimizer_states": True,
                },
            "torch_autocast": {
                "enabled": use_torch_autocast,
                "dtype": str(torch_dtype),
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": learning_rate,
                    "weight_decay": config["weight_decay"],
                },
            },
            "zero_allow_untested_optimizer": True,
        }

        if sequence_parallel_world_size > 1:
            ds_config.setdefault("sequence_parallel_size", sequence_parallel_world_size)
            ds_config.setdefault("data_parallel_size", data_parallel_size)

        if torch_dtype == torch.float16:
            ds_config.setdefault("communication_data_type", "fp16")
            if sequence_parallel_world_size > 1:
                ds_config.setdefault("seq_parallel_communication_data_type", "fp16")
        elif torch_dtype == torch.bfloat16:
            ds_config.setdefault("communication_data_type", "bf16")
            if sequence_parallel_world_size > 1:
                ds_config.setdefault("seq_parallel_communication_data_type", "bf16")

        if tensor_parallel_config:
            ds_config["tensor_parallel"] = tensor_parallel_config
            ds_config.setdefault("data_parallel_size", data_parallel_size)

        rank = getattr(self, "rank", 0)
        if sequence_parallel_world_size > 1:
            logger.debug(
                f"[r{rank}] DeepSpeed sequence parallel config: seq={sequence_parallel_world_size}, "
                f"data={data_parallel_size}, total_world={total_world_size}"
            )
        logger.debug(f"[r{rank}] Initializing DeepSpeed with ZeRO stage {zero_stage}...")

        # Initialize with DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=params,
            config=ds_config,
            mpu=mpu,
        )

        logger.debug(f"[r{rank}] DeepSpeed initialization complete")

        # Mark that we're using DeepSpeed
        self.use_deepspeed = True
        self.deepspeed_engine = model_engine

        return model_engine, optimizer, None, None

    # Abstract methods for model architecture (implemented by subclasses)
    @abstractmethod
    def _load_model_config(self, model_name):
        """
        Load and return model config.
        Must be implemented by subclasses.

        Args:
            model_name: Name or path of the model

        Returns:
            Model configuration object
        """
        pass

    @abstractmethod
    def _get_transformer_layers(self, model):
        """
        Get the transformer layers for parallelization.
        Must be implemented by subclasses.

        Args:
            model: The model instance

        Returns:
            List of transformer layers
        """
        pass

    @abstractmethod
    def _get_tensor_parallel_mapping(self):
        """
        Get tensor parallel layer mapping for this model.
        Must be implemented by subclasses.

        Returns:
            dict mapping layer names to parallelization strategies
        """
        pass

    # Checkpoint methods (abstract - implemented by subclasses)
    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Save model checkpoint to disk.
        Must be implemented by subclasses.

        Args:
            checkpoint_dir: Base directory for checkpoints
            epoch: Current epoch number

        Returns:
            Path to saved checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: str, epoch: int):
        """
        Load model checkpoint from disk.
        Must be implemented by subclasses.

        Args:
            checkpoint_dir: Base directory for checkpoints
            epoch: Epoch number to load from

        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        pass

    def _get_model_state_dict_for_checkpoint(self):
        """
        Helper to extract state dict based on parallelism mode.

        Returns:
            State dict for checkpointing
        """
        if self.use_deepspeed and self.deepspeed_engine is not None:
            # For DeepSpeed, get the unwrapped module's state dict
            return self.deepspeed_engine.module.state_dict()
        elif hasattr(self, "model") and self.model is not None:
            return self.model.state_dict()
        else:
            return {}
