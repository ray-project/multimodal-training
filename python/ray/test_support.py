from __future__ import annotations

from types import SimpleNamespace

import ray
import torch
import torch.nn as nn

from .payloads import VisionOutputs
from .text import BaseTextTrainer
from .vision import BaseVisionTrainer


class TinyVisionModel(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(1, hidden_size, bias=False)

    def forward(self, pixel_values, image_grid_thw=None):
        batch_size = pixel_values.shape[0]
        dummy = torch.ones(
            batch_size,
            self.num_tokens,
            1,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        return self.proj(dummy)


class TinyTextModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, image_token_id: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=True, image_token_id=image_token_id)

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None):
        hidden = self.linear(inputs_embeds)
        return SimpleNamespace(last_hidden_state=hidden)


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=2)
class TinyVisionTrainer(BaseVisionTrainer):
    def _load_model_config(self, model_name):
        vision_config = SimpleNamespace(attn_implementation=None, sequence_parallel=False)
        return SimpleNamespace(vision_config=vision_config, sequence_parallel=False)

    def _create_model_instance(self, model_config):
        hidden_size = self.config["hidden_size"]
        num_tokens = self.config["vision_tokens"]
        model = TinyVisionModel(hidden_size=hidden_size, num_tokens=num_tokens)
        return model, None

    def _get_transformer_layers(self, model):
        return []

    def _get_projector_or_merger(self, model, projector):
        return None

    def _parallelize_projector_or_merger(self, model, projector, tp_mesh):
        return None

    def _setup_sequence_parallel(self, model, sp_group):
        return None

    def _get_vision_config(self, model_name):
        return SimpleNamespace()

    def _model_forward(self, batch):
        return self.model(batch["pixel_values"], batch["image_grid_thw"])

    def _zero_padded_weights_after_init(self, model, projector):
        return None

    def initialize_trainer(self):
        batch = {
            "pixel_values": torch.zeros(1, 1, 1, 1, 1),
            "image_grid_thw": torch.tensor([1, 1, 1], dtype=torch.long),
            "sample_index": torch.tensor([0], dtype=torch.long),
        }
        self.dataloader = [batch]
        self.data_iterator = iter(self.dataloader)

    def forward_step_no_return(self, iteration: int = -1):
        result = super().forward_step(iteration)
        if isinstance(result, VisionOutputs):
            return VisionOutputs(embeddings=None, attention_mask=None, meta=result.meta)
        if isinstance(result, dict):
            result.pop("vision_embeddings", None)
        return result

    def is_process_group_initialized(self):
        import torch.distributed as dist

        return dist.is_initialized()

    def backward_step_with_dummy_grad(self):
        if not self._pending_outputs:
            raise RuntimeError("No pending vision outputs for dummy backward.")
        vision_outputs = self._pending_outputs[0]
        dummy_grad = torch.ones_like(vision_outputs)
        self._apply_vision_backward(dummy_grad)
        return {"backward_time_ms": 0.0}


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=2)
class TinyTextTrainer(BaseTextTrainer):
    def _load_model_config(self, model_name):
        text_config = SimpleNamespace(attn_implementation=None, num_key_value_heads=1)
        return SimpleNamespace(text_config=text_config, image_token_id=self.config["image_token_id"])

    def _create_model_and_lm_head(self, model_config):
        vocab_size = self.config["vocab_size"]
        hidden_size = self.config["hidden_size"]
        model = TinyTextModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            image_token_id=model_config.image_token_id,
        )
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        return model, lm_head

    def _get_embedding_module(self, model):
        return model.embed_tokens

    def _get_transformer_layers(self, model):
        return []

    def _get_tensor_parallel_mapping(self):
        return {}

    def initialize_trainer(self):
        image_token_id = self.config["image_token_id"]
        seq_len = self.config["text_seq_len"]
        input_ids = torch.tensor([image_token_id, image_token_id, 3, 4][:seq_len], dtype=torch.long)
        labels = torch.tensor([-100, -100, 5, 6][:seq_len], dtype=torch.long)
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "sample_index": torch.tensor([0], dtype=torch.long),
        }
        self.dataloader = [batch]
        self.data_iterator = iter(self.dataloader)

    def is_process_group_initialized(self):
        import torch.distributed as dist

        return dist.is_initialized()
