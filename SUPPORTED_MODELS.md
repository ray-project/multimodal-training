# Supported Models

This document describes the vision-language models (VLMs) supported by the multimodal training framework.

## Requirements

| Dependency | Minimum Version | Tested Version |
|------------|-----------------|----------------|
| transformers | 4.45.0 | 4.57.1 |
| torch | 2.0.0 | 2.5.1 |
| deepspeed | 0.14.0 | 0.17.6 |

## Qwen2.5-VL

### Model Variants
| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| Qwen2.5-VL-2B | 2B | `Qwen/Qwen2.5-VL-2B-Instruct` |
| Qwen2.5-VL-7B | 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen2.5-VL-32B | 32B | `Qwen/Qwen2.5-VL-32B-Instruct` |
| Qwen2.5-VL-72B | 72B | `Qwen/Qwen2.5-VL-72B-Instruct` |

### Supported Parallelism Strategies

| Vision Model | Text Model | Status | Notes |
|--------------|------------|--------|-------|
| Tensor Parallel (TP) | Tensor Parallel (TP) | Supported | DTensor-based TP for both components |
| Sequence Parallel (SP) | Tensor Parallel (TP) | Supported | DeepSpeed Ulysses SP for vision, DTensor TP for text |
| Tensor Parallel (TP) | AutoTP | Supported | DTensor TP for vision, DeepSpeed AutoTP for text |
| Sequence Parallel (SP) | AutoTP | Supported | DeepSpeed SP for vision, DeepSpeed AutoTP for text |

### Configuration Example
```yaml
vision:
  model_type: "qwen2_5_vl"
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  parallelism: "sequence"  # Options: "tensor", "sequence"

text:
  model_type: "qwen2_5_vl"
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  parallelism: "tensor"  # Options: "tensor", "autotp"
```

### Architecture Notes
- Vision encoder uses separate Q, K, V projections (`q_proj`, `k_proj`, `v_proj`)
- Supports windowed attention with configurable `fullatt_block_indexes`
- Merger MLP projects vision embeddings to text model hidden size

---

## Qwen3-VL

### Model Variants
| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| Qwen3-VL-2B | 2B | `Qwen/Qwen3-VL-2B-Instruct` |
| Qwen3-VL-8B | 8B | `Qwen/Qwen3-VL-8B-Instruct` |

### Supported Parallelism Strategies

| Vision Model | Text Model | Status | Notes |
|--------------|------------|--------|-------|
| Tensor Parallel (TP) | Tensor Parallel (TP) | Not Supported | Fused QKV incompatible with DTensor ColwiseParallel |
| Sequence Parallel (SP) | Tensor Parallel (TP) | Supported | DeepSpeed SP for vision, DTensor TP for text |
| Tensor Parallel (TP) | AutoTP | Not Supported | Vision TP limitation |
| Sequence Parallel (SP) | AutoTP | Supported | DeepSpeed SP for vision, DeepSpeed AutoTP for text |

### Configuration Example
```yaml
vision:
  model_type: "qwen3_vl"
  model_name: "Qwen/Qwen3-VL-8B-Instruct"
  parallelism: "sequence"  # Only "sequence" supported currently

text:
  model_type: "qwen3_vl"
  model_name: "Qwen/Qwen3-VL-8B-Instruct"
  parallelism: "tensor"  # Options: "tensor", "autotp"
```

### Architecture Notes
- Vision encoder uses **fused QKV** (single `attn.qkv` layer) instead of separate projections
- Introduces **DeepStack** feature: extracts intermediate features from vision encoder layers [8, 16, 24] and injects them into text decoder
- Different activation function: `gelu_pytorch_tanh` (vs `silu` in Qwen2.5-VL)
- Position embeddings use interpolation (`fast_pos_embed_interpolate`) for variable grid sizes

### DeepStack Feature (Qwen3-VL Specific)
DeepStack is a new architectural feature in Qwen3-VL that fuses multi-level vision features into the text decoder:

```
Vision Encoder                    Text Decoder
    Layer 8  -----> DeepStack Merger 0 ----> Injection at layer N
    Layer 16 -----> DeepStack Merger 1 ----> Injection at layer M
    Layer 24 -----> DeepStack Merger 2 ----> Injection at layer K
    Layer 27 -----> Main Merger -----------> Input embeddings
```

**Current Status**: DeepStack features are computed during forward pass but gradient flow through DeepStack is not yet implemented. This is a known limitation.

---

## Known Limitations

### Qwen3-VL Tensor Parallelism (Vision)
Tensor parallelism for the Qwen3-VL vision encoder is not currently supported due to the fused QKV architecture. The fused QKV linear layer produces output that is then reshaped assuming the full tensor dimension, which is incompatible with DTensor's column-wise sharding.

**Workaround**: Use sequence parallelism (`parallelism: "sequence"`) for the vision encoder.

### DeepStack Gradient Flow
While Qwen3-VL's DeepStack features are computed and passed to the text model during forward pass, the backward gradient flow through DeepStack mergers is not yet fully implemented for the disaggregated training setup.

---

## Transformers Version Compatibility

| Model | Minimum transformers | Notes |
|-------|---------------------|-------|
| Qwen2.5-VL | 4.45.0 | First version with Qwen2.5-VL support |
| Qwen3-VL | 4.57.0 | First version with Qwen3-VL support |

To check your transformers version:
```bash
python -c "import transformers; print(transformers.__version__)"
```

To upgrade transformers:
```bash
pip install --upgrade transformers>=4.57.0
```
