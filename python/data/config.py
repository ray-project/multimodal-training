"""
Configuration management for data loading.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration."""

    # Dataset paths
    datasets: list[str] = field(default_factory=lambda: ["mscoco2017_train_captions"])
    data_registry: dict = field(default_factory=dict)  # Dataset registry with annotation_path and data_path

    # Tokenizer settings
    model_max_length: int = 8192
    padding_side: str = "right"
    use_fast_tokenizer: bool = False

    # Image settings
    max_pixels: int = 50176  # ~224x224
    min_pixels: int = 784  # ~28x28
    force_fixed_size: bool = False  # Force all images to fixed size (sqrt(max_pixels) x sqrt(max_pixels))

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    data_flatten: bool = True
