from .collator import (
    DataCollatorForSupervisedDataset,
    FlattenedDataCollatorForSupervisedDataset,
    VisionDataCollator,
)
from .dataset import (
    LazySupervisedDataset,
    build_dataloader,
    get_dataset_configs,
    parse_sampling_rate,
)
from .preprocessing import preprocess_qwen_visual
from .rope_utils import get_rope_index_for_images

# Dataset registry - fallback for backwards compatibility
# NOTE: This is now configurable via the config file's data.data_registry field
# These are kept as defaults if no config is provided
DEFAULT_DATA_REGISTRY = {
    "mscoco2017_train_captions": {
        "annotation_path": "__PATH_TO_MSCOCO2017__/annotations/coco2017_train_qwen.json",
        "data_path": "__PATH_TO_MSCOCO2017__",
    },
    "mscoco2017_val_captions": {
        "annotation_path": "__PATH_TO_MSCOCO2017__/annotations/coco2017_val_qwen.json",
        "data_path": "__PATH_TO_MSCOCO2017__",
    },
}

__all__ = [
    "LazySupervisedDataset",
    "build_dataloader",
    "DataCollatorForSupervisedDataset",
    "FlattenedDataCollatorForSupervisedDataset",
    "VisionDataCollator",
    "preprocess_qwen_visual",
    "get_rope_index_for_images",
    "DEFAULT_DATA_REGISTRY",
    "parse_sampling_rate",
    "get_dataset_configs",
]
