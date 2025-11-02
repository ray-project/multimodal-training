import json
import random
import re
import time
from collections.abc import Sequence
from typing import Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from .collator import (
    DataCollatorForSupervisedDataset,
    FlattenedDataCollatorForSupervisedDataset,
    VisionDataCollator,
)
from .config import DataConfig
from .preprocessing import preprocess_qwen_visual
from .rope_utils import get_rope_index_for_images

local_rank = None if not dist.is_initialized() else dist.get_rank()


def rank0_print(*args, **kwargs):
    if local_rank == 0:
        print(*args, **kwargs)


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def parse_sampling_rate(dataset_name: str) -> float:
    """
    Parse sampling rate from dataset name.

    Example: 'dataset_name%50' returns 0.5 (50% sampling)
    """
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def get_dataset_configs(dataset_names: List[str], data_registry: Optional[Dict] = None) -> List[Dict]:
    """
    Get dataset configurations from registry.

    Args:
        dataset_names: List of dataset names (can include sampling rate like 'dataset%50')
        data_registry: Custom dataset registry (uses default if None)

    Returns:
        List of dataset configurations with paths and sampling rates
    """
    if not data_registry:
        from . import DEFAULT_DATA_REGISTRY

        data_registry = DEFAULT_DATA_REGISTRY

    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)

        if dataset_name in data_registry.keys():
            # Convert to regular dict (handles both dict and OmegaConf DictConfig)
            config = dict(data_registry[dataset_name])
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


def update_processor_pixels(processor, min_pixels, max_pixels):

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = min_pixels
        ip.max_pixels = max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = min_pixels
        ip.size["longest_edge"] = max_pixels
        rank0_print(f"✅ Updated image_processor size['shortest_edge'] to {min_pixels}")
        rank0_print(f"✅ Updated image_processor size['longest_edge'] to {max_pixels}")

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    return ip


ModalityType = Literal["all", "image", "text"]


class LazySupervisedDataset(Dataset):
    """
    Lazy-loading dataset for supervised fine-tuning with multimodal data.

    This dataset supports:
    - MS-COCO format annotations
    - Images with proper preprocessing
    - Multiple datasets with sampling rates
    - Lazy loading (images loaded on-demand)
    - 3D RoPE position embeddings for Qwen2.5-VL

    Data format:
        Each sample should have:
        - 'conversations': List of dicts with 'from' and 'value' keys
        - 'image': (optional) Image filename or list of filenames
    """

    def __init__(self, processor, data_config, modality: Optional[ModalityType] = None, seed: int = 42):
        """
        Initialize the dataset.

        Args:
            processor: HuggingFace processor for the model
            data_config: Data configuration object with attributes:
                - datasets: List of dataset names
                - max_pixels, min_pixels: Image resolution limits
            modality: Optional modality filter ('all', 'image', or 'text')
            seed: Random seed for reproducibility (default: 42)
        """
        super(LazySupervisedDataset, self).__init__()
        # Use seed from parameter for reproducibility
        random.seed(seed)

        # Parse and load datasets
        dataset_names = data_config.datasets
        data_registry = getattr(data_config, "data_registry", None)
        dataset_configs = get_dataset_configs(dataset_names, data_registry=data_registry)
        rank0_print(f"Loading datasets: {dataset_configs}", flush=True)

        # Use Qwen2.5-VL RoPE index function for images
        self.get_rope_index = get_rope_index_for_images

        # Load all dataset annotations
        list_data_dict = []

        for config in dataset_configs:
            file_format = config["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(config["annotation_path"])
            else:
                annotations = json.load(open(config["annotation_path"], "r"))

            # Apply sampling rate if specified
            sampling_rate = config.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                rank0_print(f"Sampled {len(annotations)} examples from {config['annotation_path']}")
            else:
                rank0_print(f"Loaded {len(annotations)} examples from {config['annotation_path']}", flush=True)

            # Add data_path to each annotation
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = config["data_path"]
                else:
                    ann["data_path"] = config["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}", flush=True)

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode", flush=True)
        self.image_processor = update_processor_pixels(processor, data_config.min_pixels, data_config.max_pixels)
        if modality is None:
            modality = "all"

        if modality not in ("all", "image", "text"):
            raise ValueError(f"Unsupported modality '{modality}'. Expected one of: 'all', 'image', 'text'.")

        self.modality: ModalityType = modality
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_config = data_config
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict
        self.force_fixed_size = data_config.force_fixed_size
        self.max_pixels = data_config.max_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """Approximate sequence lengths for each sample."""
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self._get_item(sources, index=i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self._get_item(sources, index=next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self._get_item(sources, index=i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources, index: int) -> Dict[str, torch.Tensor]:
        modality_filter = None if self.modality == "all" else self.modality
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            modality_filter=modality_filter,
            force_fixed_size=self.force_fixed_size,
            max_pixels=self.max_pixels,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        # text = self.processor.tokenizer.decode(
        #     data_dict["input_ids"][0], skip_special_tokens=False
        # )

        # labels = data_dict["labels"][0]
        # labels = [
        #     tid if tid != -100 else self.processor.tokenizer.pad_token_id
        #     for tid in labels
        # ]
        # label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        data_dict["sample_index"] = torch.tensor([index], dtype=torch.long)

        if self.modality == "image":
            # Remove text specific keys when loading vision-only samples to save memory
            for key in ("input_ids", "labels", "attention_mask", "position_ids"):
                data_dict.pop(key, None)

        return data_dict


def build_dataloader(
    processor,
    data_config: "DataConfig",
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    generator: "torch.Generator" = None,
    worker_init_fn: "Callable" = None,
    drop_last: bool = True,
    modality: Optional[ModalityType] = None,
    seed: int = 42,
):
    """
    Build training dataloader with distributed support.

    Args:
        processor: HuggingFace processor providing tokenizer/image processor
        data_config: DataConfig describing datasets and dataloader options
        batch_size: Batch size per device
        rank: Current process rank
        world_size: Total number of processes
        generator: Optional torch.Generator for deterministic shuffling
        worker_init_fn: Optional worker init callback for deterministic workers
        drop_last: Drop incomplete batches when True
        modality: Optional override for modality selection ("all", "image", "text")
        seed: Random seed for reproducibility (default: 42)

    Returns:
        DataLoader for training
    """
    import torch
    from torch.utils.data import DataLoader

    # Create dataset
    if modality is None:
        modality = getattr(data_config, "modality", "all")

    dataset = LazySupervisedDataset(processor=processor, data_config=data_config, modality=modality, seed=seed)

    # Create data collator
    if modality == "image":
        collator = VisionDataCollator()
    elif data_config.data_flatten:
        collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
    else:
        collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)

    # Create distributed sampler if multi-GPU
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # Shuffle only if not using sampler
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        collate_fn=collator,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return dataloader
