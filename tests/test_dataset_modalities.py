import copy
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from ray_hybrid_para.python.data import DEFAULT_DATA_REGISTRY  # noqa: E402
from ray_hybrid_para.python.data.config import DataConfig  # noqa: E402
from ray_hybrid_para.python.data.dataset import LazySupervisedDataset, build_dataloader  # noqa: E402

pytestmark = pytest.mark.cpu_only


class DummyTokenizer:
    pad_token_id = 0
    model_max_length = 128


class DummyImageProcessor:

    def __init__(self):
        self.min_pixels = 784
        self.max_pixels = 50176
        self.size = {"shortest_edge": 28, "longest_edge": 224}
        self.merge_size = 2


class DummyProcessor:

    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.image_processor = DummyImageProcessor()


@pytest.fixture
def dummy_processor() -> DummyProcessor:
    return DummyProcessor()


@pytest.fixture
def sample_annotations() -> list[dict]:
    return [
        {
            "id": 0,
            "conversations": [
                {"from": "human", "value": "<image> Describe"},
                {"from": "gpt", "value": "Sample answer"},
            ],
            "image": "img0.png",
        },
        {
            "id": 1,
            "conversations": [
                {"from": "human", "value": "<image> Another"},
                {"from": "gpt", "value": "Another sample"},
            ],
            "image": "img1.png",
        },
    ]


@pytest.fixture
def dataset_patches(sample_annotations: list[dict]):
    config_template = {"annotation_path": "dummy.jsonl", "data_path": "/tmp"}

    def fake_get_dataset_configs(dataset_names, data_registry=None):
        return [dict(config_template, sampling_rate=1.0) for _ in dataset_names]

    def fake_read_jsonl(_path):
        return [copy.deepcopy(item) for item in sample_annotations]

    def fake_preprocess(sources, _processor, modality_filter=None, **kwargs):
        item = sources[0]
        idx = item["id"]
        result: dict[str, torch.Tensor] = {}

        input_ids = torch.tensor([[idx, idx + 1, idx + 2]], dtype=torch.long)
        labels = torch.tensor([[idx + 10, idx + 11, idx + 12]], dtype=torch.long)
        result["input_ids"] = input_ids
        result["labels"] = labels

        if modality_filter != "text":
            pixel_values = torch.full((1, 3, 2, 2), float(idx))
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = torch.tensor([[1, 1, 1]], dtype=torch.long)

        return result

    def fake_get_rope_index(_merge_size, input_ids, image_grid_thw=None):
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, 1, -1).repeat(1, 2, 1)
        return position_ids, None

    @contextmanager
    def _apply_patches():
        with (
            patch("ray_hybrid_para.python.data.dataset.get_dataset_configs", side_effect=fake_get_dataset_configs),
            patch("ray_hybrid_para.python.data.dataset.read_jsonl", side_effect=fake_read_jsonl),
            patch("ray_hybrid_para.python.data.dataset.preprocess_qwen_visual", side_effect=fake_preprocess),
            patch("ray_hybrid_para.python.data.dataset.get_rope_index_for_images", side_effect=fake_get_rope_index),
        ):
            yield

    return _apply_patches


def _resolve_dataset_paths(dataset_name: str) -> tuple[Path, Path]:
    registry_entry = DEFAULT_DATA_REGISTRY.get(dataset_name)
    if registry_entry is None:
        raise KeyError(f"Dataset '{dataset_name}' is not registered")

    annotation_path = Path(registry_entry["annotation_path"]).expanduser()
    data_path = Path(registry_entry["data_path"]).expanduser()
    return annotation_path, data_path


def test_text_modality_excludes_images(dataset_patches, dummy_processor):
    data_cfg = DataConfig(datasets=["dummy"], num_workers=0, data_flatten=False)

    with dataset_patches():
        text_dataset = LazySupervisedDataset(dummy_processor, data_cfg, modality="text")
        image_dataset = LazySupervisedDataset(dummy_processor, data_cfg, modality="image")

        text_sample = text_dataset[0]

        assert "pixel_values" not in text_sample
        assert "input_ids" in text_sample
        assert text_sample["sample_index"].item() == 0

        text_order = [item["id"] for item in text_dataset.list_data_dict]
        image_order = [item["id"] for item in image_dataset.list_data_dict]
        assert text_order == image_order


def test_image_modality_drops_text_keys(dataset_patches, dummy_processor):
    data_cfg = DataConfig(datasets=["dummy"], num_workers=0, data_flatten=False)

    with dataset_patches():
        dataset = LazySupervisedDataset(dummy_processor, data_cfg, modality="image")
        sample = dataset[0]

        assert "input_ids" not in sample
        assert "labels" not in sample
        assert "pixel_values" in sample
        assert sample["pixel_values"].shape[0] == 1
        assert sample["sample_index"].item() == 0


def test_build_dataloader_image_modality(dataset_patches, dummy_processor):
    data_cfg = DataConfig(datasets=["dummy"], num_workers=0, data_flatten=False)

    with dataset_patches():
        loader = build_dataloader(
            processor=dummy_processor,
            data_config=data_cfg,
            batch_size=2,
            rank=0,
            world_size=1,
            drop_last=False,
            modality="image",
        )
        batch = next(iter(loader))

    assert "input_ids" not in batch
    assert sorted(batch["sample_index"].tolist()) == [0, 1]
    assert batch["pixel_values"].shape[0] == 2


def test_real_dataset_alignment():
    dataset_name = "mscoco2017_val_captions"

    try:
        annotation_path, data_path = _resolve_dataset_paths(dataset_name)
    except KeyError:
        pytest.skip(f"Dataset '{dataset_name}' not present in registry")

    if not annotation_path.exists() or not data_path.exists():
        pytest.skip("Required dataset files not available")

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    processor = AutoProcessor.from_pretrained(model_name)

    data_cfg = DataConfig(datasets=[dataset_name], num_workers=0, data_flatten=False)

    dataset_all = LazySupervisedDataset(processor, data_cfg, modality="all")
    dataset_img = LazySupervisedDataset(processor, data_cfg, modality="image")
    dataset_txt = LazySupervisedDataset(processor, data_cfg, modality="text")

    assert len(dataset_all) == len(dataset_img) == len(dataset_txt)

    idx = None
    for i in range(len(dataset_all)):
        sample_all = dataset_all[i]
        if sample_all.get("pixel_values") is not None:
            idx = i
            break

    if idx is None:
        pytest.skip("No multimodal sample found in dataset")

    sample_all = dataset_all[idx]
    sample_img = dataset_img[idx]
    sample_txt = dataset_txt[idx]

    assert sample_all["sample_index"].item() == idx
    assert sample_img["sample_index"].item() == idx
    assert sample_txt["sample_index"].item() == idx

    assert "input_ids" not in sample_img
    assert "labels" not in sample_img
    assert sample_img["pixel_values"] is not None

    assert "pixel_values" not in sample_txt
    assert "image_grid_thw" not in sample_txt
    assert "input_ids" in sample_txt

    ids_all = [item.get("id") for item in dataset_all.list_data_dict[:5]]
    ids_img = [item.get("id") for item in dataset_img.list_data_dict[:5]]
    ids_txt = [item.get("id") for item in dataset_txt.list_data_dict[:5]]
    assert ids_all == ids_img == ids_txt


def test_build_dataloader_text_modality(dataset_patches, dummy_processor):
    data_cfg = DataConfig(datasets=["dummy"], num_workers=0, data_flatten=False)

    with dataset_patches():
        loader = build_dataloader(
            processor=dummy_processor,
            data_config=data_cfg,
            batch_size=2,
            rank=0,
            world_size=1,
            drop_last=False,
            modality="text",
        )
        batch = next(iter(loader))

    assert batch["pixel_values"] is None
    assert batch["image_grid_thw"] is None
    assert sorted(batch["sample_index"].tolist()) == [0, 1]
    assert batch["input_ids"].shape[0] == 2
