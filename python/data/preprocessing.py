import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
import transformers
from PIL import Image

# Constants for token IDs
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
DEFAULT_IMAGE_TOKEN = "<image>"


def _make_abs_paths(base: Path, files: str) -> str:
    """Convert relative paths to absolute paths."""
    return f"{(base / files).resolve()}"


def _build_messages(
    item: Dict[str, Any],
    base_path: Path,
    modality_filter: Optional[Literal["image", "text"]] = None,
) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [{"type": "image", "image": _make_abs_paths(base_path, img)} for img in images]
    video_pool = [{"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError("Number of <image> placeholders exceeds the number of provided images")
                    # Only include image if not filtering for text-only
                    if modality_filter != "text":
                        content.append(image_pool.pop(0))
                    else:
                        image_pool.pop(0)
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError("Number of <video> placeholders exceeds the number of provided videos")
                    # Only include video if not filtering for text-only
                    if modality_filter != "text":
                        content.append(video_pool.pop(0))
                    else:
                        video_pool.pop(0)
                elif seg.strip():
                    if modality_filter != "image":
                        content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            if modality_filter != "image":
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)")

    return messages


def preprocess_qwen_visual(
    sources: List[Dict[str, Any]],
    processor: transformers.ProcessorMixin,
    modality_filter: Optional[Literal["image", "text"]] = None,
    force_fixed_size: bool = False,
    max_pixels: int = 50176,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess conversations with images using the processor's apply_chat_template.

    This is the processor-based approach (like in data_processor.py) which is more
    robust than manual tokenization.

    Args:
        sources: List with single source dict containing 'conversations' and optional 'image'
        processor: HuggingFace processor with apply_chat_template method
        modality_filter: Optional filter to include only 'image' or 'text' modality
            - None: Include both image and text (default)
            - 'image': Only include image content in messages
            - 'text': Only include textual content in messages
        force_fixed_size: If True, resize all images to fixed size before processing
        max_pixels: Used to calculate fixed size as sqrt(max_pixels) x sqrt(max_pixels)

    Returns:
        Dictionary with 'input_ids', 'labels', and image-related tensors
    """
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path, modality_filter=modality_filter)

    # Resize images to fixed size if requested (before any other processing)
    if force_fixed_size and modality_filter != "text":
        import math

        fixed_size = int(math.sqrt(max_pixels))
        for message in messages:
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if content_item.get("type") == "image" and "image" in content_item:
                        # Load and resize the image
                        image_path = content_item["image"]
                        img = Image.open(image_path).convert("RGB")
                        img_resized = img.resize((fixed_size, fixed_size), Image.Resampling.BILINEAR)
                        # Replace path with PIL Image object
                        content_item["image"] = img_resized

    # Use processor's apply_chat_template for proper tokenization
    full_result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Create labels: mask everything except assistant responses
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0

    # Find assistant response tokens (between <|im_start|>assistant and <|im_end|>)
    # Token 77091 is <|im_start|>, 151645 is <|im_end|>
    while pos < L:
        if input_ids_flat[pos] == 77091:  # <|im_start|>
            ans_start = pos + 2  # Skip <|im_start|> and role token
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:  # <|im_end|>
                ans_end += 1
            if ans_end < L:
                # Include the response and <|im_end|> token
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result
