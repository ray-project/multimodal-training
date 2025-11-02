from typing import Optional, Tuple

import torch


def get_rope_index_for_images(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate 3D RoPE (Rotary Position Embeddings) for Qwen2.5-VL with images.

    This computes 3D position embeddings for image tokens (temporal, height, width)
    and 1D position embeddings for text tokens.

    Args:
        spatial_merge_size: Size of spatial merging (default: 2)
        input_ids: Input token IDs of shape (batch_size, sequence_length)
        image_grid_thw: Image grid dimensions of shape (num_images, 3) for [temporal, height, width]
        attention_mask: Attention mask of shape (batch_size, sequence_length)

    Returns:
        position_ids: 3D position IDs of shape (3, batch_size, sequence_length)
        mrope_position_deltas: Position deltas of shape (batch_size, 1)
    """
    image_token_id = 151655
    vision_start_token_id = 151652
    mrope_position_deltas = []

    if input_ids is not None and image_grid_thw is not None:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index = 0
        attention_mask = attention_mask.to(input_ids.device)

        for i, seq_input_ids in enumerate(input_ids):
            seq_input_ids = seq_input_ids[attention_mask[i] == 1]

            # Count images in this sequence
            vision_start_indices = torch.argwhere(seq_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = seq_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()

            input_tokens = seq_input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0

            # Process each image
            for _ in range(image_nums):
                # Find image token position
                ed = input_tokens.index(image_token_id, st)

                # Get image grid dimensions
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1

                # Compute grid dimensions after spatial merging
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                # Text before image: 1D position IDs
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Image tokens: 3D position IDs (t, h, w)
                # For images, temporal dimension is 0 (no scaling needed)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Remaining text after all images: 1D position IDs
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # No images: simple 1D position encoding
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
