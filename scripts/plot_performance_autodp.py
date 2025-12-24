#!/usr/bin/env python3
"""
Plot performance comparison across three multimodal training systems:
- ZeRO3: vanilla ZeRO3 baseline from qwen-vl-finetune
- TP: vision AutoTP + text AutoTP (vautotp8dp2_tautotp8dp2)
- SP+TP: vision sequence parallel + text AutoTP (vseq8dp2_tautotp8dp2)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# System configurations
SYSTEMS = {
    "ZeRO3": {
        "path": "/shared/work/Qwen2.5-VL/qwen-vl-finetune/logs/sweep_torchrun_20251224_002624",
        "pattern": r"(\d+)k_tokens\.log",
        "batch_sizes": [1],  # Fixed batch size per device
        "dp_size": 16,  # ZeRO3 across 16 GPUs (2 nodes x 8 GPUs)
    },
    "TP": {
        "path": "/shared/work/multimodal-training/logs/sweep_20251223_031752",
        "pattern": r"vautotp8dp2_tautotp8dp2_(\d+)k_tokens_bs(\d+)_",
        "batch_sizes": [1, 2, 4, 8],
        "dp_size": 2,  # Data parallel size for throughput calculation
    },
    "SP+TP": {
        "path": "/shared/work/multimodal-training/logs/sweep_20251223_031752",
        "pattern": r"vseq8dp2_tautotp8dp2_(\d+)k_tokens_bs(\d+)_",
        "batch_sizes": [1, 2, 4, 8],
        "dp_size": 2,  # Data parallel size for throughput calculation
    },
}

SEQUENCE_LENGTHS = [1, 4, 8, 16, 32, 64]  # in K tokens

# Mapping from filename token designation to actual vision token count (before spatial merging)
TOKEN_COUNT_MAPPING = {
    1: 1_024,
    4: 4_096,
    8: 9_216,
    16: 16_384,
    32: 25_600,
    64: 65_536,
}


def extract_iteration_time(log_file):
    """
    Extract average iteration time from a log file.
    Handles both old and new log formats.
    """
    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Try different patterns for average iteration time
        patterns = [
            r"Average iteration time:\s*([0-9.]+)s",
            r"Average iteration time \(excluding warmup\):\s*([0-9.]+)\s*seconds",
            r"Average iteration time:\s*([0-9.]+)\s*seconds",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the last match (most recent/final)
                return float(matches[-1])

        print(f"Warning: Could not find iteration time in {log_file}")
        return None

    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def parse_logs(system_name, config):
    """
    Parse log files for a given system and return timing data.
    Returns: dict mapping (seq_len, batch_size) -> (iteration_time, actual_token_count, dp_size)
    """
    log_dir = Path(config["path"])
    pattern = config["pattern"]
    dp_size = config["dp_size"]
    data = {}

    if not log_dir.exists():
        print(f"Warning: Directory not found: {log_dir}")
        return data

    for log_file in log_dir.glob("*.log"):
        filename = log_file.name

        # Parse filename to extract sequence length and batch size
        match = re.search(pattern, filename)
        if not match:
            continue

        seq_len = int(match.group(1))

        # Skip sequence lengths not in mapping
        if seq_len not in TOKEN_COUNT_MAPPING or TOKEN_COUNT_MAPPING[seq_len] is None:
            print(f"Skipping {filename} (seq_len={seq_len}k not in mapping)")
            continue

        actual_token_count = TOKEN_COUNT_MAPPING[seq_len]

        # For ZeRO3 (old format), batch size is fixed at 1 per device
        if system_name == "ZeRO3":
            batch_size = 1
        else:
            batch_size = int(match.group(2))

        # Extract iteration time
        iter_time = extract_iteration_time(log_file)
        if iter_time is not None:
            data[(seq_len, batch_size)] = (iter_time, actual_token_count, dp_size)
            print(
                f"{system_name}: {seq_len}k tokens ({actual_token_count} actual), bs={batch_size}, dp={dp_size} -> {iter_time:.3f}s"
            )

    return data


def select_max_batch_size(data, seq_lengths):
    """
    For each sequence length, select the data from the largest available batch size.
    Returns: dict mapping seq_len -> (tokens_per_second, batch_size, iter_time)
    """
    result = {}

    for seq_len in seq_lengths:
        # Skip if not in mapping
        if seq_len not in TOKEN_COUNT_MAPPING or TOKEN_COUNT_MAPPING[seq_len] is None:
            continue

        # Find all entries for this sequence length
        entries = [
            (bs, iter_time, actual_tokens, dp_size)
            for (sl, bs), (iter_time, actual_tokens, dp_size) in data.items()
            if sl == seq_len
        ]

        if entries:
            # Sort by batch size (descending) and take the largest
            entries.sort(reverse=True)
            max_bs, iter_time, actual_tokens, dp_size = entries[0]

            # Calculate tokens per second: (tokens * batch_size * dp_size) / time
            # This gives total throughput across all devices
            tokens_per_second = (actual_tokens * max_bs * dp_size) / iter_time

            result[seq_len] = (tokens_per_second, max_bs, iter_time, dp_size)
            print(
                f"  Selected: {seq_len}k tokens ({actual_tokens} actual) with bs={max_bs}, dp={dp_size}: {iter_time:.3f}s -> {tokens_per_second:.1f} tokens/s"
            )

    return result


def plot_performance(data_by_system):
    """
    Create a grouped bar chart comparing performance across systems.
    """
    # Get all sequence lengths that have at least one data point
    all_seq_lengths = sorted(set(seq_len for system_data in data_by_system.values() for seq_len in system_data.keys()))

    if not all_seq_lengths:
        print("Error: No data to plot!")
        return

    # Prepare data for plotting
    system_names = list(data_by_system.keys())
    n_systems = len(system_names)
    n_seq_lengths = len(all_seq_lengths)

    # Set up the bar positions
    x = np.arange(n_seq_lengths)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each system
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, system_name in enumerate(system_names):
        system_data = data_by_system[system_name]

        # Get tokens per second for each sequence length (or None if not available)
        data_tuples = [system_data.get(seq_len, None) for seq_len in all_seq_lengths]
        tokens_per_sec = [d[0] if d is not None else None for d in data_tuples]

        # Create x positions for this system's bars
        x_pos = x + (i - n_systems / 2 + 0.5) * width

        # Plot bars (skip None values)
        for j, (pos, throughput) in enumerate(zip(x_pos, tokens_per_sec)):
            if throughput is not None:
                ax.bar(
                    pos,
                    throughput,
                    width,
                    label=system_name if j == 0 else "",
                    color=colors[i % len(colors)],
                    alpha=0.8,
                )

    # Customize plot
    ax.set_xlabel("Vision Token Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (tokens/second)", fontsize=12, fontweight="bold")
    ax.set_title("Throughput Comparison: ZeRO3 vs TP vs SP+TP\n(AutoTP + DP experiments)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    # Create better x-axis labels with actual token counts
    x_labels = []
    for seq_len in all_seq_lengths:
        actual_tokens = TOKEN_COUNT_MAPPING.get(seq_len, 0)
        if actual_tokens >= 1000:
            x_labels.append(f"{actual_tokens // 1000}K")
        else:
            x_labels.append(f"{actual_tokens}")
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars with batch size and ratios for SP+TP
    for i, system_name in enumerate(system_names):
        system_data = data_by_system[system_name]
        data_tuples = [system_data.get(seq_len, None) for seq_len in all_seq_lengths]
        x_pos = x + (i - n_systems / 2 + 0.5) * width

        for j, (pos, data_tuple) in enumerate(zip(x_pos, data_tuples)):
            if data_tuple is not None:
                tokens_per_sec, batch_size, iter_time, dp_size = data_tuple

                # For SP+TP, add speedup ratios
                if system_name == "SP+TP":
                    seq_len = all_seq_lengths[j]

                    # Get baseline throughputs
                    zero3_data = data_by_system.get("ZeRO3", {}).get(seq_len)
                    tp_data = data_by_system.get("TP", {}).get(seq_len)

                    label_parts = [f"{tokens_per_sec:.0f}", f"(bs={batch_size}*{dp_size})"]

                    # Calculate ratio vs ZeRO3
                    if zero3_data is not None:
                        zero3_throughput = zero3_data[0]
                        ratio_zero3 = tokens_per_sec / zero3_throughput
                        label_parts.append(f"{ratio_zero3:.2f}x")
                    else:
                        label_parts.append("--")

                    # Calculate ratio vs TP
                    if tp_data is not None:
                        tp_throughput = tp_data[0]
                        ratio_tp = tokens_per_sec / tp_throughput
                        label_parts.append(f"{ratio_tp:.2f}x")
                    else:
                        label_parts.append("--")

                    label = "\n".join(label_parts)
                else:
                    label = f"{tokens_per_sec:.0f}\n(bs={batch_size}*{dp_size})"

                ax.text(pos, tokens_per_sec, label, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    # Save the plot
    output_file = "performance_comparison_autodp.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    plt.close()


def main():
    print("=" * 60)
    print("Performance Comparison Script (AutoTP + DP)")
    print("=" * 60)
    print()

    # Parse logs for each system
    all_data = {}
    for system_name, config in SYSTEMS.items():
        print(f"\nParsing {system_name} logs...")
        print("-" * 40)
        data = parse_logs(system_name, config)

        print(f"\nSelecting maximum batch size for {system_name}...")
        selected_data = select_max_batch_size(data, SEQUENCE_LENGTHS)
        all_data[system_name] = selected_data
        print()

    # Create comparison plot
    print("=" * 60)
    print("Creating performance comparison plot...")
    print("=" * 60)
    plot_performance(all_data)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for system_name, system_data in all_data.items():
        print(f"\n{system_name}:")
        for seq_len in sorted(system_data.keys()):
            tokens_per_sec, batch_size, iter_time, dp_size = system_data[seq_len]
            actual_tokens = TOKEN_COUNT_MAPPING[seq_len]

            # Add speedup ratios for SP+TP
            if system_name == "SP+TP":
                zero3_data = all_data.get("ZeRO3", {}).get(seq_len)
                tp_data = all_data.get("TP", {}).get(seq_len)

                ratios = []
                if zero3_data is not None:
                    ratio_zero3 = tokens_per_sec / zero3_data[0]
                    ratios.append(f"{ratio_zero3:.2f}x vs ZeRO3")
                if tp_data is not None:
                    ratio_tp = tokens_per_sec / tp_data[0]
                    ratios.append(f"{ratio_tp:.2f}x vs TP")

                ratio_str = ", ".join(ratios) if ratios else ""
                print(
                    f"  {actual_tokens:6} tokens: {tokens_per_sec:8.1f} tokens/s (bs={batch_size}*{dp_size}, {iter_time:.3f}s/iter) [{ratio_str}]"
                )
            else:
                print(
                    f"  {actual_tokens:6} tokens: {tokens_per_sec:8.1f} tokens/s (bs={batch_size}*{dp_size}, {iter_time:.3f}s/iter)"
                )


if __name__ == "__main__":
    main()
