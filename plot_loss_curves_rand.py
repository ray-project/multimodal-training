#!/usr/bin/env python3
"""Plot loss curves from training log files."""

import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(filepath):
    """Parse a log file and extract iteration numbers and loss values."""
    iterations = []
    losses = []

    pattern = r"Iter (\d+)/\d+ \((warmup|training)\) - Loss: ([\d.]+)"

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                loss = float(match.group(3))
                iterations.append(iteration)
                losses.append(loss)

    return np.array(iterations), np.array(losses)


def parse_qwen_log_file(filepath):
    """Parse a Qwen original repo log file and extract iteration numbers and loss values."""
    iterations = []
    losses = []

    # Pattern to match lines like: {'loss': 12.7565, 'grad_norm': ...}
    pattern = r"\{'loss': ([\d.]+),"

    with open(filepath, 'r') as f:
        iteration = 0
        for line in f:
            match = re.search(pattern, line)
            if match:
                iteration += 1
                loss = float(match.group(1))
                iterations.append(iteration)
                losses.append(loss)

    return np.array(iterations), np.array(losses)


def smooth_curve(values, weight=0.9):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def main():
    # Parse log files
    cond0_iters, cond0_losses = parse_log_file(
        '/home/ray/default/multimodal-training/rand_sp_tp_step1000.log'
    )
    cond1_iters, cond1_losses = parse_log_file(
        '/home/ray/default/multimodal-training/rand_tp_tp_step1000.log'
    )
    cond2_iters, cond2_losses = parse_log_file(
        '/home/ray/default/multimodal-training/rand_tp_autotp_step1000.log'
    )
    qwen_iters, qwen_losses = parse_qwen_log_file(
        '/home/ray/default/Qwen2.5-VL/qwen-vl-finetune/qwen_rand_step1000.log'
    )

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Raw loss curves
    ax1 = axes[0]
    ax1.plot(cond0_iters, cond0_losses, alpha=0.5, label='Cond0: SP (vision) & TP (language)', color='green')
    ax1.plot(cond1_iters, cond1_losses, alpha=0.5, label='Cond1: TP (vision) & TP (language)', color='blue')
    ax1.plot(cond2_iters, cond2_losses, alpha=0.5, label='Cond2: TP (vision) & AutoTP (language)', color='orange')
    ax1.plot(qwen_iters, qwen_losses, alpha=0.5, label='Qwen original repo', color='red')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves (Raw)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Smoothed loss curves
    ax2 = axes[1]
    cond0_smooth = smooth_curve(cond0_losses)
    cond1_smooth = smooth_curve(cond1_losses)
    cond2_smooth = smooth_curve(cond2_losses)
    qwen_smooth = smooth_curve(qwen_losses)
    # ax2.plot(cond0_iters, cond0_smooth, label='Cond0: DeepSpeed (vision) & DeepSpeed (language)', color='green', linewidth=2)
    ax2.plot(cond0_iters, cond0_smooth, label='Cond0: SP (vision) & TP (language)', color='green', linewidth=2)
    ax2.plot(cond1_iters, cond1_smooth, label='Cond1: TP (vision) & TP (language)', color='blue', linewidth=2)
    ax2.plot(cond2_iters, cond2_smooth, label='Cond2: TP (vision) & AutoTP (language)', color='orange', linewidth=2)
    ax2.plot(qwen_iters, qwen_smooth, label='Qwen original repo', color='red', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Curves (Smoothed, EMA=0.9)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure
    output_path = '/home/ray/default/multimodal-training/loss_curves_rand.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    # Print some statistics
    print(f"\nStatistics:")
    print(f"Cond0: {len(cond0_losses)} iterations, final loss: {cond0_losses[-1]:.4f}, min loss: {cond0_losses.min():.4f}")
    print(f"Cond1: {len(cond1_losses)} iterations, final loss: {cond1_losses[-1]:.4f}, min loss: {cond1_losses.min():.4f}")
    print(f"Cond2: {len(cond2_losses)} iterations, final loss: {cond2_losses[-1]:.4f}, min loss: {cond2_losses.min():.4f}")
    print(f"Qwen original repo: {len(qwen_losses)} iterations, final loss: {qwen_losses[-1]:.4f}, min loss: {qwen_losses.min():.4f}")

    plt.show()


if __name__ == '__main__':
    main()
