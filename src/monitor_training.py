#!/usr/bin/env python3

import re
import sys
from typing import List, Dict, Tuple

def parse_training_log(lines: List[str]) -> Tuple[Dict[int, List[float]], Dict[int, Dict[str, float]]]:
    epoch_losses = {}
    eval_metrics = {}
    
    current_epoch = -1

    for line in lines:
        epoch_match = re.search(r'===== Epoch (\d+) =====', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in epoch_losses:
                epoch_losses[current_epoch] = []
                eval_metrics[current_epoch] = {}

        loss_match = re.search(r'\[Epoch \d+ \| Step \d+\] Total: ([\d.]+) \| Ans: ([\d.]+) \| Aux: ([\d.]+)', line)
        if loss_match and current_epoch >= 0:
            total_loss = float(loss_match.group(1))
            epoch_losses[current_epoch].append(total_loss)

        eval_loss_match = re.search(r'\[Eval\] mean loss = ([\d.]+)', line)
        if eval_loss_match and current_epoch >= 0:
            eval_metrics[current_epoch]['eval_loss'] = float(eval_loss_match.group(1))

        em_f1_match = re.search(r'\[Eval\] EM = ([\d.]+), F1 = ([\d.]+)', line)
        if em_f1_match and current_epoch >= 0:
            eval_metrics[current_epoch]['em'] = float(em_f1_match.group(1))
            eval_metrics[current_epoch]['f1'] = float(em_f1_match.group(1))

    return epoch_losses, eval_metrics

def print_summary(epoch_losses: Dict[int, List[float]], eval_metrics: Dict[int, Dict[str, float]]):
    print("\n" + "="*60)
    print("TRAINING PROGRESS SUMMARY")
    print("="*60)
    
    for epoch in sorted(epoch_losses.keys()):
        losses = epoch_losses[epoch]
        if losses:
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            print(f"\nEpoch {epoch}:")
            print(f"  Training - Avg Loss: {avg_loss:.4f}, Min Loss: {min_loss:.4f}")
        
        if epoch in eval_metrics:
            metrics = eval_metrics[epoch]
            if 'eval_loss' in metrics:
                print(f"  Eval Loss: {metrics['eval_loss']:.4f}")
            if 'em' in metrics:
                print(f"  EM: {metrics['em']:.2%}, F1: {metrics['f1']:.2%}")

    print("\n" + "-"*60)
    print("RECOMMENDATIONS:")

    if len(eval_metrics) >= 2:
        epochs = sorted(eval_metrics.keys())
        last_epoch = epochs[-1]
        prev_epoch = epochs[-2]
        
        if last_epoch in eval_metrics and prev_epoch in eval_metrics:
            if 'em' in eval_metrics[last_epoch] and 'em' in eval_metrics[prev_epoch]:
                em_improvement = eval_metrics[last_epoch]['em'] - eval_metrics[prev_epoch]['em']

                if em_improvement < 0.01:
                    print("⚠️  WARNING: EM improvement < 1% between epochs")
                    print("   Consider adjusting hyperparameters")
                elif em_improvement < 0.02:
                    print("⚠️  CAUTION: EM improvement is slow (< 2%)")
                    print("   Training might benefit from tuning")
                else:
                    print("✓  Good improvement between epochs")

    all_losses = []
    for losses in epoch_losses.values():
        all_losses.extend(losses)

    if all_losses and min(all_losses) > 3.0:
        print("⚠️  Loss is still high (> 3.0)")
        print("   Model may need more epochs or better hyperparameters")

    print("="*60 + "\n")

if __name__ == "__main__":
    lines = sys.stdin.readlines()
    
    if not lines:
        print("Usage: python monitor_training.py < training.log")
        print("   Or: tail -f training.log | python monitor_training.py")
        sys.exit(1)
    
    epoch_losses, eval_metrics = parse_training_log(lines)
    print_summary(epoch_losses, eval_metrics)