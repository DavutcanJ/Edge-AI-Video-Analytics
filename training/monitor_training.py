"""
Real-time Training Monitor
Monitors training progress from YOLO logs and displays formatted output.
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import threading


class TrainingMonitor:
    """Monitor and display training progress in real-time."""
    
    def __init__(self, logs_dir: str = "training/logs"):
        self.logs_dir = Path(logs_dir)
        self.latest_exp = None
        self.last_epoch = -1
        self.total_epochs = 0
        self.metrics_history = defaultdict(list)
        
    def find_latest_experiment(self):
        """Find the latest experiment directory."""
        if not self.logs_dir.exists():
            return None
        
        exp_dirs = sorted(
            [d for d in self.logs_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if exp_dirs:
            return exp_dirs[0]
        return None
    
    def parse_results_csv(self, csv_path: Path):
        """Parse results.csv file from YOLO logs."""
        if not csv_path.exists():
            return None
        
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            # Parse header
            header = [h.strip() for h in lines[0].split(',')]
            
            # Parse last line
            last_line = lines[-1].split(',')
            metrics = {header[i].strip(): last_line[i].strip() for i in range(len(header))}
            
            return metrics
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            return None
    
    def display_progress(self):
        """Display training progress with formatting."""
        print("\n" + "="*80)
        print("📊 YOLO Training Monitor")
        print("="*80)
        
        while True:
            self.latest_exp = self.find_latest_experiment()
            
            if not self.latest_exp:
                print("⏳ Waiting for training to start...")
                time.sleep(2)
                continue
            
            results_csv = self.latest_exp / "results.csv"
            
            if results_csv.exists():
                metrics = self.parse_results_csv(results_csv)
                
                if metrics and 'epoch' in metrics:
                    epoch = int(float(metrics.get('epoch', 0)))
                    
                    if epoch != self.last_epoch:
                        self.last_epoch = epoch
                        
                        # Extract key metrics
                        loss_box = float(metrics.get('train/box_loss', 0))
                        loss_cls = float(metrics.get('train/cls_loss', 0))
                        loss_obj = float(metrics.get('train/obj_loss', 0))
                        
                        val_map50 = float(metrics.get('metrics/mAP50(B)', 0))
                        val_precision = float(metrics.get('metrics/precision(B)', 0))
                        val_recall = float(metrics.get('metrics/recall(B)', 0))
                        
                        # Display epoch info
                        print(f"\n🔄 Epoch {epoch}")
                        print("-" * 80)
                        
                        # Training losses
                        print(f"  📉 Training Losses:")
                        print(f"      Box Loss:   {loss_box:.4f}")
                        print(f"      Class Loss: {loss_cls:.4f}")
                        print(f"      Obj Loss:   {loss_obj:.4f}")
                        
                        # Validation metrics
                        print(f"  ✅ Validation Metrics:")
                        print(f"      mAP@0.5:    {val_map50:.4f}")
                        print(f"      Precision:  {val_precision:.4f}")
                        print(f"      Recall:     {val_recall:.4f}")
                        
                        # Progress bar
                        if self.total_epochs > 0:
                            progress = (epoch + 1) / self.total_epochs
                            bar_length = 40
                            filled = int(bar_length * progress)
                            bar = "█" * filled + "░" * (bar_length - filled)
                            percentage = progress * 100
                            print(f"\n  Progress: [{bar}] {percentage:.1f}% ({epoch + 1}/{self.total_epochs})")
            
            time.sleep(5)  # Update every 5 seconds


def main():
    """Main monitoring loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor YOLO training progress')
    parser.add_argument('--logs-dir', type=str, default='training/logs',
                       help='Path to training logs directory')
    parser.add_argument('--total-epochs', type=int, default=50,
                       help='Total number of epochs')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(logs_dir=args.logs_dir)
    monitor.total_epochs = args.total_epochs
    
    try:
        monitor.display_progress()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")


if __name__ == "__main__":
    main()
