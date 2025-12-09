"""
Simple dashboard for visualizing performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict
import json
from pathlib import Path


class PerformanceDashboard:
    """
    Real-time performance visualization dashboard.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize dashboard.
        
        Args:
            figsize: Figure size (width, height)
        """
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('Edge AI Performance Dashboard', fontsize=16)
        
        # Create subplots
        self.ax_latency = self.fig.add_subplot(2, 2, 1)
        self.ax_throughput = self.fig.add_subplot(2, 2, 2)
        self.ax_gpu = self.fig.add_subplot(2, 2, 3)
        self.ax_hist = self.fig.add_subplot(2, 2, 4)
        
        # Data storage
        self.latencies: List[float] = []
        self.throughputs: List[float] = []
        self.gpu_utils: List[float] = []
        self.timestamps: List[float] = []
        
        # Setup plots
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup plot styles and labels."""
        # Latency plot
        self.ax_latency.set_title('Inference Latency Over Time')
        self.ax_latency.set_xlabel('Time (s)')
        self.ax_latency.set_ylabel('Latency (ms)')
        self.ax_latency.grid(True, alpha=0.3)
        
        # Throughput plot
        self.ax_throughput.set_title('Throughput (FPS)')
        self.ax_throughput.set_xlabel('Time (s)')
        self.ax_throughput.set_ylabel('FPS')
        self.ax_throughput.grid(True, alpha=0.3)
        
        # GPU utilization plot
        self.ax_gpu.set_title('GPU Utilization')
        self.ax_gpu.set_xlabel('Time (s)')
        self.ax_gpu.set_ylabel('Utilization (%)')
        self.ax_gpu.set_ylim(0, 100)
        self.ax_gpu.grid(True, alpha=0.3)
        
        # Latency histogram
        self.ax_hist.set_title('Latency Distribution')
        self.ax_hist.set_xlabel('Latency (ms)')
        self.ax_hist.set_ylabel('Frequency')
        self.ax_hist.grid(True, alpha=0.3)
    
    def update(self, latency: float, throughput: float, gpu_util: float = None, timestamp: float = None):
        """
        Update dashboard with new data point.
        
        Args:
            latency: Inference latency in ms
            throughput: Throughput in FPS
            gpu_util: GPU utilization percentage
            timestamp: Timestamp (default: current time)
        """
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        self.latencies.append(latency)
        self.throughputs.append(throughput)
        self.timestamps.append(timestamp)
        
        if gpu_util is not None:
            self.gpu_utils.append(gpu_util)
        
        # Keep only recent data (last 100 points)
        max_points = 100
        if len(self.latencies) > max_points:
            self.latencies = self.latencies[-max_points:]
            self.throughputs = self.throughputs[-max_points:]
            self.timestamps = self.timestamps[-max_points:]
            if self.gpu_utils:
                self.gpu_utils = self.gpu_utils[-max_points:]
    
    def render(self, save_path: str = None):
        """
        Render the dashboard.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.latencies:
            print("[WARNING] No data to display")
            return
        
        # Normalize timestamps to start from 0
        time_axis = np.array(self.timestamps) - self.timestamps[0]
        
        # Clear previous plots
        self.ax_latency.clear()
        self.ax_throughput.clear()
        self.ax_gpu.clear()
        self.ax_hist.clear()
        
        # Re-setup plots
        self._setup_plots()
        
        # Plot latency
        self.ax_latency.plot(time_axis, self.latencies, 'b-', linewidth=1.5, label='Latency')
        self.ax_latency.axhline(y=np.mean(self.latencies), color='r', linestyle='--', 
                                label=f'Mean: {np.mean(self.latencies):.2f}ms')
        self.ax_latency.legend()
        
        # Plot throughput
        self.ax_throughput.plot(time_axis, self.throughputs, 'g-', linewidth=1.5, label='FPS')
        self.ax_throughput.axhline(y=np.mean(self.throughputs), color='r', linestyle='--',
                                   label=f'Mean: {np.mean(self.throughputs):.2f} FPS')
        self.ax_throughput.legend()
        
        # Plot GPU utilization
        if self.gpu_utils:
            self.ax_gpu.plot(time_axis[:len(self.gpu_utils)], self.gpu_utils, 'orange', 
                            linewidth=1.5, label='GPU Util')
            self.ax_gpu.axhline(y=np.mean(self.gpu_utils), color='r', linestyle='--',
                               label=f'Mean: {np.mean(self.gpu_utils):.1f}%')
            self.ax_gpu.legend()
        
        # Plot latency histogram
        self.ax_hist.hist(self.latencies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add statistics text
        stats_text = f'Stats:\n'
        stats_text += f'Min: {np.min(self.latencies):.2f}ms\n'
        stats_text += f'Max: {np.max(self.latencies):.2f}ms\n'
        stats_text += f'P50: {np.percentile(self.latencies, 50):.2f}ms\n'
        stats_text += f'P95: {np.percentile(self.latencies, 95):.2f}ms'
        
        self.ax_hist.text(0.7, 0.7, stats_text, transform=self.ax_hist.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Dashboard saved to: {save_path}")
        else:
            plt.show()
    
    def load_from_json(self, json_path: str):
        """
        Load metrics from JSON file.
        
        Args:
            json_path: Path to metrics JSON file
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'recent_latencies' in data:
            self.latencies = data['recent_latencies']
            # Generate synthetic timestamps and throughputs for visualization
            self.timestamps = list(range(len(self.latencies)))
            self.throughputs = [1000.0 / lat for lat in self.latencies]


def create_dashboard_from_metrics(metrics_path: str, output_path: str = None):
    """
    Create and save dashboard from metrics file.
    
    Args:
        metrics_path: Path to metrics JSON
        output_path: Output path for dashboard image
    """
    dashboard = PerformanceDashboard()
    dashboard.load_from_json(metrics_path)
    
    if output_path is None:
        output_path = Path(metrics_path).with_suffix('.png')
    
    dashboard.render(save_path=str(output_path))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance dashboard')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for dashboard image')
    
    args = parser.parse_args()
    
    create_dashboard_from_metrics(args.metrics, args.output)
