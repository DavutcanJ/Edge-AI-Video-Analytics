"""
Performance metrics logger with JSON export.
Tracks latency, throughput, and GPU metrics.
"""

import time
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
from collections import deque
from datetime import datetime


class MetricsLogger:
    """
    Log and track performance metrics.
    """
    
    def __init__(self, window_size: int = 100, save_path: str = "logs/metrics.json"):
        """
        Initialize metrics logger.
        
        Args:
            window_size: Moving window size for metrics
            save_path: Path to save metrics JSON
        """
        self.window_size = window_size
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.latencies = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.gpu_utils = deque(maxlen=window_size)
        self.gpu_memory = deque(maxlen=window_size)
        
        # Cumulative stats
        self.total_inferences = 0
        self.start_time = time.time()
        
        # Historical data for persistence
        self.history = []
    
    def log_latency(self, latency_ms: float):
        """
        Log inference latency.
        
        Args:
            latency_ms: Inference time in milliseconds
        """
        self.latencies.append(latency_ms)
        self.timestamps.append(time.time())
        self.total_inferences += 1
    
    def log_gpu_metrics(self, utilization: float, memory_used_mb: float):
        """
        Log GPU metrics.
        
        Args:
            utilization: GPU utilization percentage
            memory_used_mb: GPU memory used in MB
        """
        self.gpu_utils.append(utilization)
        self.gpu_memory.append(memory_used_mb)
    
    def get_stats(self) -> Dict:
        """
        Get current statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.latencies:
            return {
                'total_inferences': 0,
                'avg_latency_ms': 0.0,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0,
                'throughput_fps': 0.0
            }
        
        latencies = list(self.latencies)
        
        stats = {
            'total_inferences': self.total_inferences,
            'avg_latency_ms': float(np.mean(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p90_latency_ms': float(np.percentile(latencies, 90)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
        }
        
        # Calculate throughput
        if len(self.timestamps) > 1:
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span > 0:
                stats['throughput_fps'] = len(self.timestamps) / time_span
            else:
                stats['throughput_fps'] = 0.0
        else:
            stats['throughput_fps'] = 0.0
        
        # Add GPU stats if available
        if self.gpu_utils:
            stats['avg_gpu_util'] = float(np.mean(list(self.gpu_utils)))
            stats['max_gpu_util'] = float(np.max(list(self.gpu_utils)))
        
        if self.gpu_memory:
            stats['avg_gpu_memory_mb'] = float(np.mean(list(self.gpu_memory)))
            stats['max_gpu_memory_mb'] = float(np.max(list(self.gpu_memory)))
        
        # Uptime
        stats['uptime_seconds'] = time.time() - self.start_time
        
        return stats
    
    def get_moving_average(self, window: int = None) -> float:
        """
        Get moving average latency.
        
        Args:
            window: Window size (default: use configured window_size)
        
        Returns:
            Moving average latency in ms
        """
        if not self.latencies:
            return 0.0
        
        if window is None:
            window = self.window_size
        
        recent = list(self.latencies)[-window:]
        return float(np.mean(recent))
    
    def save_metrics(self, path: str = None):
        """
        Save metrics to JSON file.
        
        Args:
            path: Optional path to save (default: use configured path)
        """
        save_path = Path(path) if path else self.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current stats
        stats = self.get_stats()
        
        # Add metadata
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'recent_latencies': list(self.latencies),
            'window_size': self.window_size
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"[INFO] Metrics saved to: {save_path}")
    
    def load_metrics(self, path: str = None):
        """
        Load metrics from JSON file.
        
        Args:
            path: Path to load from
        """
        load_path = Path(path) if path else self.save_path
        
        if not load_path.exists():
            print(f"[WARNING] Metrics file not found: {load_path}")
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Restore recent latencies
        if 'recent_latencies' in data:
            self.latencies = deque(data['recent_latencies'], maxlen=self.window_size)
        
        print(f"[INFO] Metrics loaded from: {load_path}")
    
    def reset(self):
        """Reset all metrics."""
        self.latencies.clear()
        self.timestamps.clear()
        self.gpu_utils.clear()
        self.gpu_memory.clear()
        self.total_inferences = 0
        self.start_time = time.time()
        self.history.clear()
    
    def print_summary(self):
        """Print metrics summary."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("PERFORMANCE METRICS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Inferences: {stats['total_inferences']}")
        print(f"Uptime:           {stats['uptime_seconds']:.1f}s")
        print(f"\nLatency Statistics:")
        print(f"  Average:  {stats['avg_latency_ms']:.2f} ms")
        print(f"  Min:      {stats['min_latency_ms']:.2f} ms")
        print(f"  Max:      {stats['max_latency_ms']:.2f} ms")
        print(f"  Std Dev:  {stats['std_latency_ms']:.2f} ms")
        print(f"  P50:      {stats['p50_latency_ms']:.2f} ms")
        print(f"  P95:      {stats['p95_latency_ms']:.2f} ms")
        print(f"  P99:      {stats['p99_latency_ms']:.2f} ms")
        print(f"\nThroughput:       {stats['throughput_fps']:.2f} FPS")
        
        if 'avg_gpu_util' in stats:
            print(f"\nGPU Utilization:")
            print(f"  Average:  {stats['avg_gpu_util']:.1f}%")
            print(f"  Max:      {stats['max_gpu_util']:.1f}%")
        
        if 'avg_gpu_memory_mb' in stats:
            print(f"\nGPU Memory:")
            print(f"  Average:  {stats['avg_gpu_memory_mb']:.1f} MB")
            print(f"  Max:      {stats['max_gpu_memory_mb']:.1f} MB")
        
        print(f"{'='*60}\n")
