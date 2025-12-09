"""
FPS (Frames Per Second) meter for performance tracking.
"""

import time
from collections import deque
from typing import Optional


class FPSMeter:
    """
    Measure and track frames per second.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS meter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = None
    
    def tick(self):
        """Record a frame."""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        self.timestamps.append(current_time)
        self.frame_count += 1
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current FPS based on recent frames
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return len(self.timestamps) / time_span
        
        return 0.0
    
    def get_average_fps(self) -> float:
        """
        Get average FPS since start.
        
        Returns:
            Average FPS
        """
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        
        return 0.0
    
    def reset(self):
        """Reset the FPS meter."""
        self.timestamps.clear()
        self.frame_count = 0
        self.start_time = None
    
    def __str__(self) -> str:
        """String representation."""
        return f"FPS: {self.get_fps():.2f} (avg: {self.get_average_fps():.2f})"
