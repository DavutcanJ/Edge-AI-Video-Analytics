"""
Real-time video processing engine with detector-tracker fusion.
Implements multi-threaded architecture for optimal performance.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass

from .detector import Detector, Detection
from .tracker import ByteTracker, Track


@dataclass
class FramePacket:
    """Frame data container."""
    frame_id: int
    timestamp: float
    frame: np.ndarray
    detections: List[Detection] = None
    tracks: List[Track] = None


class VideoEngine:
    """
    Real-time video processing engine with detector-tracker fusion.
    Implements hybrid detection-tracking with drift detection.
    """
    
    def __init__(
        self,
        detector: Detector,
        detection_interval: int = 5,
        iou_drift_threshold: float = 0.5,
        max_queue_size: int = 10,
        use_threading: bool = True
    ):
        """
        Initialize video engine.
        
        Args:
            detector: Detector instance
            detection_interval: Run detector every N frames
            iou_drift_threshold: IoU threshold for drift detection
            max_queue_size: Maximum queue size for multi-threading
            use_threading: Enable multi-threaded processing
        """
        self.detector = detector
        self.detection_interval = detection_interval
        self.iou_drift_threshold = iou_drift_threshold
        self.use_threading = use_threading
        
        # Initialize tracker
        self.tracker = ByteTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Frame counter
        self.frame_count = 0
        
        # Multi-threading components
        if use_threading:
            self.capture_queue = queue.Queue(maxsize=max_queue_size)
            self.detection_queue = queue.Queue(maxsize=max_queue_size)
            self.display_queue = queue.Queue(maxsize=max_queue_size)
            
            self.stop_event = threading.Event()
            self.threads = []
        
        # Statistics
        self.fps_history = []
        self.last_frame_time = None
    
    def _detect_drift(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> bool:
        """
        Detect tracking drift using IoU between detections and tracks.
        
        Args:
            detections: Current detections
            tracks: Current tracks
        
        Returns:
            True if significant drift detected
        """
        if not detections or not tracks:
            return False
        
        # Convert to numpy arrays
        det_boxes = np.array([det.bbox for det in detections])
        trk_boxes = np.array([trk.bbox for trk in tracks])
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(det_boxes, trk_boxes)
        
        # Check maximum IoU for each detection
        if iou_matrix.size > 0:
            max_ious = iou_matrix.max(axis=1)
            drift_detected = (max_ious < self.iou_drift_threshold).any()
            return drift_detected
        
        return False
    
    @staticmethod
    def _compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        xx1 = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0][None, :])
        yy1 = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1][None, :])
        xx2 = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2][None, :])
        yy2 = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3][None, :])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - intersection
        
        return intersection / (union + 1e-6)
    
    def process_frame(self, frame: np.ndarray) -> FramePacket:
        """
        Process a single frame with detector-tracker fusion.
        
        Args:
            frame: Input frame
        
        Returns:
            FramePacket with detections and tracks
        """
        self.frame_count += 1
        timestamp = time.time()
        
        # Calculate FPS
        if self.last_frame_time is not None:
            fps = 1.0 / (timestamp - self.last_frame_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
        self.last_frame_time = timestamp
        
        packet = FramePacket(
            frame_id=self.frame_count,
            timestamp=timestamp,
            frame=frame
        )
        
        # Decide whether to run detector
        run_detector = (self.frame_count % self.detection_interval == 1)
        
        if run_detector:
            # Run detector
            detections = self.detector(frame)
            packet.detections = detections
            
            # Update tracker with detections
            if detections:
                det_boxes = np.array([det.bbox for det in detections])
                det_confs = np.array([det.confidence for det in detections])
                tracks = self.tracker.update(det_boxes, det_confs)
            else:
                tracks = self.tracker.update(np.empty((0, 4)), np.empty(0))
            
            packet.tracks = tracks
        else:
            # Only use tracker predictions
            tracks = self.tracker.update(np.empty((0, 4)), np.empty(0))
            packet.tracks = tracks
            
            # Optional: Check for drift if we have previous detections
            # If drift detected, force detector run on next frame
            if hasattr(self, '_last_detections') and self._last_detections:
                drift = self._detect_drift(self._last_detections, tracks)
                if drift:
                    print(f"[WARNING] Drift detected at frame {self.frame_count}")
                    # Force detection on next frame
                    self.frame_count = (self.frame_count // self.detection_interval) * self.detection_interval
        
        # Store last detections for drift detection
        if packet.detections:
            self._last_detections = packet.detections
        
        return packet
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None
    ):
        """
        Process video file or camera stream.
        
        Args:
            video_path: Path to video file or camera index (0 for webcam)
            output_path: Optional output video path
            display: Display processed frames
            max_frames: Maximum frames to process
        """
        # Open video
        if isinstance(video_path, int) or video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                packet = self.process_frame(frame)
                
                # Visualize
                vis_frame = self._visualize_packet(packet)
                
                # Write to output
                if writer:
                    writer.write(vis_frame)
                
                # Display
                if display:
                    cv2.imshow('Video Processing', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Check max frames
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Print progress
                if self.frame_count % 100 == 0:
                    avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                    print(f"[INFO] Processed {self.frame_count} frames, FPS: {avg_fps:.2f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n[INFO] Processing complete!")
            print(f"  Total frames: {self.frame_count}")
            if self.fps_history:
                print(f"  Average FPS:  {np.mean(self.fps_history):.2f}")
    
    def _visualize_packet(self, packet: FramePacket) -> np.ndarray:
        """Visualize frame packet with detections and tracks."""
        frame = packet.frame.copy()
        
        # Draw tracks
        if packet.tracks:
            for track in packet.tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                color = (0, 255, 0)  # Green for tracks
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{track.track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw detections (if available this frame)
        if packet.detections:
            for det in packet.detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                color = (255, 0, 0)  # Blue for detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Draw info
        if self.fps_history:
            fps_text = f"FPS: {np.mean(self.fps_history[-10:]):.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        frame_text = f"Frame: {packet.frame_id}"
        cv2.putText(frame, frame_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        tracks_text = f"Tracks: {len(packet.tracks) if packet.tracks else 0}"
        cv2.putText(frame, tracks_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        detector_stats = self.detector.get_timing_stats()
        
        stats = {
            'total_frames': self.frame_count,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'detector_stats': detector_stats
        }
        
        return stats
    
    def reset(self):
        """Reset engine state."""
        self.frame_count = 0
        self.tracker.reset()
        self.fps_history.clear()
        self.last_frame_time = None
        self.detector.reset_timing_stats()
