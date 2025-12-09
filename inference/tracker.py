"""
Object tracking module supporting multiple tracking algorithms.
Implements ByteTrack, DeepSORT-style tracking, and OpenCV trackers.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


@dataclass
class Track:
    """Object track container."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    state: str = "tentative"  # tentative, confirmed, deleted


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, var]
    """
    
    count = 0
    
    def __init__(self, bbox: np.ndarray):
        """
        Initialize Kalman filter tracker.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox: np.ndarray):
        """
        Update tracker with observed bbox.
        
        Args:
            bbox: Observed bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Advance state and return predicted bbox."""
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Get current state as bbox."""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        Convert bbox from [x1, y1, x2, y2] to [x_center, y_center, area, aspect_ratio].
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
        """
        Convert state [x_center, y_center, area, aspect_ratio] to bbox [x1, y1, x2, y2].
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0
        ]).reshape((1, 4))[0]


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
    Simplified implementation.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1
    ):
        """
        Initialize ByteTracker.
        
        Args:
            max_age: Maximum frames to keep alive without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for matching
            high_threshold: High confidence threshold
            low_threshold: Low confidence threshold
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray = None
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: Array of detections [N, 4] in format (x1, y1, x2, y2)
            confidences: Array of confidence scores [N]
        
        Returns:
            List of active Track objects
        """
        self.frame_count += 1
        
        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Split detections by confidence
        if confidences is not None:
            high_dets = detections[confidences > self.high_threshold]
            low_dets = detections[(confidences > self.low_threshold) & (confidences <= self.high_threshold)]
        else:
            high_dets = detections
            low_dets = np.empty((0, 4))
        
        # First association with high confidence detections
        matched, unmatched_dets, unmatched_trks = self._associate(high_dets, trks, self.iou_threshold)
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(high_dets[m[0]])
        
        # Second association with low confidence detections
        if len(low_dets) > 0 and len(unmatched_trks) > 0:
            unmatched_trks_bboxes = np.array([trks[i] for i in unmatched_trks])
            matched_low, unmatched_low, unmatched_trks = self._associate(
                low_dets, unmatched_trks_bboxes, self.iou_threshold
            )
            
            for m in matched_low:
                self.trackers[unmatched_trks[m[1]]].update(low_dets[m[0]])
        
        # Create new trackers for unmatched high confidence detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(high_dets[i])
            self.trackers.append(trk)
        
        # Remove dead tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            bbox = trk.get_state()
            
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                track = Track(
                    track_id=trk.id,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=1.0,
                    class_id=0,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    state="confirmed"
                )
                ret.append(track)
            
            i -= 1
            
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        return ret
    
    @staticmethod
    def _associate(
        detections: np.ndarray,
        trackers: np.ndarray,
        iou_threshold: float
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate detections to trackers using IoU.
        
        Returns:
            matched_indices, unmatched_detections, unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = ByteTracker._iou_batch(detections, trackers)
        
        # Hungarian algorithm
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack((row_ind, col_ind))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter matches by IoU threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                continue
            matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        # Get unmatched detections and trackers
        unmatched_detections = list(set(range(len(detections))) - set(matches[:, 0]))
        unmatched_trackers = list(set(range(len(trackers))) - set(matches[:, 1]))
        
        return matches, unmatched_detections, unmatched_trackers
    
    @staticmethod
    def _iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of bboxes.
        
        Args:
            bboxes1: [N, 4]
            bboxes2: [M, 4]
        
        Returns:
            IoU matrix [N, M]
        """
        xx1 = np.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])
        yy1 = np.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])
        xx2 = np.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
        yy2 = np.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        union = area1[:, None] + area2[None, :] - intersection
        
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def reset(self):
        """Reset tracker."""
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
