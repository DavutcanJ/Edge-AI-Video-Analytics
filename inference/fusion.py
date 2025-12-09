"""
Fusion module for combining detector and tracker outputs.
Implements intelligent fusion strategies with drift correction.
"""

import numpy as np
from typing import List, Tuple, Dict
from .detector import Detection
from .tracker import Track


class DetectionTrackerFusion:
    """
    Fuse detector and tracker outputs for robust tracking.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        confidence_boost: float = 0.1,
        max_disappeared: int = 30
    ):
        """
        Initialize fusion module.
        
        Args:
            iou_threshold: IoU threshold for matching
            confidence_boost: Boost confidence for matched tracks
            max_disappeared: Maximum frames before removing track
        """
        self.iou_threshold = iou_threshold
        self.confidence_boost = confidence_boost
        self.max_disappeared = max_disappeared
        
        self.track_history: Dict[int, List[Track]] = {}
    
    def fuse(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> List[Track]:
        """
        Fuse detections with tracks.
        
        Args:
            detections: List of Detection objects
            tracks: List of Track objects
        
        Returns:
            Fused list of Track objects
        """
        if not detections:
            return tracks
        
        if not tracks:
            # Convert detections to tracks
            return self._detections_to_tracks(detections)
        
        # Match detections to tracks
        matched_pairs, unmatched_dets, unmatched_tracks = self._match(detections, tracks)
        
        fused_tracks = []
        
        # Update matched tracks with detection information
        for det_idx, trk_idx in matched_pairs:
            track = tracks[trk_idx]
            detection = detections[det_idx]
            
            # Update track with detection
            track.bbox = detection.bbox
            track.confidence = min(1.0, detection.confidence + self.confidence_boost)
            track.class_id = detection.class_id
            track.time_since_update = 0
            track.state = "confirmed"
            
            fused_tracks.append(track)
        
        # Keep unmatched tracks (up to max_disappeared)
        for trk_idx in unmatched_tracks:
            track = tracks[trk_idx]
            if track.time_since_update < self.max_disappeared:
                fused_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            new_track = Track(
                track_id=-1,  # Will be assigned by tracker
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                state="tentative"
            )
            fused_tracks.append(new_track)
        
        return fused_tracks
    
    def _match(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Compute IoU matrix
        det_boxes = np.array([det.bbox for det in detections])
        trk_boxes = np.array([trk.bbox for trk in tracks])
        
        iou_matrix = self._compute_iou(det_boxes, trk_boxes)
        
        # Greedy matching
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Sort by IoU descending
        flat_indices = np.argsort(iou_matrix.ravel())[::-1]
        
        for flat_idx in flat_indices:
            det_idx = flat_idx // len(tracks)
            trk_idx = flat_idx % len(tracks)
            
            if iou_matrix[det_idx, trk_idx] < self.iou_threshold:
                break
            
            if det_idx in unmatched_dets and trk_idx in unmatched_tracks:
                matched_pairs.append((det_idx, trk_idx))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(trk_idx)
        
        return matched_pairs, unmatched_dets, unmatched_tracks
    
    @staticmethod
    def _compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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
    
    @staticmethod
    def _detections_to_tracks(detections: List[Detection]) -> List[Track]:
        """Convert detections to tracks."""
        tracks = []
        for det in detections:
            track = Track(
                track_id=-1,
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                state="tentative"
            )
            tracks.append(track)
        return tracks
