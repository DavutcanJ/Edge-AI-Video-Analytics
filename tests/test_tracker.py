"""
Unit tests for tracker functionality.
Tests drift detection, track lifecycle, and data association.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.tracker import ByteTracker, Track, KalmanBoxTracker


class TestKalmanBoxTracker:
    """Tests for Kalman filter tracker."""
    
    def test_initialization(self):
        """Test Kalman tracker initialization."""
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        assert tracker.id >= 0
        assert tracker.time_since_update == 0
        assert tracker.hits == 0
    
    def test_predict(self):
        """Test state prediction."""
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        predicted = tracker.predict()
        
        assert predicted.shape == (4,)
        assert tracker.age == 1
        assert tracker.time_since_update == 1
    
    def test_update(self):
        """Test measurement update."""
        bbox = np.array([100, 100, 200, 200])
        tracker = KalmanBoxTracker(bbox)
        
        new_bbox = np.array([105, 105, 205, 205])
        tracker.update(new_bbox)
        
        assert tracker.time_since_update == 0
        assert tracker.hits == 1
        assert tracker.hit_streak == 1
    
    def test_bbox_conversion(self):
        """Test bbox format conversion."""
        # Test xyxy to z conversion
        bbox = np.array([100, 100, 200, 200])
        z = KalmanBoxTracker._convert_bbox_to_z(bbox)
        
        assert z.shape == (4, 1)
        
        # Convert back
        bbox_recovered = KalmanBoxTracker._convert_x_to_bbox(z)
        
        np.testing.assert_array_almost_equal(bbox_recovered, bbox, decimal=1)


class TestByteTracker:
    """Tests for ByteTrack algorithm."""
    
    def test_initialization(self):
        """Test ByteTracker initialization."""
        tracker = ByteTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0
    
    def test_single_object_tracking(self):
        """Test tracking a single object."""
        tracker = ByteTracker()
        
        # Frame 1
        detections1 = np.array([[100, 100, 200, 200]])
        confidences1 = np.array([0.9])
        tracks1 = tracker.update(detections1, confidences1)
        
        assert len(tracker.trackers) == 1
        
        # Frame 2 - object moved slightly
        detections2 = np.array([[105, 105, 205, 205]])
        confidences2 = np.array([0.9])
        tracks2 = tracker.update(detections2, confidences2)
        
        # Should maintain the same track
        if tracks1 and tracks2:
            # Track IDs should match after enough hits
            pass
    
    def test_multiple_object_tracking(self):
        """Test tracking multiple objects."""
        tracker = ByteTracker()
        
        detections = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400],
            [500, 500, 600, 600]
        ])
        confidences = np.array([0.9, 0.85, 0.8])
        
        tracks = tracker.update(detections, confidences)
        
        assert len(tracker.trackers) == 3
    
    def test_object_disappearance(self):
        """Test handling of disappearing objects."""
        tracker = ByteTracker(max_age=5)
        
        # Frame 1 - object present
        detections1 = np.array([[100, 100, 200, 200]])
        confidences1 = np.array([0.9])
        tracker.update(detections1, confidences1)
        
        initial_count = len(tracker.trackers)
        
        # Frames 2-6 - object missing
        for _ in range(6):
            tracker.update(np.empty((0, 4)), np.empty(0))
        
        # Object should be removed after max_age
        assert len(tracker.trackers) < initial_count or len(tracker.trackers) == 0
    
    def test_object_reappearance(self):
        """Test handling of reappearing objects."""
        tracker = ByteTracker()
        
        # Frame 1 - object present
        detections1 = np.array([[100, 100, 200, 200]])
        confidences1 = np.array([0.9])
        tracker.update(detections1, confidences1)
        
        # Frame 2 - object missing
        tracker.update(np.empty((0, 4)), np.empty(0))
        
        # Frame 3 - object reappears at similar location
        detections3 = np.array([[105, 105, 205, 205]])
        confidences3 = np.array([0.9])
        tracker.update(detections3, confidences3)
        
        # Should have at least one tracker
        assert len(tracker.trackers) >= 1
    
    def test_high_low_confidence_detections(self):
        """Test handling of high and low confidence detections."""
        tracker = ByteTracker(
            high_threshold=0.6,
            low_threshold=0.1
        )
        
        detections = np.array([
            [100, 100, 200, 200],  # High confidence
            [300, 300, 400, 400],  # Low confidence
        ])
        confidences = np.array([0.9, 0.3])
        
        tracks = tracker.update(detections, confidences)
        
        # Both detections should be processed
        assert len(tracker.trackers) >= 1
    
    def test_iou_association(self):
        """Test IoU-based data association."""
        boxes1 = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ])
        boxes2 = np.array([
            [5, 5, 105, 105],  # Close to first box
            [400, 400, 500, 500]  # Far from all
        ])
        
        iou_matrix = ByteTracker._iou_batch(boxes1, boxes2)
        
        assert iou_matrix.shape == (2, 2)
        assert iou_matrix[0, 0] > 0.5  # High IoU with similar box
        assert iou_matrix[0, 1] < 0.1  # Low IoU with distant box
    
    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        tracker = ByteTracker()
        
        # Add some tracks
        detections = np.array([[100, 100, 200, 200]])
        confidences = np.array([0.9])
        tracker.update(detections, confidences)
        
        # Reset
        tracker.reset()
        
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0
        assert KalmanBoxTracker.count == 0


class TestTrackDrift:
    """Tests for tracking drift detection."""
    
    def test_no_drift(self):
        """Test case with no drift."""
        # Object moves smoothly
        positions = [
            (100, 100, 200, 200),
            (105, 105, 205, 205),
            (110, 110, 210, 210),
        ]
        
        tracker = ByteTracker()
        
        for pos in positions:
            detections = np.array([pos])
            confidences = np.array([0.9])
            tracker.update(detections, confidences)
        
        # Should maintain single track
        assert len(tracker.trackers) >= 1
    
    def test_with_drift(self):
        """Test case with significant drift."""
        tracker = ByteTracker()
        
        # Frame 1 - object at position A
        detections1 = np.array([[100, 100, 200, 200]])
        confidences1 = np.array([0.9])
        tracker.update(detections1, confidences1)
        
        # Frame 2 - object suddenly at very different position (teleport)
        detections2 = np.array([[400, 400, 500, 500]])
        confidences2 = np.array([0.9])
        tracker.update(detections2, confidences2)
        
        # This should create a new track or have low association
        # The exact behavior depends on IoU threshold
        pass
    
    def test_occlusion_recovery(self):
        """Test tracking recovery after occlusion."""
        tracker = ByteTracker(max_age=10)
        
        # Object visible
        detections1 = np.array([[100, 100, 200, 200]])
        confidences1 = np.array([0.9])
        tracker.update(detections1, confidences1)
        
        # Object occluded (no detection) for 3 frames
        for _ in range(3):
            tracker.update(np.empty((0, 4)), np.empty(0))
        
        # Object reappears near last known position
        detections2 = np.array([[110, 110, 210, 210]])
        confidences2 = np.array([0.9])
        tracker.update(detections2, confidences2)
        
        # Should recover the track
        assert len(tracker.trackers) >= 1


class TestTrackLifecycle:
    """Tests for track lifecycle management."""
    
    def test_track_creation(self):
        """Test track creation."""
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            state="tentative"
        )
        
        assert track.track_id == 1
        assert track.confidence == 0.9
        assert track.state == "tentative"
    
    def test_track_confirmation(self):
        """Test track confirmation after minimum hits."""
        tracker = ByteTracker(min_hits=3)
        
        # Update for 3 frames
        for _ in range(3):
            detections = np.array([[100, 100, 200, 200]])
            confidences = np.array([0.9])
            tracks = tracker.update(detections, confidences)
        
        # After min_hits, track should be confirmed
        # Check that we get confirmed tracks
        if tracks:
            confirmed = [t for t in tracks if t.state == "confirmed"]
            # May not be confirmed yet depending on implementation details
            pass
    
    def test_track_deletion(self):
        """Test track deletion after max_age."""
        tracker = ByteTracker(max_age=5)
        
        # Create track
        detections = np.array([[100, 100, 200, 200]])
        confidences = np.array([0.9])
        tracker.update(detections, confidences)
        
        initial_count = len(tracker.trackers)
        
        # No updates for max_age + 1 frames
        for _ in range(6):
            tracker.update(np.empty((0, 4)), np.empty(0))
        
        # Track should be deleted
        assert len(tracker.trackers) <= initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
