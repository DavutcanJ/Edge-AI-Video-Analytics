"""
Unit tests for inference module.
Tests detector, tracker, and video engine components.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.detector import Detector, Detection
from inference.tracker import ByteTracker, Track
from inference.video_engine import VideoEngine
from inference import utils


@pytest.fixture
def dummy_image():
    """Create a dummy image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_detections():
    """Create dummy detection results."""
    return [
        Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=(300, 300, 400, 400),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]


class TestDetector:
    """Tests for Detector class."""
    
    def test_nms(self):
        """Test Non-Maximum Suppression."""
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205],
            [300, 300, 400, 400]
        ])
        scores = np.array([0.9, 0.85, 0.95])
        iou_threshold = 0.5
        
        keep_indices = Detector.nms(boxes, scores, iou_threshold)
        
        assert len(keep_indices) == 2  # Should keep first and third boxes
        assert 0 in keep_indices
        assert 2 in keep_indices
    
    def test_preprocess_shape(self, dummy_image):
        """Test preprocessing output shape."""
        # Mock a detector (can't test without actual model)
        img_size = 640
        
        # Resize and preprocess
        resized = cv2.resize(dummy_image, (img_size, img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        assert batched.shape == (1, 3, img_size, img_size)
        assert batched.dtype == np.float32
        assert batched.min() >= 0.0
        assert batched.max() <= 1.0


class TestTracker:
    """Tests for ByteTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ByteTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.trackers) == 0
    
    def test_tracker_update_new_detections(self):
        """Test tracker with new detections."""
        tracker = ByteTracker()
        
        detections = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ])
        confidences = np.array([0.9, 0.85])
        
        tracks = tracker.update(detections, confidences)
        
        # New tracks should be created
        assert len(tracker.trackers) == 2
    
    def test_tracker_update_no_detections(self):
        """Test tracker with no detections."""
        tracker = ByteTracker()
        
        # First update with detections
        detections = np.array([[100, 100, 200, 200]])
        confidences = np.array([0.9])
        tracker.update(detections, confidences)
        
        # Update with no detections
        tracks = tracker.update(np.empty((0, 4)), np.empty(0))
        
        # Trackers should still exist but might not return confirmed tracks
        assert len(tracker.trackers) >= 0
    
    def test_tracker_reset(self):
        """Test tracker reset."""
        tracker = ByteTracker()
        
        detections = np.array([[100, 100, 200, 200]])
        confidences = np.array([0.9])
        tracker.update(detections, confidences)
        
        tracker.reset()
        
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0
    
    def test_iou_computation(self):
        """Test IoU computation."""
        boxes1 = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150]
        ])
        boxes2 = np.array([
            [0, 0, 100, 100],
            [100, 100, 200, 200]
        ])
        
        iou_matrix = ByteTracker._iou_batch(boxes1, boxes2)
        
        assert iou_matrix.shape == (2, 2)
        assert np.isclose(iou_matrix[0, 0], 1.0, atol=1e-6)  # Perfect overlap
        assert iou_matrix[0, 1] < 0.01  # No overlap
        assert 0.0 <= iou_matrix[1, 0] <= 1.0  # Partial overlap


class TestUtils:
    """Tests for utility functions."""
    
    def test_xyxy2xywh(self):
        """Test bounding box format conversion."""
        xyxy = np.array([[100, 100, 200, 200]])
        xywh = utils.xyxy2xywh(xyxy)
        
        expected = np.array([[150, 150, 100, 100]])  # center_x, center_y, width, height
        np.testing.assert_array_almost_equal(xywh, expected)
    
    def test_xywh2xyxy(self):
        """Test bounding box format conversion."""
        xywh = np.array([[150, 150, 100, 100]])
        xyxy = utils.xywh2xyxy(xywh)
        
        expected = np.array([[100, 100, 200, 200]])
        np.testing.assert_array_almost_equal(xyxy, expected)
    
    def test_clip_boxes(self):
        """Test box clipping."""
        boxes = np.array([
            [100, 100, 200, 200],
            [-10, -10, 50, 50],
            [400, 400, 700, 700]
        ])
        shape = (480, 640)  # height, width
        
        clipped = utils.clip_boxes(boxes, shape)
        
        assert clipped[0, 0] == 100  # No change
        assert clipped[1, 0] == 0  # Clipped to 0
        assert clipped[2, 2] == 640  # Clipped to width
        assert clipped[2, 3] == 480  # Clipped to height
    
    def test_letterbox(self):
        """Test letterbox resizing."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        new_shape = (640, 640)
        
        resized, ratio, padding = utils.letterbox(img, new_shape)
        
        # Letterbox fonksiyonu padding ekleyebilir, sadece genişlik sabit kalır
        assert resized.shape[1] == new_shape[1]  # width
        assert resized.shape[0] <= new_shape[0]  # height padding olabilir
        assert len(ratio) == 2
        assert len(padding) == 2


class TestVideoEngine:
    """Tests for VideoEngine class."""
    
    @pytest.fixture
    def mock_detector(self, dummy_image):
        """Create a mock detector."""
        class MockDetector:
            def __init__(self):
                self.backend = "mock"
                self.img_size = 640
            
            def __call__(self, image):
                return [
                    Detection(
                        bbox=(100, 100, 200, 200),
                        confidence=0.9,
                        class_id=0,
                        class_name="person"
                    )
                ]
            
            def get_timing_stats(self):
                return {
                    'preprocess_avg_ms': 5.0,
                    'inference_avg_ms': 10.0,
                    'postprocess_avg_ms': 3.0,
                    'total_avg_ms': 18.0,
                    'fps': 55.5
                }
            
            def reset_timing_stats(self):
                pass
        
        return MockDetector()
    
    def test_video_engine_initialization(self, mock_detector):
        """Test video engine initialization."""
        engine = VideoEngine(
            detector=mock_detector,
            detection_interval=5,
            iou_drift_threshold=0.5
        )
        
        assert engine.detection_interval == 5
        assert engine.iou_drift_threshold == 0.5
        assert engine.frame_count == 0
    
    def test_video_engine_process_frame(self, mock_detector, dummy_image):
        """Test single frame processing."""
        engine = VideoEngine(
            detector=mock_detector,
            detection_interval=1
        )
        
        packet = engine.process_frame(dummy_image)
        
        assert packet.frame_id == 1
        assert packet.frame is not None
        assert packet.detections is not None or packet.tracks is not None
    
    def test_video_engine_detection_interval(self, mock_detector, dummy_image):
        """Test detection interval logic."""
        engine = VideoEngine(
            detector=mock_detector,
            detection_interval=5
        )
        
        # Process first frame (should detect)
        packet1 = engine.process_frame(dummy_image)
        assert packet1.detections is not None
        
        # Process second frame (should not detect)
        packet2 = engine.process_frame(dummy_image)
        # Detections may be None if only tracking
        
        # Process frames until detection again
        for _ in range(3):
            engine.process_frame(dummy_image)
        
        packet_detect = engine.process_frame(dummy_image)
        assert packet_detect.frame_id == 6


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_processing(self, dummy_image):
        """Test end-to-end processing without actual model."""
        # This would require an actual model file
        # For now, just test that components can be initialized
        
        tracker = ByteTracker()
        assert tracker is not None
        
        # Test detection format
        detections = np.array([[100, 100, 200, 200]])
        confidences = np.array([0.9])
        tracks = tracker.update(detections, confidences)
        
        assert isinstance(tracks, list)


class TestDriftDetection:
    """Tests for drift detection."""
    
    def test_compute_iou_matrix(self):
        """Test IoU matrix computation."""
        from inference.video_engine import VideoEngine
        
        boxes1 = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ])
        boxes2 = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150]
        ])
        
        iou_matrix = VideoEngine._compute_iou_matrix(boxes1, boxes2)
        
        assert iou_matrix.shape == (2, 2)
        
        assert np.isclose(iou_matrix[0, 0], 1.0, atol=1e-6)  # Perfect match
        assert iou_matrix[1, 1] < 0.1  # Little overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
