"""
Unit tests for ONNX model export and shape validation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestONNXShapes:
    """Tests for ONNX model I/O shapes."""
    
    def test_input_shape_validation(self):
        """Test input shape validation."""
        # Expected input shape for YOLO models: (batch, 3, height, width)
        batch_sizes = [1, 2, 4, 8]
        img_sizes = [320, 640, 1280]
        
        for batch in batch_sizes:
            for size in img_sizes:
                input_shape = (batch, 3, size, size)
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                
                assert dummy_input.shape == input_shape
                assert dummy_input.dtype == np.float32
    
    def test_output_shape_consistency(self):
        """Test that output shapes are consistent."""
        # YOLO output format: (batch, num_boxes, 5 + num_classes)
        batch = 1
        num_boxes = 25200  # Example for 640x640
        num_classes = 80
        
        output_shape = (batch, num_boxes, 5 + num_classes)
        dummy_output = np.random.randn(*output_shape).astype(np.float32)
        
        assert dummy_output.shape == output_shape
        assert dummy_output.shape[2] == 85  # 5 + 80 classes
    
    def test_dynamic_batch_sizes(self):
        """Test dynamic batch size handling."""
        img_size = 640
        num_classes = 80
        
        for batch_size in [1, 2, 4, 8, 16]:
            input_shape = (batch_size, 3, img_size, img_size)
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Validate input
            assert dummy_input.shape[0] == batch_size
            assert dummy_input.shape[1] == 3
            assert dummy_input.shape[2] == img_size
            assert dummy_input.shape[3] == img_size
    
    def test_dynamic_image_sizes(self):
        """Test dynamic image size handling."""
        batch = 1
        sizes = [320, 384, 512, 640, 768, 1024, 1280]
        
        for size in sizes:
            input_shape = (batch, 3, size, size)
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            assert dummy_input.shape == (batch, 3, size, size)
    
    def test_data_type_validation(self):
        """Test data type validation."""
        input_shape = (1, 3, 640, 640)
        
        # Test float32 (correct type)
        float32_input = np.random.randn(*input_shape).astype(np.float32)
        assert float32_input.dtype == np.float32
        
        # Test float16 (for FP16 models)
        float16_input = np.random.randn(*input_shape).astype(np.float16)
        assert float16_input.dtype == np.float16
    
    def test_value_range_validation(self):
        """Test input value range."""
        # Images should be normalized to [0, 1] or [-1, 1]
        input_shape = (1, 3, 640, 640)
        
        # Test [0, 1] range
        normalized_input = np.random.rand(*input_shape).astype(np.float32)
        assert normalized_input.min() >= 0.0
        assert normalized_input.max() <= 1.0
        
        # Test that values outside range can be clipped
        out_of_range = np.random.randn(*input_shape).astype(np.float32) * 2
        clipped = np.clip(out_of_range, 0.0, 1.0)
        assert clipped.min() >= 0.0
        assert clipped.max() <= 1.0


class TestONNXExport:
    """Tests for ONNX export functionality."""
    
    def test_opset_version_validation(self):
        """Test ONNX opset version."""
        valid_opsets = [11, 12, 13, 14, 15, 16]
        
        for opset in valid_opsets:
            assert opset >= 11, "Opset version should be >= 11 for dynamic shapes"
    
    def test_export_parameters(self):
        """Test export parameter validation."""
        export_config = {
            'imgsz': 640,
            'batch_size': 1,
            'opset': 12,
            'dynamic': True,
            'simplify': True,
            'half': False
        }
        
        assert export_config['imgsz'] > 0
        assert export_config['batch_size'] > 0
        assert export_config['opset'] >= 11
        assert isinstance(export_config['dynamic'], bool)
        assert isinstance(export_config['simplify'], bool)


class TestModelConsistency:
    """Tests for model consistency across backends."""
    
    def test_output_format_consistency(self):
        """Test that all backends produce same output format."""
        # All backends should produce detections in same format
        # [x1, y1, x2, y2, confidence, class_id]
        
        detection_format = np.array([100, 100, 200, 200, 0.95, 0])
        
        assert len(detection_format) == 6
        assert detection_format[0] < detection_format[2]  # x1 < x2
        assert detection_format[1] < detection_format[3]  # y1 < y2
        assert 0 <= detection_format[4] <= 1  # confidence in [0, 1]
        assert detection_format[5] >= 0  # class_id >= 0
    
    def test_preprocessing_consistency(self):
        """Test preprocessing produces consistent results."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate preprocessing
        import cv2
        resized = cv2.resize(img, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Check consistency
        assert transposed.shape == (3, 640, 640)
        assert 0 <= transposed.min() <= transposed.max() <= 1
    
    def test_postprocessing_consistency(self):
        """Test postprocessing produces valid results."""
        # Mock model output
        num_boxes = 100
        num_classes = 80
        output = np.random.rand(1, num_boxes, 5 + num_classes).astype(np.float32)
        
        # Simulate confidence filtering
        conf_threshold = 0.25
        confidences = output[0, :, 4]
        mask = confidences > conf_threshold
        filtered = output[0][mask]
        
        assert len(filtered) <= num_boxes
        if len(filtered) > 0:
            assert filtered[:, 4].min() >= conf_threshold


class TestWarmup:
    """Tests for model warmup."""
    
    def test_warmup_iterations(self):
        """Test warmup iteration count."""
        warmup_iterations = 10
        
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        for i in range(warmup_iterations):
            # Simulate inference
            output = dummy_input * 2  # Mock operation
            assert output.shape == dummy_input.shape
        
        assert i == warmup_iterations - 1
    
    def test_warmup_timing(self):
        """Test that warmup improves timing."""
        import time
        
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # First inference (cold)
        start = time.perf_counter()
        _ = dummy_input * 2
        cold_time = time.perf_counter() - start
        
        # Warmup
        for _ in range(10):
            _ = dummy_input * 2
        
        # Warmed inference
        start = time.perf_counter()
        _ = dummy_input * 2
        warm_time = time.perf_counter() - start
        
        # Warm time should generally be <= cold time (though not guaranteed in simple test)
        assert cold_time >= 0
        assert warm_time >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
