"""
Multi-backend object detector supporting PyTorch, ONNX, and TensorRT.
Provides consistent interface across different inference backends.
"""

import cv2
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result container."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str = ""


class Detector:
    """
    Multi-backend object detector.
    Supports PyTorch, ONNX Runtime, and TensorRT backends.
    """
    
    def __init__(
                
        self,
        model_path: str,
        backend: str = "pytorch",
        device: str = "cuda:0",
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: List[str] = None,
        warmup_iterations: int = 10
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to model file
            backend: Backend type ('pytorch', 'onnx', 'tensorrt')
            device: Device for PyTorch backend
            img_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            class_names: List of class names
            warmup_iterations: Number of warmup iterations
        """
   
                
        self.model_path = Path(model_path)
        self.backend = backend.lower()
        self.device = device
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        self.warmup_iterations = warmup_iterations
        
        # Timing statistics
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        
        # Load model based on backend
        print(f"[INFO] Initializing {self.backend.upper()} detector...")
        self._load_model()
        
        # Warmup
        self._warmup()
        
        print(f"[INFO] Detector ready: {self.backend.upper()}")

             # CUDA device check (if backend is tensorrt or pytorch and device contains 'cuda')
        if (self.backend in ["tensorrt", "pytorch"]) and ("cuda" in str(self.device)):
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device_count = torch.cuda.device_count()
                    device_name = torch.cuda.get_device_name(0)
                    print(f"[INFO] CUDA detected: {device_name} (total devices: {device_count})")
                else:
                    print("[WARNING] CUDA device not available! Inference will run on CPU.")
            except ImportError:
                print("[WARNING] torch not installed, cannot check CUDA device.")
    
    def _load_model(self):
        """Load model based on backend."""
        if self.backend == "pytorch":
            self._load_pytorch()
        elif self.backend == "onnx":
            self._load_onnx()
        elif self.backend == "tensorrt":
            self._load_tensorrt()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _load_pytorch(self):
        """Load PyTorch model."""
        from ultralytics import YOLO
        
        self.model = YOLO(str(self.model_path))
        self.model.model.to(self.device)
        self.model.model.eval()
        
        print(f"[OK] PyTorch model loaded on {self.device}")
    
    def _load_onnx(self):
        """Load ONNX model."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"[OK] ONNX model loaded with providers: {self.session.get_providers()}")
    
    def _load_tensorrt(self):
        """Load TensorRT engine (TensorRT 10+ IO Tensor API)."""
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(self.model_path, "rb") as f:
            self.runtime = trt.Runtime(self.TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # IO tensor indexleri (çoğu engine: 0=input, 1=output)
        self.input_idx = 0
        self.output_idx = 1

        in_name = self.engine.get_tensor_name(self.input_idx)
        out_name = self.engine.get_tensor_name(self.output_idx)

        # Input shape
        self.input_shape = (1, 3, self.img_size, self.img_size)

        # Dinamik shape varsa önce input shape set et (çok kritik)
        self.context.set_input_shape(in_name, self.input_shape)

        # Output shape'i context'ten al (engine.get_tensor_shape da olur ama context daha güvenli)
        self.output_shape = tuple(self.context.get_tensor_shape(out_name))

        # Allocate buffers
        input_size = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
        output_size = int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize

        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)

        # TRT10: binding list yerine tensor address set edilir
        self.context.set_tensor_address(in_name, int(self.d_input))
        self.context.set_tensor_address(out_name, int(self.d_output))

        self.stream = cuda.Stream()
        self.h_output = np.empty(self.output_shape, dtype=np.float32)

        print("[OK] TensorRT engine loaded")
        print(f"  Input tensor:  {in_name}  shape={self.input_shape}")
        print(f"  Output tensor: {out_name} shape={self.output_shape}")
    
    def _warmup(self):
        """Warmup the model."""
        print(f"[INFO] Warming up ({self.warmup_iterations} iterations)...")
        dummy_image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        for _ in range(self.warmup_iterations):
            self._infer(dummy_image)
        
        print("[OK] Warmup complete")
    
    def preprocess(self, image: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR, HWC)
        
        Returns:
            Preprocessed tensor/array
        """
        # Resize
        img = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Convert to tensor for PyTorch
        if self.backend == "pytorch":
            img = torch.from_numpy(img).to(self.device)
        
        return img
    
    def _infer_pytorch(self, preprocessed: torch.Tensor) -> np.ndarray:
        """PyTorch inference."""
        with torch.no_grad():
            output = self.model.model(preprocessed)
        
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        return output.cpu().numpy()
    
    def _infer_onnx(self, preprocessed: np.ndarray) -> np.ndarray:
        """ONNX Runtime inference."""
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
        return outputs[0]
    
    def _infer_tensorrt(self, preprocessed: np.ndarray) -> np.ndarray:
        """TensorRT inference."""
        preprocessed = np.ascontiguousarray(preprocessed)
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, preprocessed, self.stream)
        
        # Run inference
        # Use execute_async_v3 if available (TensorRT >= 10), else fallback to v2
        if hasattr(self.context, 'execute_async_v3'):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(stream_handle=self.stream.handle)
        
        # Copy output to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return self.h_output
    
    def _infer(self, image: np.ndarray) -> np.ndarray:
        """Internal inference method."""
        # Preprocess
        t0 = time.perf_counter()
        preprocessed = self.preprocess(image)
        t1 = time.perf_counter()
        self.preprocess_times.append((t1 - t0) * 1000)
        
        # Inference
        t2 = time.perf_counter()
        if self.backend == "pytorch":
            output = self._infer_pytorch(preprocessed)
        elif self.backend == "onnx":
            output = self._infer_onnx(preprocessed)
        elif self.backend == "tensorrt":
            output = self._infer_tensorrt(preprocessed)
        t3 = time.perf_counter()
        self.inference_times.append((t3 - t2) * 1000)
        
        return output
    
    def postprocess(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Postprocess model output to detections.
        
        Args:
            output: Model output
            orig_shape: Original image shape (H, W)
        
        Returns:
            List of Detection objects
        """
        t0 = time.perf_counter()
        
        # Parse output (YOLO format: [batch, num_boxes, 5 + num_classes])
        # Format: [x_center, y_center, width, height, confidence, class_scores...]
        
        detections = []
        
        # Handle different output formats
        if len(output.shape) == 3:
            # Shape could be [batch, num_boxes, attributes] OR [batch, attributes, num_predictions]
            # YOLO v8 format: [batch, 4+num_classes, num_predictions]
            # e.g., (1, 84, 8400) where 84 = 4 (bbox) + 80 (classes)
            
            # Check if it's YOLO format (channels-first)
            if output.shape[1] <= 100:  # Likely [batch, attrs, preds] format
                # Transpose to [batch, num_predictions, attrs]
                output = output.transpose(0, 2, 1)  # (1, 8400, 84)
            
            boxes = output[0]  # Remove batch dimension -> (8400, 84)
        else:
            boxes = output
        
        # DEBUG: Print output info
        print(f"\n[DEBUG] ========== POSTPROCESS START ==========")
        print(f"[DEBUG] Original output shape: {output.shape}")
        print(f"[DEBUG] Boxes shape after transpose: {boxes.shape}")
        print(f"[DEBUG] Original image shape: {orig_shape}, inference size: {self.img_size}x{self.img_size}")
        if boxes.shape[0] > 0:
            print(f"[DEBUG] First box raw: {boxes[0]}")
        
        # Extract predictions
        # Format: [x_center, y_center, width, height, conf_score, class_1, class_2, ..., class_80]
        box_coords = boxes[:, :4]  # [x_center, y_center, width, height]
        box_conf = boxes[:, 4]      # Objectness confidence
        class_scores = boxes[:, 5:] # Class probabilities [num_preds, 80]
        
        # DEBUG: Print raw scores BEFORE sigmoid
        print(f"[DEBUG] RAW box_conf - min={np.min(box_conf):.8f}, max={np.max(box_conf):.8f}, mean={np.mean(box_conf):.8f}")
        print(f"[DEBUG] RAW first 10 box_conf values: {box_conf[:10]}")
        print(f"[DEBUG] RAW class_scores - min={np.min(class_scores):.8f}, max={np.max(class_scores):.8f}, mean={np.mean(class_scores):.8f}")
        
        # Apply sigmoid if values are too small (raw model output)
        if np.max(box_conf) < 1.0 and np.max(box_conf) > 1e-5:
            # Values look like they need sigmoid
            print(f"[DEBUG] APPLYING SIGMOID (raw values detected)")
            box_conf_before = box_conf.copy()
            box_conf = 1.0 / (1.0 + np.exp(-box_conf))
            class_scores = 1.0 / (1.0 + np.exp(-class_scores))
            print(f"[DEBUG] After sigmoid - box_conf: min={np.min(box_conf):.8f}, max={np.max(box_conf):.8f}, mean={np.mean(box_conf):.8f}")
            print(f"[DEBUG] Before sigmoid (first 10): {box_conf_before[:10]}")
            print(f"[DEBUG] After sigmoid (first 10): {box_conf[:10]}")
        else:
            print(f"[DEBUG] SIGMOID NOT APPLIED (max={np.max(box_conf):.8f}, min={np.min(box_conf):.8f})")
        
        # Apply confidence threshold on objectness score
        confidences = box_conf
        print(f"[DEBUG] Confidence range: min={np.min(confidences):.6f}, max={np.max(confidences):.6f}, mean={np.mean(confidences):.6f}")
        print(f"[DEBUG] Threshold: {self.conf_threshold}")
        
        # Use much lower threshold for debugging (0.001 = 0.1%)
        DEBUG_THRESHOLD = 0.001
        mask = confidences > DEBUG_THRESHOLD
        print(f"[DEBUG] Boxes passing threshold (0.1%): {np.sum(mask)} / {len(confidences)}")
        
        boxes_filtered = boxes[mask]
        box_coords_filtered = box_coords[mask]
        box_conf_filtered = confidences[mask]
        class_scores_filtered = class_scores[mask]
        
        if len(boxes_filtered) == 0:
            print(f"[DEBUG] No detections after threshold\n")
            self.postprocess_times.append((time.perf_counter() - t0) * 1000)
            return []
        
        print(f"[DEBUG] Total detections: {len(boxes_filtered)}")
        print(f"[DEBUG] Box conf range: min={np.min(box_conf_filtered):.6f}, max={np.max(box_conf_filtered):.6f}")
        print(f"[DEBUG] Class scores range: min={np.min(class_scores_filtered):.6f}, max={np.max(class_scores_filtered):.6f}")
        
        print(f"[DEBUG] Detections after threshold: {len(boxes_filtered)}")
        
        # Get class IDs and max class confidence
        class_ids = np.argmax(class_scores_filtered, axis=1)
        class_confidences = np.max(class_scores_filtered, axis=1)
        
        print(f"[DEBUG] Class scores shape: {class_scores_filtered.shape}")
        print(f"[DEBUG] First 5 class_ids: {class_ids[:5]}")
        print(f"[DEBUG] First 5 class_confs: {class_confidences[:5]}")
        print(f"[DEBUG] First 5 box_confs: {box_conf_filtered[:5]}")
        
        # Normalize class confidences to [0, 1]
        class_confidences = np.clip(class_confidences, 0.0, 1.0)
        box_conf_filtered = np.clip(box_conf_filtered, 0.0, 1.0)
        
        # Use max of objectness and class confidence as final score
        final_confidences = np.maximum(box_conf_filtered, class_confidences)
        
        print(f"[DEBUG] Final confidences range: min={np.min(final_confidences):.6f}, max={np.max(final_confidences):.6f}")
        x_center, y_center = box_coords_filtered[:, 0], box_coords_filtered[:, 1]
        width, height = box_coords_filtered[:, 2], box_coords_filtered[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale to original image size
        scale_x = orig_shape[1] / self.img_size
        scale_y = orig_shape[0] / self.img_size
        
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        
        # Clip to original image bounds
        x1 = np.maximum(0, x1)
        y1 = np.maximum(0, y1)
        x2 = np.minimum(orig_shape[1], x2)
        y2 = np.minimum(orig_shape[0], y2)
        
        print(f"[DEBUG] First bbox after scaling: ({x1[0]:.1f}, {y1[0]:.1f}, {x2[0]:.1f}, {y2[0]:.1f})")
        
        # Apply NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        keep_indices = self.nms(boxes_for_nms, final_confidences, self.iou_threshold)
        
        print(f"[DEBUG] Final detections after NMS: {len(keep_indices)}")
        
        # Create Detection objects
        for idx in keep_indices:
            class_id = int(class_ids[idx])
            # Validate class_id
            if class_id < 0 or class_id >= len(self.class_names):
                print(f"[WARNING] Invalid class_id {class_id}, total_classes={len(self.class_names)}")
                class_name = f"unknown_{class_id}"
            else:
                class_name = self.class_names[class_id]
            
            detection = Detection(
                bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                confidence=float(final_confidences[idx]),
                class_id=class_id,
                class_name=class_name
            )
            detections.append(detection)
            print(f"[DEBUG] → {class_name} (id={class_id}), conf={final_confidences[idx]:.4f}, bbox=({x1[idx]:.1f},{y1[idx]:.1f},{x2[idx]:.1f},{y2[idx]:.1f})")
        
        print(f"[DEBUG] ========== POSTPROCESS END ==========\n")
        self.postprocess_times.append((time.perf_counter() - t0) * 1000)
        
        return detections
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Custom Non-Maximum Suppression.
        
        Args:
            boxes: Array of boxes [N, 4] in format (x1, y1, x2, y2)
            scores: Array of scores [N]
            iou_threshold: IoU threshold
        
        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on an image.
        
        Args:
            image: Input image (BGR, HWC)
        
        Returns:
            List of Detection objects
        """
        orig_shape = image.shape[:2]
        output = self._infer(image)
        detections = self.postprocess(output, orig_shape)
        return detections
    
    def get_timing_stats(self) -> dict:
        """Get timing statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'preprocess_avg_ms': np.mean(self.preprocess_times),
            'inference_avg_ms': np.mean(self.inference_times),
            'postprocess_avg_ms': np.mean(self.postprocess_times),
            'total_avg_ms': np.mean(self.preprocess_times) + np.mean(self.inference_times) + np.mean(self.postprocess_times),
            'fps': 1000.0 / (np.mean(self.preprocess_times) + np.mean(self.inference_times) + np.mean(self.postprocess_times))
        }
    
    def reset_timing_stats(self):
        """Reset timing statistics."""
        self.inference_times.clear()
        self.preprocess_times.clear()
        self.postprocess_times.clear()


def visualize_detections(
    image: np.ndarray,
    detections: List[Detection],
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Visualize detections on image.
    
    Args:
        image: Input image
        detections: List of Detection objects
        thickness: Box thickness
        font_scale: Font scale
    
    Returns:
        Image with visualizations
    """
    img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        
        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    return img
