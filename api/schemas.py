"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from enum import Enum


class BackendType(str, Enum):
    """Supported inference backends."""
    pytorch = "pytorch"
    onnx = "onnx"
    tensorrt = "tensorrt"


class DetectionBox(BaseModel):
    """Bounding box for detection."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class DetectionResult(BaseModel):
    """Single detection result."""
    bbox: DetectionBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., ge=0, description="Class ID")
    class_name: str = Field(default="", description="Class name")


class DetectionResponse(BaseModel):
    """Response for detection endpoint."""
    detections: List[DetectionResult]
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")
    preprocess_time_ms: Optional[float] = None
    postprocess_time_ms: Optional[float] = None
    image_shape: Tuple[int, int] = Field(..., description="Input image shape (height, width)")
    num_detections: int = Field(..., description="Number of detections")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Service status")
    backend: str = Field(..., description="Current backend")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")


class MetricsResponse(BaseModel):
    """Response for metrics endpoint."""
    total_requests: int = Field(..., description="Total number of requests processed")
    avg_latency_ms: float = Field(..., description="Average inference latency")
    p50_latency_ms: float = Field(..., description="P50 inference latency")
    p95_latency_ms: float = Field(..., description="P95 inference latency")
    p99_latency_ms: float = Field(..., description="P99 inference latency")
    throughput_fps: float = Field(..., description="Average throughput in FPS")
    gpu_utilization_pct: Optional[float] = Field(None, description="GPU utilization percentage")
    gpu_memory_used_mb: Optional[float] = Field(None, description="GPU memory used in MB")
    gpu_memory_total_mb: Optional[float] = Field(None, description="Total GPU memory in MB")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class ModelConfig(BaseModel):
    """Model configuration."""
    model_path: str = Field(..., description="Path to model file")
    backend: BackendType = Field(BackendType.tensorrt, description="Inference backend")
    device: str = Field("cuda:0", description="Device for inference")
    img_size: int = Field(640, ge=320, le=1280, description="Input image size")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
