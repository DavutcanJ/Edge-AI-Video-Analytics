"""
FastAPI server for object detection inference.
Provides REST API endpoints with performance monitoring.
Supports multi-backend (PyTorch, ONNX, TensorRT) and real-time video streaming.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# =============================================================================
# Sentry Integration
# =============================================================================
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

# Check if Sentry metrics should be enabled (disabled by default for memory)
SENTRY_METRICS_ENABLED = os.environ.get("SENTRY_METRICS_ENABLED", "false").lower() == "true"

if SENTRY_METRICS_ENABLED:
    try:
        # Available in sentry-sdk >= 2.x
        from sentry_sdk import metrics as sentry_metrics
    except Exception:  # pragma: no cover
        SENTRY_METRICS_ENABLED = False
        sentry_metrics = None
else:
    sentry_metrics = None

# No-op metrics wrapper (zero overhead when disabled)
class _SentryMetrics:
    """Lightweight metrics wrapper - only sends if enabled."""
    @staticmethod
    def incr(*args, **kwargs):
        if sentry_metrics:
            sentry_metrics.incr(*args, **kwargs)
    @staticmethod
    def gauge(*args, **kwargs):
        if sentry_metrics:
            sentry_metrics.gauge(*args, **kwargs)
    @staticmethod
    def distribution(*args, **kwargs):
        if sentry_metrics:
            sentry_metrics.distribution(*args, **kwargs)

metrics = _SentryMetrics()


# =============================================================================
# SENTRY EVENT FILTERS (Memory Optimization)
# =============================================================================
def _sentry_before_send(event, hint):
    """
    Filter events before sending to reduce memory and noise.
    Return None to drop the event entirely.
    """
    # Drop connection reset errors (usually client disconnects)
    if "exception" in event:
        exc_type = event.get("exception", {}).get("values", [{}])[0].get("type", "")
        if exc_type in ("ConnectionResetError", "BrokenPipeError", "ConnectionAbortedError"):
            return None
    
    # Drop 4xx client errors (not our problem)
    if event.get("level") == "error":
        status_code = event.get("extra", {}).get("status_code", 500)
        if 400 <= status_code < 500:
            return None
    
    return event


def _sentry_before_send_transaction(event, hint):
    """
    Filter transactions to reduce volume.
    Drop health checks and high-frequency low-value endpoints.
    """
    transaction = event.get("transaction", "")
    
    # Skip health checks - too frequent, low value
    if transaction in ("/health", "/", "/sentry/status", "/metrics"):
        return None
    
    return event


# Initialize Sentry (set SENTRY_DSN environment variable)
# =============================================================================
# MEMORY OPTIMIZATION SETTINGS
# Set these env vars to tune Sentry memory footprint:
#   SENTRY_TRACES_SAMPLE_RATE=0.01   (1% of requests, default 0.05)
#   SENTRY_PROFILES_ENABLED=false    (disable profiling entirely)
#   SENTRY_MAX_BREADCRUMBS=20        (default 100)
# =============================================================================
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")

# Configurable sample rates (lower = less memory)
SENTRY_TRACES_RATE = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.05"))  # 5% default
SENTRY_PROFILES_ENABLED = os.environ.get("SENTRY_PROFILES_ENABLED", "false").lower() == "true"
SENTRY_MAX_BREADCRUMBS = int(os.environ.get("SENTRY_MAX_BREADCRUMBS", "20"))  # Reduced from 100

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        
        # ===== MEMORY OPTIMIZATIONS =====
        # 1. Lower trace sampling (fewer transactions stored in memory)
        traces_sample_rate=SENTRY_TRACES_RATE,
        
        # 2. Disable profiling by default (significant RAM saver)
        #    Profiling keeps call stacks in memory - very expensive
        profiles_sample_rate=0.05 if SENTRY_PROFILES_ENABLED else 0.0,
        enable_tracing=SENTRY_TRACES_RATE > 0,
        
        # 3. Reduce breadcrumb buffer (each breadcrumb consumes memory)
        max_breadcrumbs=SENTRY_MAX_BREADCRUMBS,
        
        # 4. Disable auto session tracking (reduces background memory)
        auto_session_tracking=False,
        
        # 5. Limit stack trace locals (can be large objects)
        include_local_variables=False,
        
        # 6. Disable source context fetching (reads files into memory)
        include_source_context=False,
        
        # Environment tag
        environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
        # Release tag (use git commit or version)
        release=os.environ.get("SENTRY_RELEASE", "cv-advanced-api@1.0.0"),
        
        # 7. Minimal integrations (each integration adds overhead)
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        # Keep default integrations for error capturing
        default_integrations=True,
        
        # Debug mode - console'da ne olduğunu gör
        debug=True,
        
        # Don't send PII
        send_default_pii=False,
        
        # 8. Event filtering - drop low-value events before they consume memory
        before_send=_sentry_before_send,
        before_send_transaction=_sentry_before_send_transaction,
    )
    # Set some static tags
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("service", "cv-advanced-api")
        scope.set_tag("component", "inference-api")
    print(f"[INFO] Sentry initialized (traces={SENTRY_TRACES_RATE}, profiles={'on' if SENTRY_PROFILES_ENABLED else 'off'}, breadcrumbs={SENTRY_MAX_BREADCRUMBS})")
else:
    print("[INFO] Sentry DSN not configured. Set SENTRY_DSN environment variable to enable.")

import cv2
import numpy as np
import time
import torch
import asyncio
import base64
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
import io
from PIL import Image
import logging
from dataclasses import dataclass
from enum import Enum

from inference.detector import Detector
from inference.tracker import ByteTracker, Track
from inference.fusion import DetectionTrackerFusion
from inference.video_engine import VideoEngine
from api.schemas import (
    DetectionResponse,
    DetectionResult,
    DetectionBox,
    HealthResponse,
    MetricsResponse,
    ErrorResponse
)
from monitoring.logger import MetricsLogger
from monitoring.fps_meter import FPSMeter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class BackendType(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


# Global detector instances for each backend
detectors: Dict[str, Detector] = {}
active_backend: str = "tensorrt"
metrics_logger: Optional[MetricsLogger] = None
fps_meter: Optional[FPSMeter] = None
total_requests: int = 0


# WebSocket connections manager with tracking support
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Per-connection trackers and fusion modules
        self.trackers: Dict[WebSocket, ByteTracker] = {}
        self.fusion_modules: Dict[WebSocket, DetectionTrackerFusion] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Initialize tracker and fusion for this connection
        self.trackers[websocket] = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        self.fusion_modules[websocket] = DetectionTrackerFusion(
            iou_threshold=0.5,
            confidence_boost=0.1,
            max_disappeared=30
        )
        logger.info(f"WebSocket connected with tracking. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Clean up tracker and fusion
        if websocket in self.trackers:
            del self.trackers[websocket]
        if websocket in self.fusion_modules:
            del self.fusion_modules[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

ws_manager = ConnectionManager()


# =============================================================================
# Environment Configuration
# =============================================================================
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
DEFAULT_BACKEND = os.environ.get("DEFAULT_BACKEND", "tensorrt")

# Model paths from environment
PYTORCH_MODEL_PATH = os.environ.get("PYTORCH_MODEL_PATH", "models/latest.pt")
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "models/latest.onnx")
TENSORRT_MODEL_PATH = os.environ.get("TENSORRT_MODEL_PATH", "models/latest.fp16.engine")

# Detection settings
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "640"))


def get_model_path(backend: str) -> str:
    """Get model path for the specified backend."""
    paths = {
        "tensorrt": TENSORRT_MODEL_PATH, 
        "onnx": ONNX_MODEL_PATH, 
        "pytorch": PYTORCH_MODEL_PATH
    }
    return paths.get(backend, paths["tensorrt"])


def load_detector(backend: str, force_reload: bool = False) -> Optional[Detector]:
    """Load or get cached detector for the specified backend."""
    global detectors
    
    if backend in detectors and not force_reload:
        return detectors[backend]
    
    model_path = get_model_path(backend)
    
    if not Path(model_path).exists():
        logger.warning(f"Model not found for {backend}: {model_path}")
        return None
    
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        detector = Detector(
            model_path=model_path,
            backend=backend,
            device=device,
            img_size=INPUT_SIZE,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            class_names=COCO_CLASSES,
            warmup_iterations=5
        )
        detectors[backend] = detector
        logger.info(f"Loaded {backend} detector from {model_path}")
        return detector
    except Exception as e:
        logger.error(f"Failed to load {backend} detector: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global detectors, active_backend, metrics_logger, fps_meter
    
    logger.info("Starting up server...")
    
    active_backend = DEFAULT_BACKEND
    
    # Load default backend
    detector = load_detector(active_backend)
    if detector is None:
        # Fallback to other backends
        for fallback in ["onnx", "pytorch"]:
            detector = load_detector(fallback)
            if detector:
                active_backend = fallback
                break
    
    if not detectors:
        logger.warning("No detectors loaded. Server will start but detection endpoints will fail.")
    else:
        logger.info(f"Active backend: {active_backend}")
    
    metrics_logger = MetricsLogger()
    fps_meter = FPSMeter()
    
    logger.info("Server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    if metrics_logger:
        metrics_logger.save_metrics("api_metrics.json")
    logger.info("Shutdown complete")


app = FastAPI(
    title="Edge AI Video Analytics API",
    description="High-performance object detection API with multi-backend support and real-time streaming",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Edge AI Video Analytics API",
        "version": "2.0.0",
        "active_backend": active_backend,
        "available_backends": list(detectors.keys()),
        "endpoints": {
            "detect": "/detect",
            "detect_visualize": "/detect/visualize",
            "stream": "/stream (WebSocket)",
            "video_process": "/video/process",
            "video_camera": "/video/camera",
            "test_camera": "/test/camera",
            "test_video": "/test/video",
            "test_api": "/test/api",
            "backends": "/backends",
            "health": "/health",
            "metrics": "/metrics",
            "sentry_debug": "/sentry-debug"
        }
    }


# =============================================================================
# Sentry Debug & Monitoring Endpoints
# =============================================================================

@app.get("/sentry-debug")
async def trigger_sentry_error():
    """
    Debug endpoint to test Sentry integration.
    Intentionally raises an error that will be captured by Sentry.
    """
    if not SENTRY_DSN:
        return {
            "status": "disabled",
            "message": "Sentry is not configured. Set SENTRY_DSN environment variable."
        }
    
    # Capture a test message
    sentry_sdk.capture_message("Sentry test message from CV API", level="info")
    
    # Optionally trigger an error (uncomment to test error capture)
    division_by_zero = 1 / 0
    
    return {
        "status": "success",
        "message": "Test message sent to Sentry",
        "dsn_configured": True
    }


@app.get("/sentry/status")
async def sentry_status():
    """Get Sentry integration status and memory optimization settings."""
    return {
        "enabled": bool(SENTRY_DSN),
        "dsn_configured": bool(SENTRY_DSN),
        "environment": os.environ.get("SENTRY_ENVIRONMENT", "development"),
        "release": os.environ.get("SENTRY_RELEASE", "cv-advanced-api@1.0.0"),
        # Memory optimization settings
        "memory_optimizations": {
            "traces_sample_rate": SENTRY_TRACES_RATE if SENTRY_DSN else 0,
            "profiles_enabled": SENTRY_PROFILES_ENABLED,
            "metrics_enabled": SENTRY_METRICS_ENABLED,
            "max_breadcrumbs": SENTRY_MAX_BREADCRUMBS,
            "auto_session_tracking": False,
            "include_local_variables": False,
            "include_source_context": False,
        },
        "tips": [
            "Set SENTRY_TRACES_SAMPLE_RATE=0.01 for minimal tracing",
            "Set SENTRY_PROFILES_ENABLED=false to disable profiling (default)",
            "Set SENTRY_METRICS_ENABLED=false to disable metrics (default)",
            "Set SENTRY_MAX_BREADCRUMBS=10 for minimal breadcrumb buffer",
        ]
    }


@app.get("/backends")
async def list_backends():
    """List available backends and their status."""
    backends_status = {}
    
    for backend in ["pytorch", "onnx", "tensorrt"]:
        model_path = get_model_path(backend)
        model_exists = Path(model_path).exists()
        loaded = backend in detectors
        
        backends_status[backend] = {
            "model_path": model_path,
            "model_exists": model_exists,
            "loaded": loaded,
            "active": backend == active_backend
        }
    
    return {
        "active_backend": active_backend,
        "backends": backends_status
    }


@app.post("/backends/{backend}")
async def switch_backend(backend: BackendType):
    """Switch to a different backend."""
    global active_backend
    
    backend_str = backend.value
    
    # Load if not already loaded
    detector = load_detector(backend_str)
    
    if detector is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Backend {backend_str} not available. Model file missing or load failed."
        )
    
    active_backend = backend_str
    logger.info(f"Switched to backend: {active_backend}")
    
    return {
        "message": f"Switched to {backend_str} backend",
        "active_backend": active_backend
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    detector = detectors.get(active_backend)
    
    return HealthResponse(
        status="healthy" if detector is not None else "degraded",
        backend=active_backend,
        model_loaded=detector is not None,
        gpu_available=gpu_available
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics."""
    global total_requests, metrics_logger, fps_meter
    
    if not metrics_logger:
        raise HTTPException(status_code=503, detail="Metrics logger not initialized")
    
    stats = metrics_logger.get_stats()
    
    # Get GPU metrics
    gpu_util = None
    gpu_mem_used = None
    gpu_mem_total = None
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        gpu_util = float(util.gpu)
        gpu_mem_used = float(mem_info.used / (1024 ** 2))
        gpu_mem_total = float(mem_info.total / (1024 ** 2))
        
        pynvml.nvmlShutdown()
    except:
        pass
    
    return MetricsResponse(
        total_requests=total_requests,
        avg_latency_ms=stats.get('avg_latency_ms', 0.0),
        p50_latency_ms=stats.get('p50_latency_ms', 0.0),
        p95_latency_ms=stats.get('p95_latency_ms', 0.0),
        p99_latency_ms=stats.get('p99_latency_ms', 0.0),
        throughput_fps=fps_meter.get_fps() if fps_meter else 0.0,
        gpu_utilization_pct=gpu_util,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    backend: Optional[BackendType] = None,
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    background_tasks: BackgroundTasks = None
):
    """
    Detect objects in uploaded image.
    
    Args:
        file: Image file to process
        backend: Optional backend to use (default: active backend)
        conf_threshold: Optional confidence threshold override
    
    Returns:
        Detection results with timing information
    """
    global total_requests, metrics_logger, fps_meter
    
    # Get detector for specified or active backend
    backend_str = backend.value if backend else active_backend
    detector = detectors.get(backend_str)
    
    # Load on demand if not loaded
    if detector is None:
        detector = load_detector(backend_str)
    
    if detector is None:
        raise HTTPException(status_code=503, detail=f"Detector not available for backend: {backend_str}")
    
    # Apply custom threshold if provided
    original_threshold = detector.conf_threshold
    if conf_threshold is not None:
        detector.conf_threshold = conf_threshold
    
    try:
        # Annotate Sentry scope with request-specific tags
        if SENTRY_DSN:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("endpoint", "/detect")
                scope.set_tag("backend", backend_str)
        # Read image
        with sentry_sdk.start_span(op="preprocess", description="read+decode image"):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        image_shape = image.shape[:2]
        
        # Run detection
        with sentry_sdk.start_span(op="inference", description=f"{backend_str} forward"):
            start_time = time.perf_counter()
            detections = detector(image)
            end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Update metrics
        total_requests += 1
        if metrics_logger:
            metrics_logger.log_latency(inference_time_ms)
        if fps_meter:
            fps_meter.tick()
        # Push lightweight metrics to Sentry (no-op if disabled)
        metrics.incr("api.requests", tags={"endpoint": "detect", "backend": backend_str})
        metrics.distribution("inference.latency_ms", inference_time_ms, tags={"backend": backend_str})
        if fps_meter:
            metrics.gauge("inference.fps", fps_meter.get_fps(), tags={"backend": backend_str})
        
        # Get timing breakdown
        with sentry_sdk.start_span(op="postprocess", description="format response"):
            timing_stats = detector.get_timing_stats()
        
        # Convert detections to response format
        detection_results = []
        for det in detections:
            detection_results.append(
                DetectionResult(
                    bbox=DetectionBox(
                        x1=det.bbox[0],
                        y1=det.bbox[1],
                        x2=det.bbox[2],
                        y2=det.bbox[3]
                    ),
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name
                )
            )
        
        response = DetectionResponse(
            detections=detection_results,
            inference_time_ms=inference_time_ms,
            preprocess_time_ms=timing_stats.get('preprocess_avg_ms'),
            postprocess_time_ms=timing_stats.get('postprocess_avg_ms'),
            image_shape=image_shape,
            num_detections=len(detections)
        )
        
        logger.info(f"[{backend_str}] {len(detections)} detections in {inference_time_ms:.2f}ms")
        
        return response
        
    except HTTPException as http_exc:
        # Capture 5xx as errors in Sentry; ignore 4xx noise
        try:
            if SENTRY_DSN and 500 <= getattr(http_exc, "status_code", 500) < 600:
                sentry_sdk.capture_exception(http_exc)
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        try:
            if SENTRY_DSN:
                sentry_sdk.capture_exception(e)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Restore original threshold
        if conf_threshold is not None:
            detector.conf_threshold = original_threshold


@app.post("/detect/visualize")
async def detect_and_visualize(
    file: UploadFile = File(...),
    backend: Optional[BackendType] = None
):
    """
    Detect objects and return visualized image.
    
    Args:
        file: Image file to process
        backend: Optional backend to use
    
    Returns:
        Image with bounding boxes drawn
    """
    backend_str = backend.value if backend else active_backend
    detector = detectors.get(backend_str) or load_detector(backend_str)
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        detections = detector(image)
        
        # Visualize
        from inference.detector import visualize_detections
        vis_image = visualize_detections(image, detections)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', vis_image)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(io_buf, media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    fps_limit: int = 30,
    backend: Optional[str] = None,
    enable_tracking: bool = True
):
    """
    WebSocket endpoint for real-time video stream processing with tracking.
    
    Client sends base64 encoded frames, server returns detections with track IDs.
    
    Message format (client -> server):
    {
        "frame": "<base64_encoded_jpeg>",
        "timestamp": 1234567890.123
    }
    
    Message format (server -> client):
    {
        "detections": [...],  # With track_id if tracking enabled
        "tracks": [...],      # Active tracks (if tracking enabled)
        "inference_time_ms": 12.34,
        "tracking_time_ms": 2.5,
        "fps": 30.0,
        "timestamp": 1234567890.123
    }
    """
    await ws_manager.connect(websocket)
    
    # Get detector
    backend_str = backend or active_backend
    detector = detectors.get(backend_str) or load_detector(backend_str)
    
    if detector is None:
        await websocket.send_json({"error": f"Detector not available for backend: {backend_str}"})
        await websocket.close()
        return
    
    # Get tracker and fusion for this connection
    tracker = ws_manager.trackers.get(websocket) if enable_tracking else None
    fusion = ws_manager.fusion_modules.get(websocket) if enable_tracking else None
    
    # FPS control
    min_frame_time = 1.0 / fps_limit if fps_limit > 0 else 0
    last_frame_time = 0
    fps_counter = FPSMeter()
    
    try:
        await websocket.send_json({
            "status": "connected",
            "backend": backend_str,
            "fps_limit": fps_limit,
            "tracking_enabled": enable_tracking
        })
        
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            
            current_time = time.perf_counter()
            
            # FPS limiting
            elapsed = current_time - last_frame_time
            if elapsed < min_frame_time:
                # Skip frame to maintain FPS limit
                continue
            
            last_frame_time = current_time
            
            # Decode frame
            if "frame" not in data:
                await websocket.send_json({"error": "Missing 'frame' field"})
                continue
            
            try:
                # Decode base64 image
                with sentry_sdk.start_span(op="preprocess", description="ws decode image"):
                    frame_data = base64.b64decode(data["frame"])
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid frame data"})
                    continue
                
                # Run detection
                with sentry_sdk.start_span(op="inference", description=f"{backend_str} forward (ws)"):
                    start_time = time.perf_counter()
                    detections = detector(frame)
                    inference_time = (time.perf_counter() - start_time) * 1000
                
                # Run tracking if enabled
                tracking_time = 0
                tracks = []
                if enable_tracking and tracker and fusion:
                    with sentry_sdk.start_span(op="tracking", description="update tracks"):
                        track_start = time.perf_counter()
                        
                        # Update tracker with detections
                        tracks = tracker.update(detections)
                        
                        # Fuse tracker and detector outputs
                        fused_tracks = fusion.fuse(detections, tracks)
                        
                        tracking_time = (time.perf_counter() - track_start) * 1000
                        tracks = fused_tracks
                
                # Update FPS
                fps_counter.tick()
                
                # Format response
                detection_list = []
                if enable_tracking and tracks:
                    # Return tracks with IDs
                    for track in tracks:
                        detection_list.append({
                            "track_id": track.track_id,
                            "bbox": {
                                "x1": track.bbox[0],
                                "y1": track.bbox[1],
                                "x2": track.bbox[2],
                                "y3": track.bbox[3]
                            },
                            "confidence": track.confidence,
                            "class_id": track.class_id,
                            "state": track.state,
                            "age": track.age,
                            "velocity": {
                                "vx": track.velocity[0],
                                "vy": track.velocity[1]
                            }
                        })
                else:
                    # Return detections without track IDs
                    for det in detections:
                        detection_list.append({
                            "bbox": {
                                "x1": det.bbox[0],
                                "y1": det.bbox[1],
                                "x2": det.bbox[2],
                                "y2": det.bbox[3]
                            },
                            "confidence": det.confidence,
                            "class_id": det.class_id,
                            "class_name": det.class_name
                        })
                
                response = {
                    "detections": detection_list,
                    "num_detections": len(detection_list),
                    "num_tracks": len(tracks) if enable_tracking else 0,
                    "inference_time_ms": round(inference_time, 2),
                    "tracking_time_ms": round(tracking_time, 2) if enable_tracking else 0,
                    "total_time_ms": round(inference_time + tracking_time, 2),
                    "fps": round(fps_counter.get_fps(), 1),
                    "timestamp": data.get("timestamp", time.time()),
                    "frame_shape": list(frame.shape[:2]),
                    "tracking_enabled": enable_tracking
                }
                
                await websocket.send_json(response)
                # Metrics (no-op if disabled)
                metrics.incr("ws.frames_processed", tags={"backend": backend_str, "tracking": str(enable_tracking)})
                metrics.distribution("ws.inference.latency_ms", inference_time, tags={"backend": backend_str})
                if enable_tracking:
                    metrics.distribution("ws.tracking.latency_ms", tracking_time, tags={"backend": backend_str})
                
            except Exception as e:
                try:
                    if SENTRY_DSN:
                        sentry_sdk.capture_exception(e)
                except Exception:
                    pass
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            if SENTRY_DSN:
                sentry_sdk.capture_exception(e)
        except Exception:
            pass
        ws_manager.disconnect(websocket)


@app.post("/stream/frame")
async def process_single_frame(
    frame: str,  # Base64 encoded frame
    backend: Optional[BackendType] = None,
    return_image: bool = False
):
    """
    Process a single frame sent as base64.
    Alternative to WebSocket for simpler clients.
    
    Args:
        frame: Base64 encoded JPEG image
        backend: Optional backend to use
        return_image: If True, return visualized image instead of JSON
    
    Returns:
        Detection results or visualized image
    """
    global total_requests, fps_meter
    
    backend_str = backend.value if backend else active_backend
    detector = detectors.get(backend_str) or load_detector(backend_str)
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    try:
        # Decode base64 frame
        with sentry_sdk.start_span(op="preprocess", description="decode single frame"):
            frame_data = base64.b64decode(frame)
            nparr = np.frombuffer(frame_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Run detection
        with sentry_sdk.start_span(op="inference", description=f"{backend_str} forward (single)"):
            start_time = time.perf_counter()
            detections = detector(image)
            inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        total_requests += 1
        if fps_meter:
            fps_meter.tick()
        
        if return_image:
            # Return visualized image
            from inference.detector import visualize_detections
            vis_image = visualize_detections(image, detections)
            _, buffer = cv2.imencode('.jpg', vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            return StreamingResponse(
                io.BytesIO(buffer),
                media_type="image/jpeg"
            )
        
        # Return JSON
        result = {
            "detections": [
                {
                    "bbox": {"x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3]},
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name
                }
                for d in detections
            ],
            "num_detections": len(detections),
            "inference_time_ms": round(inference_time_ms, 2),
            "fps": round(fps_meter.get_fps(), 1) if fps_meter else 0
        }
        metrics.incr("api.requests", tags={"endpoint": "stream/frame", "backend": backend_str})
        metrics.distribution("inference.latency_ms", inference_time_ms, tags={"backend": backend_str, "mode": "single"})
        return result
        
    except Exception as e:
        logger.error(f"Frame processing failed: {e}")
        try:
            if SENTRY_DSN:
                sentry_sdk.capture_exception(e)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Training Logs & Dashboard Endpoints
# =============================================================================

@app.get("/training/experiments")
async def list_training_experiments():
    """
    List all training experiments with their basic info.
    """
    logs_dir = Path("training/logs")
    
    if not logs_dir.exists():
        return {"experiments": [], "message": "No training logs found"}
    
    experiments = []
    for exp_dir in sorted(logs_dir.iterdir(), reverse=True):
        if exp_dir.is_dir() and exp_dir.name.startswith("exp_"):
            exp_info = {
                "name": exp_dir.name,
                "path": str(exp_dir),
                "has_results": (exp_dir / "results.csv").exists(),
                "has_weights": (exp_dir / "weights").exists()
            }
            
            # Parse date from experiment name
            try:
                date_str = exp_dir.name.replace("exp_", "")
                exp_info["created"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {date_str[9:11]}:{date_str[11:13]}:{date_str[13:15]}"
            except:
                exp_info["created"] = None
            
            experiments.append(exp_info)
    
    return {"experiments": experiments, "total": len(experiments)}


@app.get("/training/experiments/{exp_name}/results")
async def get_training_results(exp_name: str):
    """
    Get training results CSV data for a specific experiment.
    """
    import csv
    
    results_path = Path(f"training/logs/{exp_name}/results.csv")
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for experiment: {exp_name}")
    
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Convert to proper types
    for row in rows:
        for key in row:
            try:
                row[key] = float(row[key])
            except:
                pass
    
    # Extract key metrics
    if rows:
        last_epoch = rows[-1]
        best_map50 = max(float(r.get('metrics/mAP50(B)', 0)) for r in rows)
        best_map50_95 = max(float(r.get('metrics/mAP50-95(B)', 0)) for r in rows)
        
        summary = {
            "total_epochs": len(rows),
            "final_mAP50": float(last_epoch.get('metrics/mAP50(B)', 0)),
            "final_mAP50_95": float(last_epoch.get('metrics/mAP50-95(B)', 0)),
            "best_mAP50": best_map50,
            "best_mAP50_95": best_map50_95,
            "final_box_loss": float(last_epoch.get('train/box_loss', 0)),
            "final_cls_loss": float(last_epoch.get('train/cls_loss', 0))
        }
    else:
        summary = {}
    
    return {
        "experiment": exp_name,
        "summary": summary,
        "epochs": rows
    }


@app.get("/training/experiments/{exp_name}/charts")
async def get_training_charts(exp_name: str, chart_type: str = Query("all", enum=["all", "loss", "metrics", "mAP"])):
    """
    Get training chart data formatted for frontend visualization.
    """
    import csv
    
    results_path = Path(f"training/logs/{exp_name}/results.csv")
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for experiment: {exp_name}")
    
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    epochs = [int(float(r['epoch'])) for r in rows]
    
    charts = {}
    
    if chart_type in ["all", "loss"]:
        charts["loss"] = {
            "title": "Training Loss",
            "x_label": "Epoch",
            "y_label": "Loss",
            "series": {
                "box_loss": [float(r.get('train/box_loss', 0)) for r in rows],
                "cls_loss": [float(r.get('train/cls_loss', 0)) for r in rows],
                "dfl_loss": [float(r.get('train/dfl_loss', 0)) for r in rows]
            },
            "epochs": epochs
        }
    
    if chart_type in ["all", "metrics"]:
        charts["metrics"] = {
            "title": "Precision & Recall",
            "x_label": "Epoch",
            "y_label": "Score",
            "series": {
                "precision": [float(r.get('metrics/precision(B)', 0)) for r in rows],
                "recall": [float(r.get('metrics/recall(B)', 0)) for r in rows]
            },
            "epochs": epochs
        }
    
    if chart_type in ["all", "mAP"]:
        charts["mAP"] = {
            "title": "Mean Average Precision",
            "x_label": "Epoch",
            "y_label": "mAP",
            "series": {
                "mAP50": [float(r.get('metrics/mAP50(B)', 0)) for r in rows],
                "mAP50-95": [float(r.get('metrics/mAP50-95(B)', 0)) for r in rows]
            },
            "epochs": epochs
        }
    
    return {
        "experiment": exp_name,
        "charts": charts
    }


@app.get("/training/latest")
async def get_latest_training():
    """
    Get the latest training experiment results.
    """
    logs_dir = Path("training/logs")
    
    if not logs_dir.exists():
        raise HTTPException(status_code=404, detail="No training logs found")
    
    # Find latest experiment
    experiments = sorted([d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")], reverse=True)
    
    if not experiments:
        raise HTTPException(status_code=404, detail="No experiments found")
    
    latest = experiments[0]
    
    # Get results
    return await get_training_results(latest.name)


@app.get("/dashboard/inference")
async def get_inference_dashboard():
    """
    Get real-time inference dashboard data.
    """
    global metrics_logger, fps_meter, total_requests, active_backend
    
    if not metrics_logger:
        return {
            "status": "no_data",
            "message": "No inference metrics available yet. Run some detections first."
        }
    
    stats = metrics_logger.get_stats()
    latencies = list(metrics_logger.latencies)
    
    # GPU info
    gpu_info = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        gpu_info = {
            "name": gpu_name,
            "utilization_pct": float(util.gpu),
            "memory_used_mb": float(mem_info.used / (1024 ** 2)),
            "memory_total_mb": float(mem_info.total / (1024 ** 2)),
            "memory_pct": float(mem_info.used / mem_info.total * 100),
            "temperature_c": temp
        }
        
        pynvml.nvmlShutdown()
    except:
        pass
    
    return {
        "status": "active",
        "backend": active_backend,
        "total_requests": total_requests,
        "current_fps": fps_meter.get_fps() if fps_meter else 0,
        "avg_fps": fps_meter.get_average_fps() if fps_meter else 0,
        "latency": {
            "avg_ms": stats.get('avg_latency_ms', 0),
            "min_ms": stats.get('min_latency_ms', 0),
            "max_ms": stats.get('max_latency_ms', 0),
            "p50_ms": stats.get('p50_latency_ms', 0),
            "p95_ms": stats.get('p95_latency_ms', 0),
            "p99_ms": stats.get('p99_latency_ms', 0),
            "recent_values": latencies[-50:]  # Last 50 latency values for chart
        },
        "gpu": gpu_info
    }


@app.get("/dashboard/combined")
async def get_combined_dashboard():
    """
    Get combined training and inference dashboard data.
    """
    inference_data = await get_inference_dashboard()
    
    # Try to get latest training data
    training_data = None
    try:
        training_data = await get_latest_training()
    except:
        pass
    
    return {
        "inference": inference_data,
        "training": training_data
    }


# =============================================================================
# VideoEngine Endpoints
# =============================================================================

@app.post("/video/process")
async def process_video_file(
    file: UploadFile = File(...),
    backend: Optional[BackendType] = None,
    output_format: str = Query("mp4", enum=["mp4", "avi"]),
    visualize: bool = True,
    max_frames: Optional[int] = None
):
    """
    Process uploaded video file with detection and tracking.
    Returns processed video with bounding boxes.
    
    Args:
        file: Video file to process
        backend: Backend to use
        output_format: Output video format
        visualize: Draw bounding boxes
        max_frames: Maximum frames to process
    """
    import tempfile
    import uuid
    
    backend_str = backend.value if backend else active_backend
    detector = detectors.get(backend_str) or load_detector(backend_str)
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    # Save uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}")
    
    try:
        # Write uploaded file
        contents = await file.read()
        temp_input.write(contents)
        temp_input.close()
        
        # Create video engine
        video_engine = VideoEngine(
            detector=detector,
            detection_interval=5,
            use_threading=False
        )
        
        # Process video
        logger.info(f"Processing video: {file.filename}")
        video_engine.process_video(
            video_path=temp_input.name,
            output_path=temp_output.name,
            display=False,
            max_frames=max_frames
        )
        
        # Read processed video
        with open(temp_output.name, 'rb') as f:
            video_data = f.read()
        
        # Cleanup
        os.unlink(temp_input.name)
        os.unlink(temp_output.name)
        
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type=f"video/{output_format}",
            headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
        )
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        try:
            os.unlink(temp_input.name)
            os.unlink(temp_output.name)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/video/camera")
async def process_camera_stream(
    camera_id: int = Query(0, ge=0, le=10),
    backend: Optional[BackendType] = None,
    duration_seconds: int = Query(10, ge=1, le=300),
    fps: int = Query(30, ge=1, le=60)
):
    """
    Process camera stream and return as video.
    
    Args:
        camera_id: Camera device ID (0 for default)
        backend: Backend to use
        duration_seconds: Recording duration
        fps: Output video FPS
    """
    import tempfile
    
    backend_str = backend.value if backend else active_backend
    detector = detectors.get(backend_str) or load_detector(backend_str)
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    try:
        # Create video engine
        video_engine = VideoEngine(
            detector=detector,
            detection_interval=3,
            use_threading=False
        )
        
        # Calculate max frames
        max_frames = duration_seconds * fps
        
        # Process camera
        logger.info(f"Processing camera {camera_id} for {duration_seconds}s")
        video_engine.process_video(
            video_path=camera_id,
            output_path=temp_output.name,
            display=False,
            max_frames=max_frames
        )
        
        # Read processed video
        with open(temp_output.name, 'rb') as f:
            video_data = f.read()
        
        # Cleanup
        os.unlink(temp_output.name)
        
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=camera_{camera_id}_processed.mp4"}
        )
        
    except Exception as e:
        logger.error(f"Camera processing failed: {e}")
        try:
            os.unlink(temp_output.name)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Test Endpoints
# =============================================================================

@app.get("/test/camera")
async def test_camera(camera_id: int = Query(0, ge=0, le=10)):
    """
    Test camera access and return basic info.
    
    Args:
        camera_id: Camera device ID
    """
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return {
                "status": "error",
                "camera_id": camera_id,
                "accessible": False,
                "message": f"Camera {camera_id} not accessible"
            }
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Try to read a frame
        ret, frame = cap.read()
        frame_read = ret and frame is not None
        
        cap.release()
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "accessible": True,
            "properties": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_read_test": frame_read
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "camera_id": camera_id,
            "accessible": False,
            "error": str(e)
        }


@app.post("/test/video")
async def test_video_file(file: UploadFile = File(...)):
    """
    Test video file and return properties.
    
    Args:
        file: Video file to test
    """
    import tempfile
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    
    try:
        # Save uploaded file
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Open video
        cap = cv2.VideoCapture(temp_file.name)
        
        if not cap.isOpened():
            os.unlink(temp_file.name)
            return {
                "status": "error",
                "filename": file.filename,
                "readable": False,
                "message": "Video file not readable"
            }
        
        # Get properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Try to read first frame
        ret, frame = cap.read()
        frame_read = ret and frame is not None
        
        cap.release()
        os.unlink(temp_file.name)
        
        return {
            "status": "success",
            "filename": file.filename,
            "readable": True,
            "properties": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": round(duration, 2),
                "first_frame_read_test": frame_read
            }
        }
        
    except Exception as e:
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return {
            "status": "error",
            "filename": file.filename,
            "readable": False,
            "error": str(e)
        }


@app.get("/test/api")
async def test_api_features():
    """
    Test all API features and return status report.
    """
    report = {
        "timestamp": time.time(),
        "tests": {}
    }
    
    # Test 1: Backend availability
    backends_test = {}
    for backend_name in ["pytorch", "onnx", "tensorrt"]:
        model_path = get_model_path(backend_name)
        model_exists = Path(model_path).exists()
        loaded = backend_name in detectors
        
        # Try loading if not loaded
        if model_exists and not loaded:
            try:
                load_detector(backend_name)
                loaded = backend_name in detectors
            except:
                pass
        
        backends_test[backend_name] = {
            "model_exists": model_exists,
            "loaded": loaded,
            "status": "ok" if (model_exists and loaded) else "fail"
        }
    
    report["tests"]["backends"] = backends_test
    
    # Test 2: GPU availability
    gpu_test = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        try:
            gpu_test["device_name"] = torch.cuda.get_device_name(0)
            gpu_test["status"] = "ok"
        except:
            gpu_test["status"] = "fail"
    else:
        gpu_test["status"] = "warning"
        gpu_test["message"] = "CUDA not available, using CPU"
    
    report["tests"]["gpu"] = gpu_test
    
    # Test 3: Detector functionality
    detector_test = {}
    active_detector = detectors.get(active_backend)
    
    if active_detector:
        try:
            # Create dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Run detection
            start = time.perf_counter()
            detections = active_detector(dummy_image)
            latency_ms = (time.perf_counter() - start) * 1000
            
            detector_test["status"] = "ok"
            detector_test["latency_ms"] = round(latency_ms, 2)
            detector_test["num_detections"] = len(detections)
        except Exception as e:
            detector_test["status"] = "fail"
            detector_test["error"] = str(e)
    else:
        detector_test["status"] = "fail"
        detector_test["message"] = "No active detector"
    
    report["tests"]["detector"] = detector_test
    
    # Test 4: Tracking functionality
    tracking_test = {}
    try:
        tracker = ByteTracker()
        tracking_test["status"] = "ok"
        tracking_test["message"] = "Tracker initialized successfully"
    except Exception as e:
        tracking_test["status"] = "fail"
        tracking_test["error"] = str(e)
    
    report["tests"]["tracking"] = tracking_test
    
    # Test 5: Sentry integration
    sentry_test = {
        "enabled": bool(SENTRY_DSN),
        "status": "ok" if SENTRY_DSN else "disabled"
    }
    
    report["tests"]["sentry"] = sentry_test
    
    # Overall status
    all_ok = all(
        test.get("status") in ["ok", "disabled", "warning"]
        for test in report["tests"].values()
        if isinstance(test, dict)
    )
    
    backend_tests_ok = all(
        b["status"] == "ok"
        for b in backends_test.values()
    )
    
    report["overall_status"] = "healthy" if (all_ok and backend_tests_ok) else "degraded"
    report["active_backend"] = active_backend
    
    return report


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    try:
        if SENTRY_DSN:
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run server with environment configuration
    uvicorn.run(
        "server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        workers=1
    )
