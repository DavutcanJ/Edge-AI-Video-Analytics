"""
FastAPI server for object detection inference.
Provides REST API endpoints with performance monitoring.
Supports multi-backend (PyTorch, ONNX, TensorRT) and real-time video streaming.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

ws_manager = ConnectionManager()


def get_model_path(backend: str) -> str:
    """Get model path for the specified backend."""
    paths = {
        "tensorrt": "models/latest.fp16.engine",
        "onnx": "models/latest.onnx", 
        "pytorch": "models/latest.pt"
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
            img_size=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
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
    
    import os
    active_backend = os.getenv("BACKEND", "tensorrt")
    
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
            "backends": "/backends",
            "health": "/health",
            "metrics": "/metrics"
        }
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
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        image_shape = image.shape[:2]
        
        # Run detection
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
        
        # Get timing breakdown
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
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
    backend: Optional[str] = None
):
    """
    WebSocket endpoint for real-time video stream processing.
    
    Client sends base64 encoded frames, server returns detections.
    
    Message format (client -> server):
    {
        "frame": "<base64_encoded_jpeg>",
        "timestamp": 1234567890.123
    }
    
    Message format (server -> client):
    {
        "detections": [...],
        "inference_time_ms": 12.34,
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
    
    # FPS control
    min_frame_time = 1.0 / fps_limit if fps_limit > 0 else 0
    last_frame_time = 0
    fps_counter = FPSMeter()
    
    try:
        await websocket.send_json({
            "status": "connected",
            "backend": backend_str,
            "fps_limit": fps_limit
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
                frame_data = base64.b64decode(data["frame"])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid frame data"})
                    continue
                
                # Run detection
                start_time = time.perf_counter()
                detections = detector(frame)
                inference_time = (time.perf_counter() - start_time) * 1000
                
                # Update FPS
                fps_counter.tick()
                
                # Format response
                detection_list = []
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
                    "num_detections": len(detections),
                    "inference_time_ms": round(inference_time, 2),
                    "fps": round(fps_counter.get_fps(), 1),
                    "timestamp": data.get("timestamp", time.time()),
                    "frame_shape": list(frame.shape[:2])
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
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
        frame_data = base64.b64decode(frame)
        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Run detection
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
        return {
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
        
    except Exception as e:
        logger.error(f"Frame processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
