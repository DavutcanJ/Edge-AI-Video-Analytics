"""
FastAPI server for object detection inference.
Provides REST API endpoints with performance monitoring.
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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import io
from PIL import Image
import logging

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

# Initialize FastAPI app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global detector, metrics_logger, fps_meter
    logger.info("Starting up server...")
    import os
    model_path = os.getenv("MODEL_PATH", "models/model_fp16.engine")
    backend = os.getenv("BACKEND", "tensorrt")
    device = os.getenv("DEVICE", "cuda:0")
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Device: {device}")
    try:
        if Path(model_path).exists():
            detector = Detector(
                model_path=model_path,
                backend=backend,
                device=device,
                img_size=640,
                conf_threshold=0.25,
                iou_threshold=0.45,
                warmup_iterations=10
            )
            logger.info("Detector initialized successfully")
        else:
            logger.warning(f"Model not found at {model_path}. Server will start but /detect endpoint will fail.")
            detector = None
        metrics_logger = MetricsLogger()
        fps_meter = FPSMeter()
        logger.info("Server startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down server...")
    if metrics_logger:
        metrics_logger.save_metrics("api_metrics.json")
    logger.info("Shutdown complete")

app = FastAPI(
    title="Edge AI Video Analytics API",
    description="High-performance object detection API with multiple backend support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Edge AI Video Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/detect",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    
    return HealthResponse(
        status="healthy" if detector is not None else "degraded",
        backend=detector.backend if detector else "none",
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
    background_tasks: BackgroundTasks = None
):
    """
    Detect objects in uploaded image.
    
    Args:
        file: Image file to process
    
    Returns:
        Detection results with timing information
    """
    global detector, total_requests, metrics_logger, fps_meter
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
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
        
        logger.info(f"Processed request: {len(detections)} detections in {inference_time_ms:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/visualize")
async def detect_and_visualize(file: UploadFile = File(...)):
    """
    Detect objects and return visualized image.
    
    Args:
        file: Image file to process
    
    Returns:
        Image with bounding boxes drawn
    """
    global detector
    
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
