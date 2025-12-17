"""
Real-time video stream client for Edge AI Video Analytics API.
Demonstrates how to use WebSocket for live camera detection.
"""

import cv2
import asyncio
import websockets
import json
import base64
import time
import numpy as np
from typing import Optional, Callable
import threading
from queue import Queue, Empty


class VideoStreamClient:
    """
    WebSocket client for real-time video stream detection.
    
    Usage:
        client = VideoStreamClient("ws://localhost:8000/stream")
        client.start_camera(camera_id=0)
        # or
        client.start_video("video.mp4")
    """
    
    def __init__(
        self,
        ws_url: str = "ws://localhost:8000/stream",
        fps_limit: int = 30,
        backend: Optional[str] = None,
        on_detection: Optional[Callable] = None,
        show_preview: bool = True
    ):
        """
        Initialize video stream client.
        
        Args:
            ws_url: WebSocket server URL
            fps_limit: Maximum FPS to send
            backend: Backend to use (pytorch, onnx, tensorrt)
            on_detection: Callback function for detection results
            show_preview: Whether to show preview window
        """
        self.ws_url = ws_url
        if backend:
            self.ws_url += f"?backend={backend}&fps_limit={fps_limit}"
        else:
            self.ws_url += f"?fps_limit={fps_limit}"
        
        self.fps_limit = fps_limit
        self.on_detection = on_detection
        self.show_preview = show_preview
        
        self.running = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=10)
        
        self.current_detections = []
        self.current_fps = 0
        self.current_inference_time = 0
        
    def _encode_frame(self, frame: np.ndarray, quality: int = 80) -> str:
        """Encode frame to base64 JPEG."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection boxes on frame."""
        for det in self.current_detections:
            bbox = det.get('bbox', {})
            x1 = int(bbox.get('x1', 0))
            y1 = int(bbox.get('y1', 0))
            x2 = int(bbox.get('x2', 0))
            y2 = int(bbox.get('y2', 0))
            
            confidence = det.get('confidence', 0)
            class_name = det.get('class_name', 'unknown')
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw FPS info
        info_text = f"FPS: {self.current_fps:.1f} | Inference: {self.current_inference_time:.1f}ms"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    async def _ws_handler(self):
        """WebSocket connection handler."""
        try:
            async with websockets.connect(self.ws_url) as ws:
                # Receive connection confirmation
                response = await ws.recv()
                print(f"Connected: {response}")
                
                while self.running:
                    try:
                        # Get frame from queue
                        frame = self.frame_queue.get(timeout=0.1)
                        
                        # Encode and send
                        frame_b64 = self._encode_frame(frame)
                        message = {
                            "frame": frame_b64,
                            "timestamp": time.time()
                        }
                        await ws.send(json.dumps(message))
                        
                        # Receive response
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        result = json.loads(response)
                        
                        if "error" not in result:
                            self.current_detections = result.get("detections", [])
                            self.current_fps = result.get("fps", 0)
                            self.current_inference_time = result.get("inference_time_ms", 0)
                            
                            # Call callback if provided
                            if self.on_detection:
                                self.on_detection(result)
                        
                    except Empty:
                        await asyncio.sleep(0.01)
                    except asyncio.TimeoutError:
                        continue
                        
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.running = False
    
    def _capture_loop(self, source):
        """Video capture loop."""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Failed to open video source: {source}")
            self.running = False
            return
        
        # Get source FPS
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps == 0:
            source_fps = 30
        
        frame_delay = 1.0 / min(source_fps, self.fps_limit)
        last_frame_time = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                # Video ended or error
                if isinstance(source, str):
                    # Video file - restart
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Camera - break
                    break
            
            current_time = time.time()
            
            # FPS limiting
            if current_time - last_frame_time < frame_delay:
                continue
            
            last_frame_time = current_time
            
            # Add to queue (drop old frames)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put(frame.copy())
            
            # Show preview
            if self.show_preview:
                display_frame = self._draw_detections(frame.copy())
                cv2.imshow("Edge AI Video Analytics", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def start_camera(self, camera_id: int = 0):
        """Start streaming from camera."""
        self.running = True
        
        # Start WebSocket handler in background
        ws_thread = threading.Thread(
            target=lambda: asyncio.run(self._ws_handler()),
            daemon=True
        )
        ws_thread.start()
        
        # Run capture loop in main thread
        self._capture_loop(camera_id)
    
    def start_video(self, video_path: str):
        """Start streaming from video file."""
        self.running = True
        
        # Start WebSocket handler in background
        ws_thread = threading.Thread(
            target=lambda: asyncio.run(self._ws_handler()),
            daemon=True
        )
        ws_thread.start()
        
        # Run capture loop in main thread
        self._capture_loop(video_path)
    
    def stop(self):
        """Stop streaming."""
        self.running = False


class HTTPStreamClient:
    """
    HTTP-based streaming client (no WebSocket).
    Simpler but may have higher latency.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        fps_limit: int = 15,
        backend: Optional[str] = None
    ):
        self.api_url = api_url
        self.fps_limit = fps_limit
        self.backend = backend
        self.running = False
        
        self.current_detections = []
        self.current_fps = 0
        
    def _send_frame(self, frame: np.ndarray) -> dict:
        """Send frame to API and get detections."""
        import requests
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
        
        params = {}
        if self.backend:
            params['backend'] = self.backend
        
        try:
            response = requests.post(
                f"{self.api_url}/detect",
                files=files,
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Request error: {e}")
        
        return {}
    
    def start_camera(self, camera_id: int = 0):
        """Start streaming from camera using HTTP."""
        import requests
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Failed to open camera: {camera_id}")
            return
        
        self.running = True
        frame_delay = 1.0 / self.fps_limit
        last_frame_time = 0
        fps_counter = []
        
        print(f"Starting HTTP stream to {self.api_url}")
        print("Press 'q' to quit")
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            current_time = time.time()
            
            # FPS limiting
            if current_time - last_frame_time < frame_delay:
                # Just show frame without detection
                self._draw_frame(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Send frame and get detections
            start = time.time()
            result = self._send_frame(frame)
            latency = (time.time() - start) * 1000
            
            if result:
                self.current_detections = result.get('detections', [])
                
                # Calculate FPS
                fps_counter.append(current_time)
                fps_counter = [t for t in fps_counter if current_time - t < 1.0]
                self.current_fps = len(fps_counter)
                
                last_frame_time = current_time
            
            # Draw and show
            self._draw_frame(frame, latency)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_frame(self, frame: np.ndarray, latency: float = 0):
        """Draw detections and info on frame."""
        for det in self.current_detections:
            bbox = det.get('bbox', {})
            x1 = int(bbox.get('x1', 0))
            y1 = int(bbox.get('y1', 0))
            x2 = int(bbox.get('x2', 0))
            y2 = int(bbox.get('y2', 0))
            
            confidence = det.get('confidence', 0)
            class_name = det.get('class_name', 'unknown')
            
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw info
        info = f"FPS: {self.current_fps} | Latency: {latency:.0f}ms"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Edge AI Video Analytics (HTTP)", frame)
    
    def stop(self):
        """Stop streaming."""
        self.running = False


def main():
    """Demo main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Stream Client")
    parser.add_argument("--source", type=str, default="0", help="Camera ID or video path")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/stream", help="WebSocket URL")
    parser.add_argument("--fps", type=int, default=30, help="FPS limit")
    parser.add_argument("--backend", type=str, choices=["pytorch", "onnx", "tensorrt"], help="Backend")
    parser.add_argument("--http", action="store_true", help="Use HTTP instead of WebSocket")
    
    args = parser.parse_args()
    
    # Determine source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    if args.http:
        # HTTP client
        http_url = args.url.replace("ws://", "http://").replace("/stream", "")
        client = HTTPStreamClient(
            api_url=http_url,
            fps_limit=args.fps,
            backend=args.backend
        )
        
        if isinstance(source, int):
            client.start_camera(source)
        else:
            print("HTTP client only supports camera input")
    else:
        # WebSocket client
        client = VideoStreamClient(
            ws_url=args.url,
            fps_limit=args.fps,
            backend=args.backend
        )
        
        if isinstance(source, int):
            client.start_camera(source)
        else:
            client.start_video(source)


if __name__ == "__main__":
    main()
