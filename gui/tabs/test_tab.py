"""
Test Tab - Inference Testing and Live Camera
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from customtkinter import CTkImage
import logging
import requests
import threading
from pathlib import Path
from PIL import Image
import cv2
import time
import queue

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8002"


class TestTab(BaseTab):
    """Inference Testing and Live Camera Tab"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.loaded_image_path = None
        self.displayed_image = None
        self.camera_running = False
        self.camera_thread = None
        self.frame_queue = None
    
    def setup(self):
        """Setup test tab UI"""
        logger.debug("Setting up Test tab")
        
        # Main container
        main_frame = ctk.CTkFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Image display
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(left_frame, text="🧪 Inference Testing", 
                    font=("Arial", 18, "bold")).pack(pady=10)
        
        # Image display area
        self.widgets['image_label'] = ctk.CTkLabel(
            left_frame,
            text="Load an image to test",
            width=640,
            height=480
        )
        self.widgets['image_label'].pack(pady=10, padx=10, fill="both", expand=True)
        
        # Stats
        self.widgets['stats_label'] = ctk.CTkLabel(
            left_frame,
            text="FPS: -- | Inference: -- ms | Detections: --",
            font=("Consolas", 12)
        )
        self.widgets['stats_label'].pack(pady=5)
        
        # Right: Controls
        right_frame = ctk.CTkFrame(main_frame, width=350)
        right_frame.pack(side="right", fill="y", padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        ctk.CTkLabel(right_frame, text="⚙️ Controls", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        # Backend selection
        ctk.CTkLabel(right_frame, text="Backend:", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        self.widgets['backend_var'] = ctk.StringVar(value="tensorrt")
        ctk.CTkOptionMenu(
            right_frame,
            values=["tensorrt", "onnx", "pytorch"],
            variable=self.widgets['backend_var'],
            width=200
        ).pack(pady=5)
        
        ctk.CTkLabel(right_frame, text="─" * 30).pack(pady=10)
        
        # Image testing
        ctk.CTkLabel(right_frame, text="Image Detection", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        ctk.CTkButton(
            right_frame, 
            text="📂 Load Image", 
            command=self._load_image,
            width=250
        ).pack(pady=5)
        
        ctk.CTkButton(
            right_frame, 
            text="🔍 Detect Objects", 
            command=self._detect_objects,
            width=250,
            fg_color="#2B8A3E"
        ).pack(pady=5)
        
        ctk.CTkButton(
            right_frame, 
            text="🎨 Visualize Results", 
            command=self._visualize_detections,
            width=250,
            fg_color="#E67700"
        ).pack(pady=5)
        
        ctk.CTkLabel(right_frame, text="─" * 30).pack(pady=10)
        
        # Camera testing
        ctk.CTkLabel(right_frame, text="Live Camera", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        # Camera source
        cam_source_frame = ctk.CTkFrame(right_frame)
        cam_source_frame.pack(pady=5)
        
        ctk.CTkLabel(cam_source_frame, text="Source:", 
                    font=("Arial", 11)).pack(side="left", padx=5)
        
        self.widgets['camera_var'] = ctk.StringVar(value="0")
        ctk.CTkEntry(
            cam_source_frame,
            textvariable=self.widgets['camera_var'],
            width=80
        ).pack(side="left", padx=5)
        
        self.widgets['start_camera_btn'] = ctk.CTkButton(
            right_frame,
            text="📹 Start Camera",
            command=self._toggle_camera,
            width=250,
            fg_color="#1971C2"
        )
        self.widgets['start_camera_btn'].pack(pady=5)
        
        ctk.CTkLabel(right_frame, text="─" * 30).pack(pady=10)
        
        # Results display
        ctk.CTkLabel(right_frame, text="Detection Results", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.widgets['results'] = ctk.CTkTextbox(right_frame, width=300, height=300)
        self.widgets['results'].pack(pady=5, padx=10, fill="both", expand=True)
        self.widgets['results'].insert("end", "💡 Load an image and run detection\n\n")
        self.widgets['results'].insert("end", "Results will appear here:\n")
        self.widgets['results'].insert("end", "• Object classes\n")
        self.widgets['results'].insert("end", "• Confidence scores\n")
        self.widgets['results'].insert("end", "• Bounding boxes\n")
        
        logger.debug("Test tab setup complete")
    
    def _load_image(self):
        """Load image for testing"""
        logger.debug("Loading test image")
        
        # CRITICAL: Stop camera before loading image
        if self.camera_running:
            logger.info("Stopping camera before loading image")
            self._stop_camera()
            time.sleep(0.2)
        
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not path:
            return
        
        try:
            self.loaded_image_path = path
            img = Image.open(path)
            
            # Resize for display
            max_size = (640, 480)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            
            # Clear previous
            if self.displayed_image:
                del self.displayed_image
            
            # Display
            self.displayed_image = CTkImage(
                light_image=img, 
                dark_image=img, 
                size=img.size
            )
            self.widgets['image_label'].configure(
                image=self.displayed_image, 
                text=""
            )
            
            # Update results
            self.widgets['results'].delete("1.0", "end")
            self.widgets['results'].insert("end", f"✅ Loaded: {Path(path).name}\n\n")
            self.widgets['results'].insert("end", "Click 'Detect Objects' to run inference\n")
            
            logger.info(f"Loaded image: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def _detect_objects(self):
        """Run object detection"""
        if not self.loaded_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        logger.info("Running object detection")
        
        def run():
            try:
                with open(self.loaded_image_path, "rb") as f:
                    files = {"file": (self.loaded_image_path, f, "image/png")}
                    params = {"backend": self.widgets['backend_var'].get()}
                    
                    response = requests.post(
                        f"{API_URL}/detect",
                        files=files,
                        params=params,
                        timeout=30
                    )
                
                result = response.json()
                
                # Update stats
                inf_time = result.get("inference_time_ms", 0)
                num_det = result.get("num_detections", 0)
                
                self.app.after(0, lambda: self.widgets['stats_label'].configure(
                    text=f"Inference: {inf_time:.1f}ms | Detections: {num_det}"
                ))
                
                # Show results
                def update_results():
                    self.widgets['results'].delete("1.0", "end")
                    self.widgets['results'].insert("end", f"✅ Detection Complete\n")
                    self.widgets['results'].insert("end", f"═" * 40 + "\n\n")
                    self.widgets['results'].insert("end", f"Detections: {num_det}\n")
                    self.widgets['results'].insert("end", f"Inference: {inf_time:.1f}ms\n\n")
                    
                    for det in result.get("detections", []):
                        self.widgets['results'].insert("end", 
                            f"• {det['class_name']}: {det['confidence']:.2%}\n")
                
                self.app.after(0, update_results)
                
            except Exception as e:
                logger.error(f"Detection failed: {e}", exc_info=True)
                self.app.after(0, lambda: messagebox.showerror(
                    "Error", f"Detection failed:\n{e}"
                ))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _visualize_detections(self):
        """Get visualized results"""
        if not self.loaded_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        logger.info("Visualizing detections")
        
        def run():
            try:
                with open(self.loaded_image_path, "rb") as f:
                    files = {"file": (self.loaded_image_path, f, "image/png")}
                    params = {"backend": self.widgets['backend_var'].get()}
                    
                    response = requests.post(
                        f"{API_URL}/detect/visualize",
                        files=files,
                        params=params,
                        timeout=30
                    )
                
                from io import BytesIO
                img = Image.open(BytesIO(response.content))
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                
                def update_image():
                    if self.displayed_image:
                        del self.displayed_image
                    
                    self.displayed_image = CTkImage(
                        light_image=img,
                        dark_image=img,
                        size=img.size
                    )
                    self.widgets['image_label'].configure(
                        image=self.displayed_image,
                        text=""
                    )
                
                self.app.after(0, update_image)
                
            except Exception as e:
                logger.error(f"Visualization failed: {e}", exc_info=True)
                self.app.after(0, lambda: messagebox.showerror(
                    "Error", f"Visualization failed:\n{e}"
                ))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        """Start live camera detection - SAFE VERSION with queue"""
        logger.info("Starting camera")
        
        # CRITICAL: Ensure camera is fully stopped before starting
        if self.camera_running:
            logger.warning("Camera already running, stopping first")
            self._stop_camera()
            time.sleep(0.3)
        
        self.camera_running = True
        self.widgets['start_camera_btn'].configure(
            text="⏹️ Stop Camera",
            fg_color="#C92A2A"
        )
        
        # Clear loaded image to prevent conflicts
        self.loaded_image_path = None
        
        # Create frame queue for thread-safe communication
        self.frame_queue = queue.Queue(maxsize=2)
        
        def camera_loop():
            cap = None
            try:
                cam_source = self.widgets['camera_var'].get()
                try:
                    cam_source = int(cam_source)
                except:
                    pass
            except:
                cam_source = 0
            
            logger.info(f"Opening camera {cam_source}")
            cap = cv2.VideoCapture(cam_source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {cam_source}")
                self.app.after(0, lambda: messagebox.showerror("Error", f"Failed to open camera {cam_source}"))
                self.camera_running = False
                self.app.after(0, lambda: self.widgets['start_camera_btn'].configure(
                    text="📹 Start Camera", fg_color="#1971C2"
                ))
                return
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera opened successfully, starting loop")
            
            frame_count = 0
            fps_times = []
            last_detection_time = 0
            detection_interval = 0.5  # Detect every 500ms
            current_detections = []
            current_inf_time = 0
            
            logger.info(f"Entering main camera loop, camera_running={self.camera_running}")
            
            while self.camera_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Log first few frames
                if frame_count <= 3:
                    logger.info(f"Read frame {frame_count}: shape={frame.shape}")
                elif frame_count == 4:
                    logger.info("Camera reading frames normally")
                
                current_time = time.time()
                
                # Calculate FPS
                fps_times.append(current_time)
                fps_times = [t for t in fps_times if current_time - t < 1.0]
                fps = len(fps_times)
                
                # Run detection periodically
                if current_time - last_detection_time >= detection_interval:
                    last_detection_time = current_time
                    
                    # Run detection in separate thread
                    def run_detection():
                        try:
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                            params = {"backend": self.widgets['backend_var'].get()}
                            r = requests.post(f"{API_URL}/detect", files=files, params=params, timeout=2)
                            
                            if r.status_code == 200:
                                result = r.json()
                                nonlocal current_detections, current_inf_time
                                current_detections = result.get("detections", [])
                                current_inf_time = result.get("inference_time_ms", 0)
                        except Exception as e:
                            if frame_count % 30 == 0:  # Log occasionally
                                logger.warning(f"Detection failed: {e}")
                    
                    threading.Thread(target=run_detection, daemon=True).start()
                
                # Draw detections on frame
                display_frame = frame.copy()
                for det in current_detections:
                    bbox = det.get("bbox", {})
                    x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                    x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add FPS overlay directly on frame
                cv2.putText(display_frame, f"FPS: {fps}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Put frame in queue (non-blocking)
                try:
                    # Convert to RGB here in camera thread
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    self.frame_queue.put_nowait((frame_rgb, fps, len(current_detections), current_inf_time))
                except:
                    pass  # Queue full, skip frame
                
                time.sleep(0.033)  # ~30 FPS
            
            logger.info("Camera loop ended")
            
            if cap:
                cap.release()
                logger.info("Camera released")
            
            self.app.after(0, lambda: self.widgets['start_camera_btn'].configure(
                text="📹 Start Camera", fg_color="#1971C2"
            ))
        
        def update_from_queue():
            """Update GUI from queue - runs in main thread"""
            if not self.camera_running:
                return
            
            try:
                # Get frame from queue (non-blocking)
                frame_rgb, fps, det_count, inf_time = self.frame_queue.get_nowait()
                
                # Create PIL Image in main thread
                img_pil = Image.fromarray(frame_rgb)
                img_pil.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                # Create CTkImage in main thread
                ctk_img = CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
                
                # Update label
                self.widgets['image_label'].configure(image=ctk_img, text="")
                self.displayed_image = ctk_img  # Keep reference
                
                # Update stats
                self.widgets['stats_label'].configure(
                    text=f"FPS: {fps} | Inference: {inf_time:.1f}ms | Detections: {det_count}"
                )
                
            except:
                pass  # Queue empty
            
            # Schedule next update (every 33ms = ~30 FPS)
            if self.camera_running:
                self.app.after(33, update_from_queue)
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=camera_loop, daemon=True)
        self.camera_thread.start()
        logger.info("Camera thread started")
        
        # Start GUI update loop in main thread
        self.app.after(100, update_from_queue)
    
    def _stop_camera(self):
        """Stop live camera - IMPROVED cleanup"""
        logger.info("Stopping camera")
        self.camera_running = False
        
        # Clear queue if it exists
        if hasattr(self, 'frame_queue') and self.frame_queue:
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except:
                pass
        
        # Wait for thread to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        
        # Clear image
        self.widgets['image_label'].configure(image=None, text="Camera stopped")
        if self.displayed_image:
            del self.displayed_image
            self.displayed_image = None
        
        self.widgets['start_camera_btn'].configure(
            text="📹 Start Camera",
            fg_color="#1971C2"
        )
        
        logger.info("Camera stopped successfully")
    
    def cleanup(self):
        """Cleanup when tab is closed"""
        if self.camera_running:
            self._stop_camera()
