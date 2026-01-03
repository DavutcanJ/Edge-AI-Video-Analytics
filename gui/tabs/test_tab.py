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

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8002"


class TestTab(BaseTab):
    """Inference Testing and Live Camera Tab"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.loaded_image = None
        self.displayed_image = None
        self.camera_active = False
        self.camera_thread = None
        self.cap = None
    
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
            self.loaded_image = path
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
        if not self.loaded_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        logger.info("Running object detection")
        
        def run():
            try:
                with open(self.loaded_image, "rb") as f:
                    files = {"file": (self.loaded_image, f, "image/png")}
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
        if not self.loaded_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        logger.info("Visualizing detections")
        
        def run():
            try:
                with open(self.loaded_image, "rb") as f:
                    files = {"file": (self.loaded_image, f, "image/png")}
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
        if self.camera_active:
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        """Start camera stream"""
        if self.camera_active:
            return
        
        try:
            # Get camera source
            cam_source = self.widgets['camera_var'].get()
            try:
                cam_source = int(cam_source)
            except ValueError:
                pass  # Keep as string (file path or URL)
            
            # Open camera
            self.cap = cv2.VideoCapture(cam_source)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open camera: {cam_source}")
                return
            
            self.camera_active = True
            self.widgets['start_camera_btn'].configure(
                text="⏹️ Stop Camera",
                fg_color="#C92A2A"
            )
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            
            logger.info(f"Camera started: {cam_source}")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start camera:\n{e}")
    
    def _stop_camera(self):
        """Stop camera stream"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.widgets['start_camera_btn'].configure(
            text="📹 Start Camera",
            fg_color="#1971C2"
        )
        
        logger.info("Camera stopped")
    
    def _camera_loop(self):
        """Main camera loop with detection"""
        frame_count = 0
        start_time = time.time()
        
        while self.camera_active:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read camera frame")
                    break
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Get current backend
                backend = self.widgets['backend_var'].get()
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                
                # Send to API for detection
                try:
                    response = requests.post(
                        f"{API_URL}/detect/visualize",
                        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                        params={"backend": backend},
                        timeout=2
                    )
                    
                    if response.status_code == 200:
                        from io import BytesIO
                        img = Image.open(BytesIO(response.content))
                        img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                        
                        # Store reference and update in main thread
                        self._update_camera_display(img, fps, "Live Camera")
                    else:
                        # Show raw frame if detection fails
                        img = Image.fromarray(frame_rgb)
                        img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                        self._update_camera_display(img, fps, "Live (No Detection)")
                        
                except requests.RequestException as e:
                    # Show raw frame if API is down
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                    self._update_camera_display(img, fps, "Live (No API)")
                
                # Small delay to prevent overload
                time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"Camera loop error: {e}", exc_info=True)
                break
        
        # Cleanup
        if self.cap:
            self.cap.release()
    
    def _update_camera_display(self, image, fps, status):
        """Update camera display in main thread"""
        # Keep strong reference to image
        image_copy = image.copy()
        
        def do_update():
            try:
                self.displayed_image = CTkImage(
                    light_image=image_copy,
                    dark_image=image_copy,
                    size=image_copy.size
                )
                self.widgets['image_label'].configure(
                    image=self.displayed_image,
                    text=""
                )
                self.widgets['stats_label'].configure(
                    text=f"FPS: {fps:.1f} | {status}"
                )
            except Exception as e:
                logger.error(f"Display update error: {e}")
        
        self.app.after(0, do_update)
    
    def cleanup(self):
        """Cleanup when tab is closed"""
        if self.camera_active:
            self._stop_camera()
