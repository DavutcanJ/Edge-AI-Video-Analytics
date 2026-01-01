"""
Edge AI Video Analytics - Management GUI V2
Optimized with debugging, better performance, and fixed camera/image loading
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from customtkinter import CTkImage
import requests
import threading
import time
import subprocess
import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from queue import Queue
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce urllib3 logging noise
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
API_URL = "http://localhost:8002"


class EdgeAIManager(ctk.CTk):
    """Main application window with tabbed interface - Optimized version."""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Edge AI Manager")
        
        self.title("Edge AI Video Analytics - Management Console v2")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # State
        self.api_process = None
        self.training_process = None
        self.camera_running = False
        self.camera_thread = None
        self.image_queue = Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        
        # Test tab state
        self.loaded_image_path = None
        self.displayed_image = None
        self.current_detections = None
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self, width=1380, height=780)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Add tabs - lazy loading for performance
        self.tab_api = self.tabview.add("🌐 API")
        self.tab_training = self.tabview.add("🎯 Training")
        self.tab_export = self.tabview.add("📦 Export")
        self.tab_test = self.tabview.add("🧪 Test")
        self.tab_monitoring = self.tabview.add("📊 Monitoring")
        
        # Setup tabs
        logger.debug("Setting up tabs")
        self._setup_api_tab()
        self._setup_training_tab()
        self._setup_export_tab()
        self._setup_test_tab()
        self._setup_monitoring_tab()
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("Edge AI Manager initialized successfully")
    
    # =========================================================================
    # API TAB
    # =========================================================================
    def _setup_api_tab(self):
        """Setup API management tab."""
        logger.debug("Setting up API tab")
        
        # Left panel - Controls
        left_frame = ctk.CTkFrame(self.tab_api)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(left_frame, text="API Server Control", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Server status
        self.api_status_label = ctk.CTkLabel(left_frame, text="Status: ⚪ Stopped", font=("Arial", 14))
        self.api_status_label.pack(pady=10)
        
        # Backend selection
        ctk.CTkLabel(left_frame, text="Backend:", font=("Arial", 12)).pack(pady=(20, 5))
        self.backend_var = ctk.StringVar(value="tensorrt")
        self.backend_menu = ctk.CTkOptionMenu(
            left_frame,
            values=["tensorrt", "onnx", "pytorch"],
            variable=self.backend_var,
            width=200
        )
        self.backend_menu.pack(pady=5)
        
        # Port
        ctk.CTkLabel(left_frame, text="Port:", font=("Arial", 12)).pack(pady=(10, 5))
        self.port_var = ctk.StringVar(value="8002")
        self.port_entry = ctk.CTkEntry(left_frame, textvariable=self.port_var, width=200)
        self.port_entry.pack(pady=5)
        
        # Start/Stop buttons
        self.start_api_btn = ctk.CTkButton(
            left_frame, text="▶️ Start Server", command=self.start_api_server,
            fg_color="green", width=200
        )
        self.start_api_btn.pack(pady=10)
        
        self.stop_api_btn = ctk.CTkButton(
            left_frame, text="⏹️ Stop Server", command=self.stop_api_server,
            fg_color="red", width=200, state="disabled"
        )
        self.stop_api_btn.pack(pady=5)
        
        # Quick actions
        ctk.CTkLabel(left_frame, text="─" * 25).pack(pady=10)
        ctk.CTkLabel(left_frame, text="Quick Actions", font=("Arial", 14, "bold")).pack(pady=5)
        
        ctk.CTkButton(left_frame, text="💚 Health Check", command=self.check_health, width=200).pack(pady=5)
        ctk.CTkButton(left_frame, text="📊 Get Metrics", command=self.get_metrics, width=200).pack(pady=5)
        ctk.CTkButton(left_frame, text="🔄 Switch Backend", command=self.switch_backend, width=200).pack(pady=5)
        
        # Right panel - Logs
        right_frame = ctk.CTkFrame(self.tab_api)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(right_frame, text="Server Logs", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.api_log = ctk.CTkTextbox(right_frame, width=800, height=600)
        self.api_log.pack(pady=10, padx=10, fill="both", expand=True)
        self.api_log.insert("end", "API server logs will appear here...\n")
        
        logger.debug("API tab setup complete")
    
    def start_api_server(self):
        """Start the FastAPI server."""
        port = self.port_var.get()
        server_script = PROJECT_ROOT / "api" / "server.py"
        logger.info(f"Starting API server on port {port}")
        
        def run_server():
            try:
                self.api_process = subprocess.Popen(
                    [sys.executable, str(server_script)],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                logger.info(f"API process started with PID {self.api_process.pid}")
                self.after(0, lambda: self.api_status_label.configure(text="Status: 🟢 Running"))
                self.after(0, lambda: self.start_api_btn.configure(state="disabled"))
                self.after(0, lambda: self.stop_api_btn.configure(state="normal"))
                
                for line in self.api_process.stdout:
                    self.after(0, lambda l=line: self._append_api_log(l))
                    
            except Exception as e:
                logger.error(f"Failed to start API server: {e}", exc_info=True)
                self.after(0, lambda: self._append_api_log(f"Error: {e}\n"))
        
        self._append_api_log(f"Starting API server on port {port}...\n")
        threading.Thread(target=run_server, daemon=True).start()
    
    def stop_api_server(self):
        """Stop the FastAPI server."""
        logger.info("Stopping API server")
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
            except Exception as e:
                logger.error(f"Error stopping API: {e}")
            finally:
                self.api_process = None
                self.api_status_label.configure(text="Status: ⚪ Stopped")
                self.start_api_btn.configure(state="normal")
                self.stop_api_btn.configure(state="disabled")
                self._append_api_log("Server stopped.\n")
        else:
            logger.warning("No API process to stop")
            self.api_status_label.configure(text="Status: ⚪ Stopped")
            self.start_api_btn.configure(state="normal")
            self.stop_api_btn.configure(state="disabled")
    
    def _append_api_log(self, text):
        """Append text to API log."""
        self.api_log.insert("end", text)
        self.api_log.see("end")
        # Limit log size to prevent memory issues
        lines = int(self.api_log.index('end-1c').split('.')[0])
        if lines > 1000:
            self.api_log.delete('1.0', '500.0')
    
    def check_health(self):
        """Check API health."""
        logger.debug("Checking API health")
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            result = r.json()
            logger.info(f"Health check result: {result}")
            messagebox.showinfo("Health", json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            messagebox.showerror("Error", f"Health check failed: {e}")
    
    def get_metrics(self):
        """Get API metrics."""
        logger.debug("Getting API metrics")
        try:
            r = requests.get(f"{API_URL}/metrics", timeout=5)
            result = r.json()
            logger.info(f"Metrics: {result}")
            messagebox.showinfo("Metrics", json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"Metrics request failed: {e}")
            messagebox.showerror("Error", f"Metrics request failed: {e}")
    
    def switch_backend(self):
        """Switch API backend."""
        backend = self.backend_var.get()
        logger.info(f"Switching to {backend} backend")
        try:
            r = requests.post(f"{API_URL}/backends/{backend}", timeout=5)
            if r.status_code == 200:
                logger.info(f"Successfully switched to {backend}")
                messagebox.showinfo("Success", f"Switched to {backend} backend")
            else:
                logger.warning(f"Backend switch failed: {r.text}")
                messagebox.showerror("Error", r.text)
        except Exception as e:
            logger.error(f"Backend switch failed: {e}", exc_info=True)
            messagebox.showerror("Error", f"Backend switch failed: {e}")
    
    # =========================================================================
    # TRAINING TAB
    # =========================================================================
    def _setup_training_tab(self):
        """Setup training management tab."""
        logger.debug("Setting up Training tab")
        
        # Left panel - Training config
        left_frame = ctk.CTkFrame(self.tab_training, width=400)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        ctk.CTkLabel(left_frame, text="Training Configuration", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Model selection
        ctk.CTkLabel(left_frame, text="Base Model:", font=("Arial", 12)).pack(pady=(10, 5))
        self.train_model_var = ctk.StringVar(value="yolov8n.pt")
        ctk.CTkOptionMenu(
            left_frame,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"],
            variable=self.train_model_var,
            width=200
        ).pack(pady=5)
        
        # Dataset path
        ctk.CTkLabel(left_frame, text="Dataset Config:", font=("Arial", 12)).pack(pady=(10, 5))
        dataset_frame = ctk.CTkFrame(left_frame)
        dataset_frame.pack(pady=5, fill="x", padx=10)
        self.dataset_var = ctk.StringVar(value="")
        ctk.CTkEntry(dataset_frame, textvariable=self.dataset_var, width=250, state="readonly").pack(side="left", padx=5)
        ctk.CTkButton(dataset_frame, text="📂", width=40, command=self.browse_dataset_root).pack(side="left")
        
        # Epochs
        ctk.CTkLabel(left_frame, text="Epochs:", font=("Arial", 12)).pack(pady=(10, 5))
        self.epochs_var = ctk.StringVar(value="100")
        ctk.CTkEntry(left_frame, textvariable=self.epochs_var, width=200).pack(pady=5)
        
        # Batch size
        ctk.CTkLabel(left_frame, text="Batch Size:", font=("Arial", 12)).pack(pady=(10, 5))
        self.batch_var = ctk.StringVar(value="16")
        ctk.CTkEntry(left_frame, textvariable=self.batch_var, width=200).pack(pady=5)
        
        # Image size
        ctk.CTkLabel(left_frame, text="Image Size:", font=("Arial", 12)).pack(pady=(10, 5))
        self.imgsz_var = ctk.StringVar(value="640")
        ctk.CTkEntry(left_frame, textvariable=self.imgsz_var, width=200).pack(pady=5)
        
        # Options
        ctk.CTkLabel(left_frame, text="─" * 25).pack(pady=10)
        self.use_wandb_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left_frame, text="Use Weights & Biases", variable=self.use_wandb_var).pack(pady=5)
        
        self.use_amp_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left_frame, text="Mixed Precision (AMP)", variable=self.use_amp_var).pack(pady=5)
        
        # Buttons
        ctk.CTkLabel(left_frame, text="─" * 25).pack(pady=10)
        
        self.start_train_btn = ctk.CTkButton(
            left_frame, text="🚀 Start Training", command=self.start_training,
            fg_color="green", width=200
        )
        self.start_train_btn.pack(pady=10)
        
        self.stop_train_btn = ctk.CTkButton(
            left_frame, text="⏹️ Stop Training", command=self.stop_training,
            fg_color="red", width=200, state="disabled"
        )
        self.stop_train_btn.pack(pady=5)
        
        # Right panel - Training logs
        right_frame = ctk.CTkFrame(self.tab_training)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(right_frame, text="Training Progress", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Progress bar
        self.train_progress = ctk.CTkProgressBar(right_frame, width=700)
        self.train_progress.pack(pady=10)
        self.train_progress.set(0)
        
        self.train_status_label = ctk.CTkLabel(right_frame, text="Ready to train", font=("Arial", 12))
        self.train_status_label.pack(pady=5)
        
        self.train_log = ctk.CTkTextbox(right_frame, width=800, height=500)
        self.train_log.pack(pady=10, padx=10, fill="both", expand=True)
        self.train_log.insert("end", "Training logs will appear here...\n")
        
        logger.debug("Training tab setup complete")
    
    def browse_dataset_root(self):
        """Browse for dataset root folder and auto-generate dataset.yaml."""
        logger.debug("Browsing for dataset root")
        path = filedialog.askdirectory(title="Select dataset root folder (should contain images/ and labels/)")
        if path:
            try:
                from gui.dataset_yaml_util import create_dataset_yaml
                yaml_path = create_dataset_yaml(path)
                self.dataset_var.set(yaml_path)
                logger.info(f"Dataset config created: {yaml_path}")
                messagebox.showinfo("Dataset Config", f"dataset.yaml created: {yaml_path}")
            except Exception as e:
                logger.error(f"Failed to create dataset.yaml: {e}", exc_info=True)
                messagebox.showerror("Error", f"Failed to create dataset.yaml: {e}")
    
    def start_training(self):
        """Start model training."""
        logger.info("Starting training")
        
        def run_training():
            try:
                train_script = PROJECT_ROOT / "training" / "train.py"
                
                cmd = [
                    sys.executable, str(train_script),
                    "--model", self.train_model_var.get(),
                    "--data", self.dataset_var.get(),
                    "--epochs", self.epochs_var.get(),
                    "--batch", self.batch_var.get(),
                    "--imgsz", self.imgsz_var.get(),
                ]
                
                if not self.use_wandb_var.get():
                    cmd.extend(["--project", "local"])
                if not self.use_amp_var.get():
                    cmd.append("--amp=False")
                
                logger.debug(f"Training command: {' '.join(cmd)}")
                
                self.after(0, lambda: self.start_train_btn.configure(state="disabled"))
                self.after(0, lambda: self.stop_train_btn.configure(state="normal"))
                self.after(0, lambda: self.train_status_label.configure(text="Training in progress..."))
                
                self.training_process = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                
                logger.info(f"Training process started with PID {self.training_process.pid}")
                
                for line in self.training_process.stdout:
                    self.after(0, lambda l=line: self._append_train_log(l))
                    # Parse epoch progress
                    if "Epoch" in line:
                        try:
                            parts = line.split()
                            if len(parts) > 1:
                                epoch_info = parts[1].split('/')
                                if len(epoch_info) == 2:
                                    current = int(epoch_info[0])
                                    total = int(epoch_info[1])
                                    progress = current / total
                                    self.after(0, lambda p=progress: self.train_progress.set(p))
                        except:
                            pass
                
                self.after(0, lambda: self.train_status_label.configure(text="Training complete!"))
                self.after(0, lambda: self.start_train_btn.configure(state="normal"))
                self.after(0, lambda: self.stop_train_btn.configure(state="disabled"))
                logger.info("Training completed")
                
            except Exception as e:
                logger.error(f"Training failed: {e}", exc_info=True)
                self.after(0, lambda: self._append_train_log(f"Error: {e}\n"))
                self.after(0, lambda: self.start_train_btn.configure(state="normal"))
                self.after(0, lambda: self.stop_train_btn.configure(state="disabled"))
        
        self._append_train_log("Starting training...\n")
        threading.Thread(target=run_training, daemon=True).start()
    
    def stop_training(self):
        """Stop training process."""
        logger.info("Stopping training")
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
            self.train_status_label.configure(text="Training stopped")
            self.start_train_btn.configure(state="normal")
            self.stop_train_btn.configure(state="disabled")
            self._append_train_log("Training stopped by user.\n")
    
    def _append_train_log(self, text):
        """Append text to training log."""
        self.train_log.insert("end", text)
        self.train_log.see("end")
        # Limit log size
        lines = int(self.train_log.index('end-1c').split('.')[0])
        if lines > 1000:
            self.train_log.delete('1.0', '500.0')
    
    # =========================================================================
    # EXPORT TAB
    # =========================================================================
    def _setup_export_tab(self):
        """Setup export/optimization tab."""
        logger.debug("Setting up Export tab")
        
        main_frame = ctk.CTkFrame(self.tab_export)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(main_frame, text="Model Export & Optimization", font=("Arial", 20, "bold")).pack(pady=10)
        
        # Model selection
        ctk.CTkLabel(main_frame, text="Select Model:", font=("Arial", 14)).pack(pady=10)
        model_frame = ctk.CTkFrame(main_frame)
        model_frame.pack(pady=10)
        
        self.export_model_var = ctk.StringVar(value="")
        ctk.CTkEntry(model_frame, textvariable=self.export_model_var, width=400, state="readonly").pack(side="left", padx=5)
        ctk.CTkButton(model_frame, text="📂 Browse", command=self.browse_export_model).pack(side="left")
        
        # Export options
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(pady=20, fill="x", padx=50)
        
        # ONNX Export
        onnx_frame = ctk.CTkFrame(options_frame)
        onnx_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(onnx_frame, text="ONNX Export", font=("Arial", 16, "bold")).pack(pady=10)
        ctk.CTkButton(onnx_frame, text="📦 Export to ONNX", command=self.export_onnx, width=200).pack(pady=10)
        
        # TensorRT Build
        trt_frame = ctk.CTkFrame(options_frame)
        trt_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(trt_frame, text="TensorRT Engine", font=("Arial", 16, "bold")).pack(pady=10)
        
        ctk.CTkLabel(trt_frame, text="Precision:").pack(pady=5)
        self.trt_precision_var = ctk.StringVar(value="fp16")
        ctk.CTkOptionMenu(trt_frame, values=["fp32", "fp16", "int8"], variable=self.trt_precision_var).pack(pady=5)
        
        ctk.CTkLabel(trt_frame, text="Workspace (MB):").pack(pady=5)
        self.trt_workspace_var = ctk.StringVar(value="4096")
        ctk.CTkEntry(trt_frame, textvariable=self.trt_workspace_var, width=150).pack(pady=5)
        
        ctk.CTkButton(trt_frame, text="⚡ Build TensorRT Engine", command=self.build_tensorrt, width=200).pack(pady=10)
        
        # Log
        ctk.CTkLabel(main_frame, text="Export Logs", font=("Arial", 14, "bold")).pack(pady=10)
        self.export_log = ctk.CTkTextbox(main_frame, width=1000, height=300)
        self.export_log.pack(pady=10, padx=20, fill="both", expand=True)
        
        logger.debug("Export tab setup complete")
    
    def browse_export_model(self):
        """Browse for model to export."""
        logger.debug("Browsing for export model")
        path = filedialog.askopenfilename(filetypes=[("PyTorch models", "*.pt")])
        if path:
            self.export_model_var.set(path)
            logger.info(f"Selected model for export: {path}")
    
    def export_onnx(self):
        """Export model to ONNX format."""
        if not self.export_model_var.get():
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        logger.info(f"Exporting {self.export_model_var.get()} to ONNX")
        
        def run_export():
            try:
                model_path = self.export_model_var.get()
                
                self.after(0, lambda: self._append_export_log(f"Exporting {Path(model_path).name} to ONNX...\n"))
                
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "optimization" / "export_to_onnx.py"),
                    "--model", model_path
                ]
                
                logger.debug(f"ONNX export command: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                
                for line in process.stdout:
                    self.after(0, lambda l=line: self._append_export_log(l))
                
                logger.info("ONNX export completed")
                
            except Exception as e:
                logger.error(f"ONNX export failed: {e}", exc_info=True)
                self.after(0, lambda: self._append_export_log(f"Error: {e}\n"))
        
        threading.Thread(target=run_export, daemon=True).start()
    
    def build_tensorrt(self):
        """Build TensorRT engine."""
        if not self.export_model_var.get():
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        logger.info("Building TensorRT engine")
        
        def run_export():
            try:
                model_path = self.export_model_var.get()
                onnx_path = model_path.replace('.pt', '.onnx')
                precision = self.trt_precision_var.get()
                
                self.after(0, lambda: self._append_export_log(f"Building TensorRT engine ({precision})...\n"))
                
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "optimization" / "build_trt_engine.py"),
                    "--onnx", onnx_path,
                    "--precision", precision,
                    "--workspace", self.trt_workspace_var.get()
                ]
                
                logger.debug(f"TensorRT build command: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                
                for line in process.stdout:
                    self.after(0, lambda l=line: self._append_export_log(l))
                
                logger.info("TensorRT engine build completed")
                
            except Exception as e:
                logger.error(f"TensorRT build failed: {e}", exc_info=True)
                self.after(0, lambda: self._append_export_log(f"Error: {e}\n"))
        
        threading.Thread(target=run_export, daemon=True).start()
    
    def _append_export_log(self, text):
        """Append text to export log."""
        self.export_log.insert("end", text)
        self.export_log.see("end")
    
    # =========================================================================
    # TEST TAB - OPTIMIZED WITH CAMERA FIX
    # =========================================================================
    def _setup_test_tab(self):
        """Setup testing/inference tab - Optimized version."""
        logger.debug("Setting up Test tab")
        
        # Left panel - Image display
        left_frame = ctk.CTkFrame(self.tab_test)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.test_image_label = ctk.CTkLabel(
            left_frame, text="Load an image or start camera",
            width=640, height=480
        )
        self.test_image_label.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.test_stats_label = ctk.CTkLabel(
            left_frame, text="FPS: -- | Inference: -- ms | Detections: --",
            font=("Consolas", 12)
        )
        self.test_stats_label.pack(pady=5)
        
        # Right panel - Controls
        right_frame = ctk.CTkFrame(self.tab_test, width=350)
        right_frame.pack(side="right", fill="y", padx=10, pady=10)
        right_frame.pack_propagate(False)
        
        ctk.CTkLabel(right_frame, text="Inference Testing", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Backend selection
        ctk.CTkLabel(right_frame, text="Backend:", font=("Arial", 12)).pack(pady=(10, 5))
        self.test_backend_var = ctk.StringVar(value="tensorrt")
        ctk.CTkOptionMenu(
            right_frame, values=["tensorrt", "onnx", "pytorch"],
            variable=self.test_backend_var, width=200
        ).pack(pady=5)
        
        ctk.CTkLabel(right_frame, text="─" * 25).pack(pady=10)
        ctk.CTkLabel(right_frame, text="Image Detection", font=("Arial", 14, "bold")).pack(pady=5)
        
        ctk.CTkButton(right_frame, text="📂 Load Image", command=self.load_test_image, width=200).pack(pady=5)
        ctk.CTkButton(right_frame, text="🔍 Detect Objects", command=self.detect_objects, width=200).pack(pady=5)
        ctk.CTkButton(right_frame, text="🎨 Visualize", command=self.visualize_detections, width=200).pack(pady=5)
        
        ctk.CTkLabel(right_frame, text="─" * 25).pack(pady=10)
        ctk.CTkLabel(right_frame, text="Live Camera", font=("Arial", 14, "bold")).pack(pady=5)
        
        cam_frame = ctk.CTkFrame(right_frame)
        cam_frame.pack(pady=5)
        ctk.CTkLabel(cam_frame, text="Camera ID:").pack(side="left", padx=5)
        self.camera_id_var = ctk.StringVar(value="0")
        ctk.CTkEntry(cam_frame, textvariable=self.camera_id_var, width=50).pack(side="left", padx=5)
        
        self.start_cam_btn = ctk.CTkButton(
            right_frame, text="▶️ Start Camera", command=self.start_camera,
            fg_color="green", width=200
        )
        self.start_cam_btn.pack(pady=5)
        
        self.stop_cam_btn = ctk.CTkButton(
            right_frame, text="⏹️ Stop Camera", command=self.stop_camera,
            fg_color="red", width=200, state="disabled"
        )
        self.stop_cam_btn.pack(pady=5)
        
        # Detection results
        ctk.CTkLabel(right_frame, text="─" * 25).pack(pady=10)
        ctk.CTkLabel(right_frame, text="Detection Results", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.test_results = ctk.CTkTextbox(right_frame, width=300, height=200)
        self.test_results.pack(pady=5, padx=10, fill="both", expand=True)
        
        logger.debug("Test tab setup complete")
    
    def load_test_image(self):
        """Load image for testing - stops camera if running."""
        logger.debug("Loading test image")
        
        # CRITICAL FIX: Stop camera before loading image
        if self.camera_running:
            logger.info("Stopping camera before loading image")
            self.stop_camera()
            time.sleep(0.2)  # Give time for camera to stop
        
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.loaded_image_path = path
            logger.info(f"Loaded image: {path}")
            self._display_image(path)
            self.test_results.delete("1.0", "end")
            self.test_results.insert("end", f"Loaded: {Path(path).name}\n")
    
    def _display_image(self, path, detections=None):
        """Display image in test panel - Optimized."""
        try:
            logger.debug(f"Displaying image: {path}")
            img = Image.open(path)
            
            # Resize for display
            max_size = (640, 480)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB to avoid mode issues
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Clear previous image to free memory
            if self.displayed_image:
                try:
                    # Clear the label first to release reference
                    self.test_image_label.configure(image="", text="")
                    del self.displayed_image
                except:
                    pass
            
            # CRITICAL FIX: Keep reference and use configure properly
            self.displayed_image = CTkImage(light_image=img, dark_image=img, size=img.size)
            # Use after() to ensure GUI is ready
            self.test_image_label.configure(image=self.displayed_image, text="")
            # Force update
            self.test_image_label.update_idletasks()
            logger.debug("Image displayed successfully")
            
        except Exception as e:
            logger.error(f"Failed to display image: {e}", exc_info=True)
            self.test_image_label.configure(text=f"Error: {str(e)[:50]}")
            messagebox.showerror("Error", f"Failed to display image: {e}")
    
    def detect_objects(self):
        """Detect objects in loaded image."""
        if not self.loaded_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        logger.info("Running object detection on image")
        
        def run_detection():
            try:
                with open(self.loaded_image_path, "rb") as f:
                    files = {"file": (self.loaded_image_path, f, "image/png")}
                    params = {"backend": self.test_backend_var.get()}
                    logger.debug(f"Sending detection request with backend: {self.test_backend_var.get()}")
                    r = requests.post(f"{API_URL}/detect", files=files, params=params, timeout=30)
                
                result = r.json()
                logger.info(f"Detection result: {result.get('num_detections', 0)} detections")
                
                # Update stats
                inf_time = result.get("inference_time_ms", 0)
                num_det = result.get("num_detections", 0)
                self.after(0, lambda: self.test_stats_label.configure(
                    text=f"Inference: {inf_time:.1f}ms | Detections: {num_det}"
                ))
                
                # Show results
                self.after(0, lambda: self.test_results.delete("1.0", "end"))
                for det in result.get("detections", []):
                    text = f"{det['class_name']}: {det['confidence']:.2f}\n"
                    self.after(0, lambda t=text: self.test_results.insert("end", t))
                
            except Exception as e:
                logger.error(f"Detection failed: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {e}"))
        
        threading.Thread(target=run_detection, daemon=True).start()
    
    def visualize_detections(self):
        """Get visualized detection results."""
        if not self.loaded_image_path:
            return
        
        logger.info("Visualizing detections")
        
        def run_viz():
            try:
                with open(self.loaded_image_path, "rb") as f:
                    files = {"file": (self.loaded_image_path, f, "image/png")}
                    params = {"backend": self.test_backend_var.get()}
                    r = requests.post(f"{API_URL}/detect/visualize", files=files, params=params, timeout=30)
                
                from io import BytesIO
                img = Image.open(BytesIO(r.content))
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")
                
                def update_image():
                    if self.displayed_image:
                        del self.displayed_image
                    self.displayed_image = CTkImage(light_image=img, dark_image=img, size=img.size)
                    self.test_image_label.configure(image=self.displayed_image)
                    logger.debug("Visualization displayed")
                
                self.after(0, update_image)
                
            except Exception as e:
                logger.error(f"Visualization failed: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Error", f"Visualization failed: {e}"))
        
        threading.Thread(target=run_viz, daemon=True).start()
    
    def start_camera(self):
        """Start live camera detection - SAFE VERSION with queue."""
        logger.info("Starting camera")
        
        # CRITICAL FIX: Ensure camera is fully stopped before starting
        if self.camera_running:
            logger.warning("Camera already running, stopping first")
            self.stop_camera()
            time.sleep(0.3)
        
        self.camera_running = True
        self.start_cam_btn.configure(state="disabled")
        self.stop_cam_btn.configure(state="normal")
        
        # Clear loaded image to prevent conflicts
        self.loaded_image_path = None
        
        # Create frame queue for thread-safe communication
        import queue
        self.frame_queue = queue.Queue(maxsize=2)
        
        def camera_loop():
            cap = None
            try:
                cam_id = int(self.camera_id_var.get())
            except:
                cam_id = 0
            
            logger.info(f"Opening camera {cam_id}")
            cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {cam_id}")
                self.after(0, lambda: messagebox.showerror("Error", f"Failed to open camera {cam_id}"))
                self.camera_running = False
                self.after(0, lambda: self.start_cam_btn.configure(state="normal"))
                self.after(0, lambda: self.stop_cam_btn.configure(state="disabled"))
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
                            params = {"backend": self.test_backend_var.get()}
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
            
            self.after(0, lambda: self.start_cam_btn.configure(state="normal"))
            self.after(0, lambda: self.stop_cam_btn.configure(state="disabled"))
        
        def update_from_queue():
            """Update GUI from queue - runs in main thread."""
            if not self.camera_running:
                return
            
            try:
                # Get frame from queue (non-blocking)
                frame_rgb, fps, det_count, inf_time = self.frame_queue.get_nowait()
                
                # Create PIL Image in main thread
                img_pil = Image.fromarray(frame_rgb)
                img_pil.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                # CRITICAL FIX: Clear old image reference first
                if self.displayed_image:
                    try:
                        # Set label to empty first to release reference
                        self.test_image_label.configure(image="")
                        # Delete old image
                        del self.displayed_image
                        self.displayed_image = None
                    except:
                        pass
                
                # Create NEW CTkImage in main thread
                ctk_img = CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
                
                # Update label with new image
                self.test_image_label.configure(image=ctk_img, text="")
                
                # Keep reference to prevent garbage collection
                self.displayed_image = ctk_img
                
                # Update stats
                self.test_stats_label.configure(
                    text=f"FPS: {fps} | Inference: {inf_time:.1f}ms | Detections: {det_count}"
                )
                
            except Exception as e:
                # Queue empty or other error
                if str(e) and "Empty" not in str(e):
                    logger.debug(f"Queue error: {e}")
                pass
            
            # Schedule next update (every 33ms = ~30 FPS)
            if self.camera_running:
                self.after(33, update_from_queue)
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=camera_loop, daemon=True)
        self.camera_thread.start()
        logger.info("Camera thread started")
        
        # Start GUI update loop in main thread
        self.after(100, update_from_queue)
    
    def stop_camera(self):
        """Stop live camera - IMPROVED cleanup."""
        logger.info("Stopping camera")
        self.camera_running = False
        
        # Clear queue if it exists
        if hasattr(self, 'frame_queue'):
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except:
                pass
        
        # Wait for thread to finish
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
            logger.info("Waiting for camera thread to finish")
            self.camera_thread.join(timeout=2.0)
            if self.camera_thread.is_alive():
                logger.warning("Camera thread did not finish in time")
        
        # Clear image
        try:
            self.test_image_label.configure(image=None, text="Camera stopped")
        except:
            pass
        
        if self.displayed_image:
            try:
                del self.displayed_image
            except:
                pass
            self.displayed_image = None
        
        # Re-enable buttons
        self.start_cam_btn.configure(state="normal")
        self.stop_cam_btn.configure(state="disabled")
        
        logger.info("Camera stopped successfully")
    
    # =========================================================================
    # MONITORING TAB
    # =========================================================================
    def _setup_monitoring_tab(self):
        """Setup monitoring/dashboard tab with TensorBoard and WandB integration."""
        logger.debug("Setting up Monitoring tab")
        
        main_frame = ctk.CTkFrame(self.tab_monitoring)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title with integration indicators
        title_frame = ctk.CTkFrame(main_frame)
        title_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(title_frame, text="System Monitoring & Training Logs", font=("Arial", 20, "bold")).pack(side="left", padx=10)
        
        # Stats cards
        cards_frame = ctk.CTkFrame(main_frame)
        cards_frame.pack(fill="x", pady=20, padx=20)
        
        # API Stats card
        api_card = ctk.CTkFrame(cards_frame)
        api_card.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(api_card, text="API Statistics", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.api_requests_label = ctk.CTkLabel(api_card, text="Total Requests: --", font=("Arial", 12))
        self.api_requests_label.pack(pady=5)
        self.api_latency_label = ctk.CTkLabel(api_card, text="Avg Latency: -- ms", font=("Arial", 12))
        self.api_latency_label.pack(pady=5)
        self.api_throughput_label = ctk.CTkLabel(api_card, text="Throughput: -- FPS", font=("Arial", 12))
        self.api_throughput_label.pack(pady=5)
        
        # GPU Stats card
        gpu_card = ctk.CTkFrame(cards_frame)
        gpu_card.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(gpu_card, text="GPU Statistics", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.gpu_util_label = ctk.CTkLabel(gpu_card, text="Utilization: --%", font=("Arial", 12))
        self.gpu_util_label.pack(pady=5)
        self.gpu_mem_label = ctk.CTkLabel(gpu_card, text="Memory: -- MB", font=("Arial", 12))
        self.gpu_mem_label.pack(pady=5)
        self.gpu_temp_label = ctk.CTkLabel(gpu_card, text="Temperature: --°C", font=("Arial", 12))
        self.gpu_temp_label.pack(pady=5)
        
        # Model Stats card
        model_card = ctk.CTkFrame(cards_frame)
        model_card.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(model_card, text="Model Info", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.model_backend_label = ctk.CTkLabel(model_card, text="Backend: --", font=("Arial", 12))
        self.model_backend_label.pack(pady=5)
        self.model_size_label = ctk.CTkLabel(model_card, text="Model Size: --", font=("Arial", 12))
        self.model_size_label.pack(pady=5)
        self.model_classes_label = ctk.CTkLabel(model_card, text="Classes: --", font=("Arial", 12))
        self.model_classes_label.pack(pady=5)
        
        # TensorBoard & WandB Section
        monitoring_frame = ctk.CTkFrame(main_frame)
        monitoring_frame.pack(fill="x", pady=20, padx=20)
        
        ctk.CTkLabel(monitoring_frame, text="Training Visualization Tools", font=("Arial", 16, "bold")).pack(pady=10)
        
        # TensorBoard controls
        tb_frame = ctk.CTkFrame(monitoring_frame)
        tb_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(tb_frame, text="📊 TensorBoard", font=("Arial", 14, "bold")).pack(side="left", padx=10)
        
        self.tensorboard_status_label = ctk.CTkLabel(tb_frame, text="⚪ Not Running", font=("Arial", 12))
        self.tensorboard_status_label.pack(side="left", padx=10)
        
        self.tensorboard_port_var = ctk.StringVar(value="6006")
        ctk.CTkLabel(tb_frame, text="Port:").pack(side="left", padx=5)
        ctk.CTkEntry(tb_frame, textvariable=self.tensorboard_port_var, width=80).pack(side="left", padx=5)
        
        self.start_tensorboard_btn = ctk.CTkButton(
            tb_frame, text="▶️ Start", command=self.start_tensorboard,
            fg_color="green", width=100
        )
        self.start_tensorboard_btn.pack(side="left", padx=5)
        
        self.stop_tensorboard_btn = ctk.CTkButton(
            tb_frame, text="⏹️ Stop", command=self.stop_tensorboard,
            fg_color="red", width=100, state="disabled"
        )
        self.stop_tensorboard_btn.pack(side="left", padx=5)
        
        ctk.CTkButton(tb_frame, text="🌐 Open in Browser", command=self.open_tensorboard, width=150).pack(side="left", padx=5)
        
        # WandB controls
        wandb_frame = ctk.CTkFrame(monitoring_frame)
        wandb_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(wandb_frame, text="🏃 Weights & Biases", font=("Arial", 14, "bold")).pack(side="left", padx=10)
        
        self.wandb_status_label = ctk.CTkLabel(wandb_frame, text="⚪ Not Configured", font=("Arial", 12))
        self.wandb_status_label.pack(side="left", padx=10)
        
        ctk.CTkButton(wandb_frame, text="🔑 Configure API Key", command=self.configure_wandb, width=150).pack(side="left", padx=5)
        ctk.CTkButton(wandb_frame, text="🌐 Open Dashboard", command=self.open_wandb_dashboard, width=150).pack(side="left", padx=5)
        ctk.CTkButton(wandb_frame, text="📊 View Latest Run", command=self.view_latest_wandb_run, width=150).pack(side="left", padx=5)
        
        # Controls
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", pady=10, padx=20)
        
        ctk.CTkButton(controls_frame, text="🔄 Refresh Metrics", command=self.refresh_metrics, width=200).pack(side="left", padx=10)
        
        self.auto_refresh_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(controls_frame, text="Auto-refresh (5s)", variable=self.auto_refresh_var, 
                       command=self.toggle_auto_refresh).pack(side="left", padx=10)
        
        ctk.CTkButton(controls_frame, text="📈 Open API Dashboard", command=self.open_dashboard, width=200).pack(side="right", padx=10)
        
        # Logs
        ctk.CTkLabel(main_frame, text="Recent System Logs", font=("Arial", 14, "bold")).pack(pady=10)
        self.monitor_log = ctk.CTkTextbox(main_frame, width=1000, height=200)
        self.monitor_log.pack(pady=10, padx=20, fill="both", expand=True)
        
        # TensorBoard state
        self.tensorboard_process = None
        
        # Check WandB configuration
        self.check_wandb_config()
        
        self.auto_refresh_running = False
        logger.debug("Monitoring tab setup complete with TensorBoard/WandB integration")
    
    def refresh_metrics(self):
        """Refresh monitoring metrics."""
        logger.debug("Refreshing metrics")
        try:
            r = requests.get(f"{API_URL}/metrics", timeout=5)
            if r.status_code == 200:
                data = r.json()
                
                self.api_requests_label.configure(text=f"Total Requests: {data.get('total_requests', 0)}")
                self.api_latency_label.configure(text=f"Avg Latency: {data.get('avg_latency_ms', 0):.2f} ms")
                self.api_throughput_label.configure(text=f"Throughput: {data.get('throughput_fps', 0):.1f} FPS")
                
                self.gpu_util_label.configure(text=f"Utilization: {data.get('gpu_utilization_pct', 'N/A')}%")
                self.gpu_mem_label.configure(text=f"Memory: {data.get('gpu_memory_used_mb', 'N/A')} MB")
            
            # Get health for model info
            r2 = requests.get(f"{API_URL}/health", timeout=5)
            if r2.status_code == 200:
                health = r2.json()
                self.model_backend_label.configure(text=f"Backend: {health.get('backend', 'N/A')}")
                
        except Exception as e:
            logger.warning(f"Refresh failed: {e}")
            self.monitor_log.insert("end", f"Refresh failed: {e}\n")
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh."""
        if self.auto_refresh_var.get():
            logger.info("Auto-refresh enabled")
            self.auto_refresh_running = True
            self._auto_refresh_loop()
        else:
            logger.info("Auto-refresh disabled")
            self.auto_refresh_running = False
    
    def _auto_refresh_loop(self):
        """Auto-refresh loop."""
        if self.auto_refresh_running:
            self.refresh_metrics()
            self.after(5000, self._auto_refresh_loop)
    
    def open_dashboard(self):
        """Open API dashboard in browser."""
        import webbrowser
        url = f"{API_URL}/dashboard/combined"
        logger.info(f"Opening dashboard: {url}")
        webbrowser.open(url)
    
    # =========================================================================
    # TensorBoard & WandB Methods
    # =========================================================================
    
    def start_tensorboard(self):
        """Start TensorBoard server."""
        try:
            port = self.tensorboard_port_var.get()
            logdir = PROJECT_ROOT / "training" / "logs"
            
            if not logdir.exists():
                messagebox.showwarning("No Logs", f"Training logs directory not found: {logdir}")
                return
            
            logger.info(f"Starting TensorBoard on port {port}")
            self.tensorboard_process = subprocess.Popen(
                ["tensorboard", "--logdir", str(logdir), "--port", port, "--bind_all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT)
            )
            
            self.tensorboard_status_label.configure(text=f"🟢 Running on :{port}")
            self.start_tensorboard_btn.configure(state="disabled")
            self.stop_tensorboard_btn.configure(state="normal")
            
            self.monitor_log.insert("end", f"✅ TensorBoard started on port {port}\n")
            self.monitor_log.insert("end", f"   Open: http://localhost:{port}\n")
            messagebox.showinfo("TensorBoard Started", f"TensorBoard is running on http://localhost:{port}")
            
        except FileNotFoundError:
            messagebox.showerror("TensorBoard Not Found", 
                               "TensorBoard not installed. Install with:\npip install tensorboard")
        except Exception as e:
            logger.error(f"Failed to start TensorBoard: {e}")
            messagebox.showerror("Error", f"Failed to start TensorBoard:\n{str(e)}")
    
    def stop_tensorboard(self):
        """Stop TensorBoard server."""
        if self.tensorboard_process:
            try:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)
                self.tensorboard_process = None
                
                self.tensorboard_status_label.configure(text="⚪ Not Running")
                self.start_tensorboard_btn.configure(state="normal")
                self.stop_tensorboard_btn.configure(state="disabled")
                
                self.monitor_log.insert("end", "⏹️ TensorBoard stopped\n")
                logger.info("TensorBoard stopped")
                
            except Exception as e:
                logger.error(f"Error stopping TensorBoard: {e}")
                messagebox.showerror("Error", f"Error stopping TensorBoard:\n{str(e)}")
    
    def open_tensorboard(self):
        """Open TensorBoard in browser."""
        port = self.tensorboard_port_var.get()
        url = f"http://localhost:{port}"
        
        import webbrowser
        logger.info(f"Opening TensorBoard: {url}")
        
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open browser:\n{str(e)}")
    
    def check_wandb_config(self):
        """Check if WandB is configured."""
        try:
            import wandb
            api_key = os.environ.get("WANDB_API_KEY")
            
            if api_key:
                # Try to verify the key
                try:
                    api = wandb.Api()
                    self.wandb_status_label.configure(text="🟢 Configured")
                    self.monitor_log.insert("end", "✅ WandB API key found and valid\n")
                except Exception:
                    self.wandb_status_label.configure(text="🟡 Key Invalid")
                    self.monitor_log.insert("end", "⚠️ WandB API key found but invalid\n")
            else:
                self.wandb_status_label.configure(text="⚪ Not Configured")
                
        except ImportError:
            self.wandb_status_label.configure(text="❌ Not Installed")
            self.monitor_log.insert("end", "❌ WandB not installed. Install with: pip install wandb\n")
    
    def configure_wandb(self):
        """Configure WandB API key."""
        dialog = ctk.CTkInputDialog(
            text="Enter your Weights & Biases API key:\n(Get it from https://wandb.ai/settings)",
            title="Configure WandB"
        )
        api_key = dialog.get_input()
        
        if api_key:
            try:
                # Save to environment
                os.environ["WANDB_API_KEY"] = api_key
                
                # Save to .env file
                env_file = PROJECT_ROOT / ".env"
                with open(env_file, "a") as f:
                    f.write(f"\nWANDB_API_KEY={api_key}\n")
                
                # Verify
                import wandb
                wandb.login(key=api_key)
                
                self.wandb_status_label.configure(text="🟢 Configured")
                self.monitor_log.insert("end", "✅ WandB API key configured successfully\n")
                messagebox.showinfo("Success", "WandB API key configured!")
                
            except Exception as e:
                logger.error(f"WandB configuration failed: {e}")
                messagebox.showerror("Error", f"Failed to configure WandB:\n{str(e)}")
    
    def open_wandb_dashboard(self):
        """Open WandB dashboard in browser."""
        import webbrowser
        url = "https://wandb.ai/home"
        logger.info(f"Opening WandB dashboard: {url}")
        webbrowser.open(url)
    
    def view_latest_wandb_run(self):
        """View latest WandB run in browser."""
        try:
            # Find latest run from wandb directory
            wandb_dir = PROJECT_ROOT / "wandb"
            if not wandb_dir.exists():
                messagebox.showinfo("No Runs", "No WandB runs found in this project.")
                return
            
            # Find latest run directory
            run_dirs = sorted([d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not run_dirs:
                messagebox.showinfo("No Runs", "No WandB runs found.")
                return
            
            latest_run = run_dirs[0]
            
            # Try to extract run URL from run directory
            import webbrowser
            # Default to project page
            url = "https://wandb.ai/home"
            
            self.monitor_log.insert("end", f"📊 Latest run: {latest_run.name}\n")
            webbrowser.open(url)
            
        except Exception as e:
            logger.error(f"Error viewing WandB run: {e}")
            messagebox.showerror("Error", f"Error viewing WandB run:\n{str(e)}")
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    def on_closing(self):
        """Handle window close - IMPROVED cleanup."""
        logger.info("Application closing, cleaning up...")
        
        # Stop camera first
        self.camera_running = False
        self.auto_refresh_running = False
        
        # Wait for camera thread
        if self.camera_thread and self.camera_thread.is_alive():
            logger.info("Waiting for camera thread to finish")
            self.camera_thread.join(timeout=2.0)
        
        # Stop TensorBoard if running
        if hasattr(self, 'tensorboard_process') and self.tensorboard_process:
            logger.info("Terminating TensorBoard")
            self.tensorboard_process.terminate()
        
        # Stop processes
        if self.api_process:
            logger.info("Terminating API process")
            self.api_process.terminate()
        if self.training_process:
            logger.info("Terminating training process")
            self.training_process.terminate()
        
        time.sleep(0.3)
        logger.info("Application closed")
        self.destroy()


def main():
    """Main entry point."""
    logger.info("Starting Edge AI Manager v2")
    try:
        app = EdgeAIManager()
        app.mainloop()
    except Exception as e:
        logger.error(f"Application crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
