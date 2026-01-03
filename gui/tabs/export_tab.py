"""
Export Tab - Model Export and Optimization
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import logging
import subprocess
import sys
from pathlib import Path

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ExportTab(BaseTab):
    """Model Export and Optimization Tab"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
    
    def setup(self):
        """Setup export tab UI"""
        logger.debug("Setting up Export tab")
        
        main_frame = ctk.CTkFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(main_frame, text="📦 Model Export & Optimization", 
                    font=("Arial", 20, "bold")).pack(pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(main_frame)
        model_frame.pack(pady=20, fill="x", padx=50)
        
        ctk.CTkLabel(model_frame, text="Select Model:", font=("Arial", 14, "bold")).pack(pady=10)
        
        select_frame = ctk.CTkFrame(model_frame)
        select_frame.pack(pady=10)
        
        self.widgets['model_var'] = ctk.StringVar(value="models/latest.pt")
        self.widgets['model_entry'] = ctk.CTkEntry(
            select_frame, 
            textvariable=self.widgets['model_var'], 
            width=400, 
            state="readonly"
        )
        self.widgets['model_entry'].pack(side="left", padx=5)
        
        ctk.CTkButton(
            select_frame, 
            text="📂 Browse", 
            command=self._browse_model,
            width=100
        ).pack(side="left")
        
        # Export options in two columns
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(pady=20, fill="both", expand=True, padx=50)
        
        # Left: ONNX Export
        onnx_frame = ctk.CTkFrame(options_frame)
        onnx_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(onnx_frame, text="🔷 ONNX Export", 
                    font=("Arial", 16, "bold")).pack(pady=15)
        
        ctk.CTkLabel(onnx_frame, text="Export to ONNX format\nfor cross-platform inference",
                    font=("Arial", 11), text_color="gray").pack(pady=5)
        
        ctk.CTkButton(
            onnx_frame, 
            text="📦 Export to ONNX", 
            command=self._export_onnx,
            width=200, 
            height=40,
            fg_color="#2B8A3E"
        ).pack(pady=20)
        
        # Right: TensorRT Engine
        trt_frame = ctk.CTkFrame(options_frame)
        trt_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(trt_frame, text="⚡ TensorRT Engine", 
                    font=("Arial", 16, "bold")).pack(pady=15)
        
        ctk.CTkLabel(trt_frame, text="Build optimized engine\nfor NVIDIA GPUs",
                    font=("Arial", 11), text_color="gray").pack(pady=5)
        
        # Precision selector
        precision_frame = ctk.CTkFrame(trt_frame)
        precision_frame.pack(pady=10)
        
        ctk.CTkLabel(precision_frame, text="Precision:", 
                    font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        self.widgets['precision_var'] = ctk.StringVar(value="fp16")
        ctk.CTkOptionMenu(
            precision_frame,
            values=["fp32", "fp16", "int8"],
            variable=self.widgets['precision_var'],
            width=100
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            trt_frame, 
            text="⚡ Build Engine", 
            command=self._build_tensorrt,
            width=200, 
            height=40,
            fg_color="#E67700"
        ).pack(pady=20)
        
        # Status/Log area
        status_frame = ctk.CTkFrame(main_frame)
        status_frame.pack(fill="both", expand=True, padx=50, pady=10)
        
        ctk.CTkLabel(status_frame, text="Export Log", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.widgets['log'] = ctk.CTkTextbox(status_frame, width=800, height=200)
        self.widgets['log'].pack(fill="both", expand=True, padx=10, pady=10)
        self.widgets['log'].insert("end", "💡 Ready to export models\n\n")
        self.widgets['log'].insert("end", "Select a model and choose export format:\n")
        self.widgets['log'].insert("end", "• ONNX: Universal format for deployment\n")
        self.widgets['log'].insert("end", "• TensorRT: Optimized for NVIDIA GPUs\n")
        
        logger.debug("Export tab setup complete")
    
    def _browse_model(self):
        """Browse for model file"""
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")],
            initialdir=PROJECT_ROOT / "models"
        )
        if path:
            self.widgets['model_var'].set(path)
            self._log(f"Selected model: {Path(path).name}\n")
    
    def _export_onnx(self):
        """Export model to ONNX format"""
        model_path = self.widgets['model_var'].get()
        
        if not Path(model_path).exists():
            messagebox.showerror("Error", "Model file not found!")
            return
        
        self._log(f"\n{'='*60}\n")
        self._log(f"Exporting to ONNX: {Path(model_path).name}\n")
        self._log("This may take a few minutes...\n\n")
        
        try:
            script = PROJECT_ROOT / "optimization" / "export_to_onnx.py"
            cmd = [sys.executable, str(script), "--model", model_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                self._log("✅ ONNX export successful!\n")
                self._log(result.stdout)
                messagebox.showinfo("Success", "Model exported to ONNX successfully!")
            else:
                self._log("❌ ONNX export failed!\n")
                self._log(result.stderr)
                messagebox.showerror("Error", f"ONNX export failed:\n{result.stderr[:200]}")
                
        except Exception as e:
            logger.error(f"ONNX export error: {e}", exc_info=True)
            self._log(f"❌ Error: {e}\n")
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _build_tensorrt(self):
        """Build TensorRT engine"""
        model_path = self.widgets['model_var'].get()
        
        if not Path(model_path).exists():
            messagebox.showerror("Error", "Model file not found!")
            return
        
        precision = self.widgets['precision_var'].get()
        
        self._log(f"\n{'='*60}\n")
        self._log(f"Building TensorRT engine: {Path(model_path).name}\n")
        self._log(f"Precision: {precision}\n")
        self._log("This may take several minutes...\n\n")
        
        try:
            script = PROJECT_ROOT / "optimization" / "build_trt_engine.py"
            cmd = [
                sys.executable, str(script),
                "--model", model_path,
                "--precision", precision
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                self._log("✅ TensorRT engine built successfully!\n")
                self._log(result.stdout)
                messagebox.showinfo("Success", "TensorRT engine built successfully!")
            else:
                self._log("❌ TensorRT build failed!\n")
                self._log(result.stderr)
                messagebox.showerror("Error", f"TensorRT build failed:\n{result.stderr[:200]}")
                
        except Exception as e:
            logger.error(f"TensorRT build error: {e}", exc_info=True)
            self._log(f"❌ Error: {e}\n")
            messagebox.showerror("Error", f"Build failed: {e}")
    
    def _log(self, text):
        """Append text to log"""
        self.widgets['log'].insert("end", text)
        self.widgets['log'].see("end")
