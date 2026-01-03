"""
Monitoring Tab - System Monitoring, TensorBoard, Weights & Biases
"""

import customtkinter as ctk
from tkinter import messagebox
import logging
import subprocess
import webbrowser
from pathlib import Path

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class MonitoringTab(BaseTab):
    """System Monitoring and Dashboard Tab"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.tensorboard_process = None
    
    def setup(self):
        """Setup monitoring tab UI"""
        logger.debug("Setting up Monitoring tab")
        
        main_frame = ctk.CTkFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(main_frame, text="📊 Monitoring & Dashboards", 
                    font=("Arial", 20, "bold")).pack(pady=10)
        
        # Two columns
        columns_frame = ctk.CTkFrame(main_frame)
        columns_frame.pack(fill="both", expand=True, pady=20)
        
        # Left: TensorBoard
        tb_frame = ctk.CTkFrame(columns_frame)
        tb_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(tb_frame, text="📈 TensorBoard", 
                    font=("Arial", 16, "bold")).pack(pady=15)
        
        ctk.CTkLabel(tb_frame, text="Visualize training metrics,\nmodel graphs, and more",
                    font=("Arial", 11), text_color="gray").pack(pady=5)
        
        self.widgets['tb_status'] = ctk.CTkLabel(
            tb_frame,
            text="Status: ⚪ Stopped",
            font=("Arial", 12)
        )
        self.widgets['tb_status'].pack(pady=10)
        
        ctk.CTkButton(
            tb_frame,
            text="▶️ Start TensorBoard",
            command=self._start_tensorboard,
            width=200,
            height=40,
            fg_color="#2B8A3E"
        ).pack(pady=5)
        
        ctk.CTkButton(
            tb_frame,
            text="⏹️ Stop TensorBoard",
            command=self._stop_tensorboard,
            width=200,
            height=40,
            fg_color="#C92A2A"
        ).pack(pady=5)
        
        ctk.CTkButton(
            tb_frame,
            text="🌐 Open in Browser",
            command=lambda: webbrowser.open("http://localhost:6006"),
            width=200,
            height=35
        ).pack(pady=5)
        
        # Right: Weights & Biases
        wandb_frame = ctk.CTkFrame(columns_frame)
        wandb_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(wandb_frame, text="🔷 Weights & Biases", 
                    font=("Arial", 16, "bold")).pack(pady=15)
        
        ctk.CTkLabel(wandb_frame, text="Track experiments,\ncompare runs, collaborate",
                    font=("Arial", 11), text_color="gray").pack(pady=5)
        
        ctk.CTkButton(
            wandb_frame,
            text="🌐 Open W&B Dashboard",
            command=lambda: webbrowser.open("https://wandb.ai"),
            width=200,
            height=40,
            fg_color="#E67700"
        ).pack(pady=20)
        
        ctk.CTkButton(
            wandb_frame,
            text="🔑 Configure API Key",
            command=self._configure_wandb,
            width=200,
            height=35
        ).pack(pady=5)
        
        # Bottom: System Info
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="both", expand=True, pady=20)
        
        ctk.CTkLabel(info_frame, text="System Information", 
                    font=("Arial", 14, "bold")).pack(pady=10)
        
        self.widgets['info_text'] = ctk.CTkTextbox(info_frame, width=800, height=200)
        self.widgets['info_text'].pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load system info
        self._load_system_info()
        
        logger.debug("Monitoring tab setup complete")
    
    def _start_tensorboard(self):
        """Start TensorBoard server"""
        if self.tensorboard_process:
            messagebox.showinfo("Info", "TensorBoard is already running")
            return
        
        try:
            logdir = PROJECT_ROOT / "training" / "logs"
            logdir.mkdir(parents=True, exist_ok=True)
            
            self.tensorboard_process = subprocess.Popen(
                ["tensorboard", "--logdir", str(logdir), "--port", "6006"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.widgets['tb_status'].configure(text="Status: 🟢 Running")
            messagebox.showinfo("Success", 
                "TensorBoard started!\n\nAccess at: http://localhost:6006")
            logger.info("TensorBoard started")
            
        except Exception as e:
            logger.error(f"Failed to start TensorBoard: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start TensorBoard:\n{e}")
    
    def _stop_tensorboard(self):
        """Stop TensorBoard server"""
        if not self.tensorboard_process:
            messagebox.showinfo("Info", "TensorBoard is not running")
            return
        
        try:
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait(timeout=5)
            self.tensorboard_process = None
            
            self.widgets['tb_status'].configure(text="Status: ⚪ Stopped")
            messagebox.showinfo("Info", "TensorBoard stopped")
            logger.info("TensorBoard stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop TensorBoard: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to stop TensorBoard:\n{e}")
    
    def _configure_wandb(self):
        """Configure W&B API key"""
        from tkinter import simpledialog
        
        api_key = simpledialog.askstring(
            "W&B API Key",
            "Enter your Weights & Biases API key:\n\n(Get it from https://wandb.ai/authorize)",
            show="*"
        )
        
        if api_key:
            try:
                result = subprocess.run(
                    ["wandb", "login", api_key],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    messagebox.showinfo("Success", "W&B API key configured successfully!")
                else:
                    messagebox.showerror("Error", f"Failed to configure W&B:\n{result.stderr}")
                    
            except Exception as e:
                logger.error(f"W&B config error: {e}", exc_info=True)
                messagebox.showerror("Error", f"Failed to configure W&B:\n{e}")
    
    def _load_system_info(self):
        """Load and display system information"""
        try:
            import platform
            import psutil
            
            info = []
            info.append("=" * 60)
            info.append("SYSTEM INFORMATION")
            info.append("=" * 60)
            info.append("")
            
            # Platform
            info.append(f"OS: {platform.system()} {platform.release()}")
            info.append(f"Python: {platform.python_version()}")
            info.append("")
            
            # CPU
            info.append(f"CPU: {platform.processor()}")
            info.append(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
            info.append(f"CPU Usage: {psutil.cpu_percent()}%")
            info.append("")
            
            # Memory
            mem = psutil.virtual_memory()
            info.append(f"RAM: {mem.total / (1024**3):.1f} GB total")
            info.append(f"RAM Usage: {mem.percent}% ({mem.used / (1024**3):.1f} GB used)")
            info.append("")
            
            # GPU (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    info.append(f"GPU: {torch.cuda.get_device_name(0)}")
                    info.append(f"CUDA Version: {torch.version.cuda}")
                    info.append(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
                else:
                    info.append("GPU: Not available")
            except:
                info.append("GPU: Unknown")
            
            info.append("")
            info.append("=" * 60)
            
            self.widgets['info_text'].delete("1.0", "end")
            self.widgets['info_text'].insert("end", "\n".join(info))
            
        except Exception as e:
            logger.error(f"Failed to load system info: {e}", exc_info=True)
            self.widgets['info_text'].insert("end", f"Error loading system info: {e}")
    
    def cleanup(self):
        """Cleanup when tab is closed"""
        if self.tensorboard_process:
            try:
                self.tensorboard_process.terminate()
            except:
                pass
