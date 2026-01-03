"""
API Tab - API Server Control and Management
"""

import customtkinter as ctk
import subprocess
import sys
import threading
import requests
import json
from tkinter import messagebox
from pathlib import Path
import logging

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8002"


class APITab(BaseTab):
    """API Server Control and Management Tab"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.api_process = None
        
    def setup(self):
        """Setup API tab UI"""
        logger.debug("Setting up API tab")
        
        # Left panel - Controls
        left_frame = ctk.CTkFrame(self.parent)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(left_frame, text="API Server Control", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Server status
        self.widgets['status_label'] = ctk.CTkLabel(left_frame, text="Status: ⚪ Stopped", font=("Arial", 14))
        self.widgets['status_label'].pack(pady=10)
        
        # Backend selection
        ctk.CTkLabel(left_frame, text="Backend:", font=("Arial", 12)).pack(pady=(20, 5))
        self.widgets['backend_var'] = ctk.StringVar(value="tensorrt")
        self.widgets['backend_menu'] = ctk.CTkOptionMenu(
            left_frame,
            values=["tensorrt", "onnx", "pytorch"],
            variable=self.widgets['backend_var'],
            width=200
        )
        self.widgets['backend_menu'].pack(pady=5)
        
        # Port
        ctk.CTkLabel(left_frame, text="Port:", font=("Arial", 12)).pack(pady=(10, 5))
        self.widgets['port_var'] = ctk.StringVar(value="8002")
        self.widgets['port_entry'] = ctk.CTkEntry(left_frame, textvariable=self.widgets['port_var'], width=200)
        self.widgets['port_entry'].pack(pady=5)
        
        # Start/Stop buttons
        self.widgets['start_btn'] = ctk.CTkButton(
            left_frame, text="▶️ Start Server", command=self.start_server,
            fg_color="green", width=200
        )
        self.widgets['start_btn'].pack(pady=10)
        
        self.widgets['stop_btn'] = ctk.CTkButton(
            left_frame, text="⏹️ Stop Server", command=self.stop_server,
            fg_color="red", width=200, state="disabled"
        )
        self.widgets['stop_btn'].pack(pady=5)
        
        # Quick actions
        ctk.CTkLabel(left_frame, text="─" * 25).pack(pady=10)
        ctk.CTkLabel(left_frame, text="Quick Actions", font=("Arial", 14, "bold")).pack(pady=5)
        
        ctk.CTkButton(left_frame, text="💚 Health Check", command=self.check_health, width=200).pack(pady=5)
        ctk.CTkButton(left_frame, text="📊 Get Metrics", command=self.get_metrics, width=200).pack(pady=5)
        ctk.CTkButton(left_frame, text="🔄 Switch Backend", command=self.switch_backend, width=200).pack(pady=5)
        
        # Right panel - Logs
        right_frame = ctk.CTkFrame(self.parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(right_frame, text="Server Logs", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.widgets['log'] = ctk.CTkTextbox(right_frame, width=800, height=600)
        self.widgets['log'].pack(pady=10, padx=10, fill="both", expand=True)
        self.widgets['log'].insert("end", "API server logs will appear here...\n")
        
        logger.debug("API tab setup complete")
    
    def start_server(self):
        """Start the FastAPI server"""
        port = self.widgets['port_var'].get()
        PROJECT_ROOT = Path(self.app.workspace_root) if hasattr(self.app, 'workspace_root') else Path(__file__).parent.parent.parent
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
                self.app.after(0, lambda: self.widgets['status_label'].configure(text="Status: 🟢 Running"))
                self.app.after(0, lambda: self.widgets['start_btn'].configure(state="disabled"))
                self.app.after(0, lambda: self.widgets['stop_btn'].configure(state="normal"))
                
                for line in self.api_process.stdout:
                    self.app.after(0, lambda l=line: self._append_log(l))
                    
            except Exception as e:
                logger.error(f"Failed to start API server: {e}", exc_info=True)
                self.app.after(0, lambda: self._append_log(f"Error: {e}\n"))
        
        self._append_log(f"Starting API server on port {port}...\n")
        threading.Thread(target=run_server, daemon=True).start()
    
    def stop_server(self):
        """Stop the FastAPI server"""
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
                self.widgets['status_label'].configure(text="Status: ⚪ Stopped")
                self.widgets['start_btn'].configure(state="normal")
                self.widgets['stop_btn'].configure(state="disabled")
                self._append_log("Server stopped.\n")
        else:
            logger.warning("No API process to stop")
            self.widgets['status_label'].configure(text="Status: ⚪ Stopped")
            self.widgets['start_btn'].configure(state="normal")
            self.widgets['stop_btn'].configure(state="disabled")
    
    def check_health(self):
        """Check API health"""
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
        """Get API metrics"""
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
        """Switch API backend"""
        backend = self.widgets['backend_var'].get()
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
    
    def _append_log(self, text):
        """Append text to log"""
        self.widgets['log'].insert("end", text)
        self.widgets['log'].see("end")
        # Limit log size
        lines = int(self.widgets['log'].index('end-1c').split('.')[0])
        if lines > 1000:
            self.widgets['log'].delete('1.0', '500.0')
    
    def cleanup(self):
        """Cleanup when tab is closed"""
        if self.api_process:
            self.stop_server()
