"""
Training Tab - Model Training Configuration and Execution
"""

import customtkinter as ctk
from tkinter import messagebox
import logging
import threading
import subprocess
import sys
from pathlib import Path

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TrainingTab(BaseTab):
    """Training Tab - Configure and run model training"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.training_process = None
    
    def setup(self):
        """Setup training tab UI"""
        logger.debug("Setting up Training tab")
        
        # Get dataset manager from app (shared instance)
        self.dataset_manager = self.app.dataset_manager
        
        # Create main container
        main_container = ctk.CTkFrame(self.parent)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # ========== Dataset Selection ==========
        dataset_frame = ctk.CTkFrame(main_container)
        dataset_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(dataset_frame, text="📁 Select Dataset", 
                    font=("Arial", 16, "bold")).pack(pady=5)
        
        dataset_row = ctk.CTkFrame(dataset_frame)
        dataset_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(dataset_row, text="Dataset:", font=("Arial", 12, "bold")).pack(side="left", padx=5)
        
        self.widgets['dataset_var'] = ctk.StringVar(value="No datasets")
        self.widgets['dataset_dropdown'] = ctk.CTkOptionMenu(
            dataset_row,
            variable=self.widgets['dataset_var'],
            values=["No datasets"],
            width=250,
            command=self._on_dataset_selected
        )
        self.widgets['dataset_dropdown'].pack(side="left", padx=10)
        
        self.widgets['dataset_info'] = ctk.CTkLabel(
            dataset_row,
            text="No dataset - Go to 'Datasets' tab",
            font=("Arial", 11),
            text_color="gray"
        )
        self.widgets['dataset_info'].pack(side="left", padx=20)
        
        ctk.CTkButton(
            dataset_row, text="🔄 Refresh",
            command=self.refresh, width=100
        ).pack(side="left", padx=5)
        
        # ========== Training Configuration ==========
        config_frame = ctk.CTkFrame(main_container)
        config_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left: Configuration
        left_frame = ctk.CTkFrame(config_frame, width=350)
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        ctk.CTkLabel(left_frame, text="⚙️ Configuration", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        scroll = ctk.CTkScrollableFrame(left_frame, width=320, height=500)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Model
        ctk.CTkLabel(scroll, text="Base Model:", font=("Arial", 12, "bold")).pack(pady=(5,2), anchor="w")
        self.widgets['model_var'] = ctk.StringVar(value="yolov8n.pt")
        ctk.CTkOptionMenu(
            scroll,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", 
                   "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
            variable=self.widgets['model_var'],
            width=280
        ).pack(pady=2)
        
        ctk.CTkLabel(scroll, text="─" * 35).pack(pady=5)
        
        # Parameters
        params = [
            ("Epochs:", "epochs_var", "100"),
            ("Batch Size:", "batch_var", "16"),
            ("Image Size:", "imgsz_var", "640"),
            ("Learning Rate:", "lr_var", "0.01")
        ]
        
        for label, var_name, default in params:
            ctk.CTkLabel(scroll, text=label, font=("Arial", 11)).pack(pady=(5,2), anchor="w")
            self.widgets[var_name] = ctk.StringVar(value=default)
            ctk.CTkEntry(scroll, textvariable=self.widgets[var_name], width=280).pack(pady=2)
        
        # Augmentation
        ctk.CTkLabel(scroll, text="Augmentation:", font=("Arial", 11)).pack(pady=(5,2), anchor="w")
        self.widgets['augment_var'] = ctk.StringVar(value="medium")
        ctk.CTkOptionMenu(
            scroll,
            values=["light", "medium", "strong"],
            variable=self.widgets['augment_var'],
            width=280
        ).pack(pady=2)
        
        ctk.CTkLabel(scroll, text="─" * 35).pack(pady=5)
        
        # Options
        self.widgets['wandb_var'] = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(scroll, text="Weights & Biases", 
                       variable=self.widgets['wandb_var']).pack(pady=3, anchor="w")
        
        self.widgets['amp_var'] = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(scroll, text="Mixed Precision", 
                       variable=self.widgets['amp_var']).pack(pady=3, anchor="w")
        
        self.widgets['save_period_var'] = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(scroll, text="Save every 5 epochs", 
                       variable=self.widgets['save_period_var']).pack(pady=3, anchor="w")
        
        # Buttons
        ctk.CTkLabel(left_frame, text="─" * 35).pack(pady=5)
        
        self.widgets['start_btn'] = ctk.CTkButton(
            left_frame, text="🚀 Start Training",
            command=self._start_training,
            fg_color="green", width=320, height=40,
            font=("Arial", 14, "bold")
        )
        self.widgets['start_btn'].pack(pady=5)
        
        self.widgets['stop_btn'] = ctk.CTkButton(
            left_frame, text="⏹️ Stop Training",
            command=self._stop_training,
            fg_color="red", width=320, height=35,
            state="disabled"
        )
        self.widgets['stop_btn'].pack(pady=5)
        
        # Right: Progress & Logs
        right_frame = ctk.CTkFrame(config_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(right_frame, text="📊 Training Progress", 
                    font=("Arial", 14, "bold")).pack(pady=5)
        
        self.widgets['progress'] = ctk.CTkProgressBar(right_frame, width=700)
        self.widgets['progress'].pack(pady=10, padx=10)
        self.widgets['progress'].set(0)
        
        self.widgets['status_label'] = ctk.CTkLabel(
            right_frame, text="Ready", font=("Arial", 12)
        )
        self.widgets['status_label'].pack(pady=5)
        
        self.widgets['log'] = ctk.CTkTextbox(right_frame, width=800, height=550)
        self.widgets['log'].pack(pady=5, padx=10, fill="both", expand=True)
        self.widgets['log'].insert("end", "💡 Training Tips:\n\n")
        self.widgets['log'].insert("end", "1. Select dataset from 'Datasets' tab\n")
        self.widgets['log'].insert("end", "2. Validate dataset before training\n")
        self.widgets['log'].insert("end", "3. Choose batch size for your GPU\n")
        self.widgets['log'].insert("end", "4. Monitor progress in real-time\n\n")
        
        # Load datasets
        self.refresh()
        
        logger.debug("Training tab setup complete")
    
    def refresh(self):
        """Refresh dataset list"""
        datasets = self.dataset_manager.list_datasets()
        if datasets:
            self.widgets['dataset_dropdown'].configure(values=datasets)
            current = self.widgets['dataset_var'].get()
            if current == "No datasets" or current not in datasets:
                self.widgets['dataset_var'].set(datasets[0])
                self._on_dataset_selected(datasets[0])
        else:
            self.widgets['dataset_dropdown'].configure(values=["No datasets"])
            self.widgets['dataset_var'].set("No datasets")
            self.widgets['dataset_info'].configure(text="No dataset - Go to 'Datasets' tab")
    
    def _on_dataset_selected(self, name):
        """Handle dataset selection"""
        if name == "No datasets":
            return
        
        dataset = self.dataset_manager.get_dataset(name)
        if dataset:
            num_classes = len(dataset.get('class_names', []))
            self.widgets['dataset_info'].configure(
                text=f"✅ {name} - {num_classes} classes",
                text_color="green"
            )
    
    def _start_training(self):
        """Start training process"""
        dataset_name = self.widgets['dataset_var'].get()
        if dataset_name == "No datasets":
            messagebox.showerror("Error", "Select a dataset first!")
            return
        
        dataset = self.dataset_manager.get_dataset(dataset_name)
        if not dataset:
            messagebox.showerror("Error", "Dataset not found!")
            return
        
        # Validate parameters
        try:
            epochs = int(self.widgets['epochs_var'].get())
            batch = int(self.widgets['batch_var'].get())
            imgsz = int(self.widgets['imgsz_var'].get())
            lr = float(self.widgets['lr_var'].get())
            
            if epochs <= 0 or batch <= 0 or imgsz <= 0:
                raise ValueError("Values must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
            return
        
        def run():
            try:
                train_script = PROJECT_ROOT / "training" / "train.py"
                cmd = [
                    sys.executable, str(train_script),
                    "--model", self.widgets['model_var'].get(),
                    "--data", dataset["yaml_path"],
                    "--epochs", str(epochs),
                    "--batch", str(batch),
                    "--imgsz", str(imgsz),
                    "--lr", str(lr),
                    "--augment", self.widgets['augment_var'].get(),
                ]
                
                if not self.widgets['wandb_var'].get():
                    cmd.extend(["--project", "local"])
                if not self.widgets['amp_var'].get():
                    cmd.append("--amp=False")
                if self.widgets['save_period_var'].get():
                    cmd.extend(["--save-period", "5"])
                
                # Update UI
                self.app.after(0, lambda: self.widgets['start_btn'].configure(state="disabled"))
                self.app.after(0, lambda: self.widgets['stop_btn'].configure(state="normal"))
                self.app.after(0, lambda: self.widgets['status_label'].configure(text="Training..."))
                self.app.after(0, lambda: self._append_log(f"\n{'='*60}\n"))
                self.app.after(0, lambda: self._append_log(f"Training: {dataset_name}\n"))
                self.app.after(0, lambda: self._append_log(f"Command: {' '.join(cmd)}\n\n"))
                
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream output
                for line in self.training_process.stdout:
                    self.app.after(0, lambda l=line: self._append_log(l))
                    
                    # Update progress
                    if "Epoch" in line and "/" in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "Epoch" and i + 1 < len(parts):
                                    epoch_info = parts[i+1].split("/")
                                    if len(epoch_info) == 2:
                                        current = int(epoch_info[0])
                                        total = int(epoch_info[1])
                                        progress = current / total
                                        self.app.after(0, lambda p=progress: self.widgets['progress'].set(p))
                                        self.app.after(0, lambda c=current, t=total: 
                                                      self.widgets['status_label'].configure(
                                                          text=f"Epoch {c}/{t}"))
                        except:
                            pass
                
                self.training_process.wait()
                
                # Training complete
                self.app.after(0, lambda: self.widgets['start_btn'].configure(state="normal"))
                self.app.after(0, lambda: self.widgets['stop_btn'].configure(state="disabled"))
                self.app.after(0, lambda: self.widgets['status_label'].configure(text="Complete!"))
                self.app.after(0, lambda: self.widgets['progress'].set(1.0))
                self.app.after(0, lambda: self._append_log("\n✅ Training complete!\n"))
                
            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                self.app.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
                self.app.after(0, lambda: self.widgets['start_btn'].configure(state="normal"))
                self.app.after(0, lambda: self.widgets['stop_btn'].configure(state="disabled"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _stop_training(self):
        """Stop training process"""
        if self.training_process:
            self.training_process.terminate()
            self.widgets['status_label'].configure(text="Stopped")
            self.widgets['start_btn'].configure(state="normal")
            self.widgets['stop_btn'].configure(state="disabled")
            self._append_log("\n⛔ Training stopped by user\n")
    
    def _append_log(self, text):
        """Append text to log"""
        self.widgets['log'].insert("end", text)
        self.widgets['log'].see("end")
