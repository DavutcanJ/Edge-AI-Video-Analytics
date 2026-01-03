"""
Dataset Tab - Dataset Management and Validation
Manages datasets: add, validate, preview, remove
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import logging
from pathlib import Path

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class DatasetTab(BaseTab):
    """Dataset Management Tab"""
    
    def __init__(self, app, parent):
        super().__init__(app, parent)
    
    def setup(self):
        """Setup dataset tab UI"""
        logger.debug("Setting up Dataset tab")
        
        # Initialize dataset manager
        from dataset_manager import DatasetManager
        self.dataset_manager = DatasetManager()
        
        # Main container
        main_container = ctk.CTkFrame(self.parent)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(main_container, text="📁 Dataset Management", 
                    font=("Arial", 20, "bold")).pack(pady=10)
        
        # Controls
        controls_frame = ctk.CTkFrame(main_container)
        controls_frame.pack(fill="x", padx=20, pady=10)
        
        # Dataset dropdown
        ctk.CTkLabel(controls_frame, text="Select Dataset:", 
                    font=("Arial", 13, "bold")).pack(side="left", padx=10)
        
        self.widgets['dataset_var'] = ctk.StringVar(value="No datasets")
        self.widgets['dataset_dropdown'] = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.widgets['dataset_var'],
            values=["No datasets"],
            width=250,
            command=self.on_dataset_selected
        )
        self.widgets['dataset_dropdown'].pack(side="left", padx=10)
        
        # Buttons
        ctk.CTkButton(
            controls_frame, text="➕ Add New", 
            command=self.add_dataset, width=120, fg_color="#2D5F8D"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            controls_frame, text="🔍 Validate", 
            command=self.validate_dataset, width=100
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            controls_frame, text="👁️ Preview", 
            command=self.preview_dataset, width=100, fg_color="#2E8B57"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            controls_frame, text="🗑️ Remove", 
            command=self.remove_dataset, width=100, fg_color="#8B0000"
        ).pack(side="left", padx=5)
        
        # Dataset info panel
        info_container = ctk.CTkFrame(main_container)
        info_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(info_container, text="Dataset Information", 
                    font=("Arial", 16, "bold")).pack(pady=10)
        
        self.widgets['info_text'] = ctk.CTkTextbox(info_container, width=1100, height=500)
        self.widgets['info_text'].pack(fill="both", expand=True, padx=10, pady=10)
        self.widgets['info_text'].insert("end", "💡 Click '➕ Add New' to add your first dataset.\n\n")
        self.widgets['info_text'].insert("end", "Required structure:\n")
        self.widgets['info_text'].insert("end", "  your_dataset/\n")
        self.widgets['info_text'].insert("end", "    ├─ train/\n")
        self.widgets['info_text'].insert("end", "    │   ├─ images/\n")
        self.widgets['info_text'].insert("end", "    │   └─ labels/\n")
        self.widgets['info_text'].insert("end", "    └─ val/\n")
        self.widgets['info_text'].insert("end", "        ├─ images/\n")
        self.widgets['info_text'].insert("end", "        └─ labels/\n")
        
        # Load existing datasets
        self.refresh_datasets()
        
        logger.debug("Dataset tab setup complete")
    
    def refresh_datasets(self):
        """Refresh dataset list"""
        datasets = self.dataset_manager.list_datasets()
        if datasets:
            self.widgets['dataset_dropdown'].configure(values=datasets)
            current = self.widgets['dataset_var'].get()
            if current == "No datasets" or current not in datasets:
                self.widgets['dataset_var'].set(datasets[0])
                self.on_dataset_selected(datasets[0])
        else:
            self.widgets['dataset_dropdown'].configure(values=["No datasets"])
            self.widgets['dataset_var'].set("No datasets")
            self._show_info("No datasets available. Click '➕ Add New' to get started.")
    
    def on_dataset_selected(self, dataset_name):
        """Handle dataset selection"""
        if dataset_name == "No datasets":
            return
        
        summary = self.dataset_manager.get_dataset_summary(dataset_name)
        if summary:
            self._show_info(summary)
    
    def add_dataset(self):
        """Add new dataset - simple in-tab validation"""
        # Ask for name
        dialog = ctk.CTkInputDialog(text="Enter dataset name:", title="Add Dataset")
        dataset_name = dialog.get_input()
        
        if not dataset_name:
            return
        
        if dataset_name in self.dataset_manager.list_datasets():
            messagebox.showerror("Error", f"Dataset '{dataset_name}' already exists!")
            return
        
        # Browse folder
        path = filedialog.askdirectory(
            title="Select dataset ROOT folder (with train/ and val/ inside)"
        )
        if not path:
            return
        
        # Show progress in info panel
        self._show_info("⏳ Validating dataset, please wait...\n\n")
        
        def validate_and_add():
            try:
                from dataset_manager import DatasetValidator
                
                validator = DatasetValidator(path)
                is_valid, metadata = validator.validate()
                
                # Update UI
                def show_results():
                    self.widgets['info_text'].delete("1.0", "end")
                    
                    self.widgets['info_text'].insert("end", "="*70 + "\n")
                    self.widgets['info_text'].insert("end", "VALIDATION RESULTS\n")
                    self.widgets['info_text'].insert("end", "="*70 + "\n\n")
                    
                    self.widgets['info_text'].insert("end", f"📁 Dataset: {dataset_name}\n")
                    self.widgets['info_text'].insert("end", f"📂 Path: {path}\n\n")
                    
                    if not is_valid:
                        self.widgets['info_text'].insert("end", "❌ VALIDATION FAILED!\n\n")
                        for error in metadata.get("errors", []):
                            self.widgets['info_text'].insert("end", f"• {error}\n\n")
                        
                        self.widgets['info_text'].insert("end", "\n💡 Required structure:\n")
                        self.widgets['info_text'].insert("end", "   your_dataset/\n")
                        self.widgets['info_text'].insert("end", "     ├─ train/\n")
                        self.widgets['info_text'].insert("end", "     │   ├─ images/\n")
                        self.widgets['info_text'].insert("end", "     │   └─ labels/\n")
                        self.widgets['info_text'].insert("end", "     └─ val/\n")
                        self.widgets['info_text'].insert("end", "         ├─ images/\n")
                        self.widgets['info_text'].insert("end", "         └─ labels/\n")
                        messagebox.showerror("Validation Failed", "Dataset structure is invalid. Check the info panel for details.")
                    else:
                        # Show success
                        train_info = metadata.get("splits", {}).get("train", {})
                        val_info = metadata.get("splits", {}).get("val", {})
                        
                        train_imgs = train_info.get("image_count", 0)
                        train_lbls = train_info.get("label_count", 0)
                        val_imgs = val_info.get("image_count", 0)
                        val_lbls = val_info.get("label_count", 0)
                        
                        self.widgets['info_text'].insert("end", "✅ VALIDATION SUCCESSFUL!\n\n")
                        self.widgets['info_text'].insert("end", f"📸 Images: {train_imgs + val_imgs} total\n")
                        self.widgets['info_text'].insert("end", f"   • Train: {train_imgs}\n")
                        self.widgets['info_text'].insert("end", f"   • Val:   {val_imgs}\n\n")
                        
                        self.widgets['info_text'].insert("end", f"🏷️  Labels: {train_lbls + val_lbls} total\n")
                        self.widgets['info_text'].insert("end", f"   • Train: {train_lbls}\n")
                        self.widgets['info_text'].insert("end", f"   • Val:   {val_lbls}\n\n")
                        
                        num_classes = metadata.get("classes", {}).get("count", 0)
                        class_ids = metadata.get("classes", {}).get("ids", [])
                        self.widgets['info_text'].insert("end", f"🎯 Classes: {num_classes}\n")
                        if class_ids:
                            self.widgets['info_text'].insert("end", f"   • IDs: {class_ids}\n\n")
                        
                        warnings = metadata.get("warnings", [])
                        if warnings:
                            self.widgets['info_text'].insert("end", f"⚠️  {len(warnings)} warnings (check validation)\n\n")
                        
                        self.widgets['info_text'].insert("end", "="*70 + "\n")
                        
                        # Ask for class names
                        self.app.after(500, lambda: self._ask_class_names(dataset_name, path, num_classes, class_ids))
                
                self.app.after(0, show_results)
                
            except Exception as e:
                logger.error(f"Validation error: {e}", exc_info=True)
                self.app.after(0, lambda: messagebox.showerror("Error", f"Validation failed: {e}"))
        
        threading.Thread(target=validate_and_add, daemon=True).start()
    
    def _ask_class_names(self, dataset_name, path, num_classes, class_ids):
        """Ask for class names in a simple dialog"""
        class_dialog = ctk.CTkToplevel(self.app)
        class_dialog.title("Configure Class Names")
        class_dialog.geometry("450x550")
        class_dialog.transient(self.app)
        class_dialog.grab_set()
        
        ctk.CTkLabel(
            class_dialog, 
            text=f"Enter names for {num_classes} classes:",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(
            class_dialog,
            text=f"Detected class IDs: {class_ids}",
            font=("Arial", 10)
        ).pack(pady=5)
        
        # Class entries
        class_entries = []
        scroll_frame = ctk.CTkScrollableFrame(class_dialog, width=400, height=350)
        scroll_frame.pack(pady=10)
        
        for i in range(num_classes):
            frame = ctk.CTkFrame(scroll_frame)
            frame.pack(fill="x", pady=3)
            ctk.CTkLabel(frame, text=f"Class {i}:", width=70).pack(side="left", padx=5)
            entry = ctk.CTkEntry(frame, width=280)
            entry.insert(0, f"class_{i}")
            entry.pack(side="left", padx=5)
            class_entries.append(entry)
        
        def save():
            class_names = [e.get().strip() for e in class_entries]
            if any(not name for name in class_names):
                messagebox.showwarning("Warning", "All class names must be filled!")
                return
            
            success, message = self.dataset_manager.add_dataset(dataset_name, path, class_names)
            
            if success:
                self.refresh_datasets()
                messagebox.showinfo("Success", message)
                class_dialog.destroy()
            else:
                messagebox.showerror("Error", message)
        
        ctk.CTkButton(
            class_dialog, text="✅ Save Dataset",
            command=save, fg_color="green", width=200, height=40
        ).pack(pady=15)
    
    def validate_dataset(self):
        """Re-validate selected dataset"""
        dataset_name = self.widgets['dataset_var'].get()
        if dataset_name == "No datasets":
            messagebox.showinfo("Info", "No dataset selected")
            return
        
        dataset = self.dataset_manager.get_dataset(dataset_name)
        if not dataset:
            messagebox.showerror("Error", "Dataset not found")
            return
        
        self._show_info(f"⏳ Re-validating {dataset_name}...\n")
        
        def run_validation():
            from dataset_manager import DatasetValidator
            validator = DatasetValidator(dataset["root"])
            is_valid, metadata = validator.validate()
            
            def show_result():
                self.widgets['info_text'].delete("1.0", "end")
                self.widgets['info_text'].insert("end", f"Re-validation: {dataset_name}\n")
                self.widgets['info_text'].insert("end", "="*70 + "\n\n")
                
                if is_valid:
                    self.widgets['info_text'].insert("end", "✅ Dataset is valid\n\n")
                else:
                    self.widgets['info_text'].insert("end", "❌ Validation failed\n\n")
                    for error in metadata.get("errors", []):
                        self.widgets['info_text'].insert("end", f"• {error}\n")
                
                summary = self.dataset_manager.get_dataset_summary(dataset_name)
                self.widgets['info_text'].insert("end", f"\n{summary}")
            
            self.app.after(0, show_result)
        
        threading.Thread(target=run_validation, daemon=True).start()
    
    def preview_dataset(self):
        """Preview dataset"""
        dataset_name = self.widgets['dataset_var'].get()
        if dataset_name == "No datasets":
            messagebox.showinfo("Info", "No dataset selected")
            return
        
        dataset = self.dataset_manager.get_dataset(dataset_name)
        if not dataset:
            messagebox.showerror("Error", "Dataset not found")
            return
        
        try:
            from dataset_preview import DatasetPreviewWindow
            DatasetPreviewWindow(self.app, dataset)
        except Exception as e:
            logger.error(f"Preview error: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to open preview: {e}")
    
    def remove_dataset(self):
        """Remove dataset"""
        dataset_name = self.widgets['dataset_var'].get()
        if dataset_name == "No datasets":
            messagebox.showinfo("Info", "No dataset selected")
            return
        
        if messagebox.askyesno("Confirm", f"Remove '{dataset_name}'?\n(Files will NOT be deleted)"):
            if self.dataset_manager.remove_dataset(dataset_name):
                self.refresh_datasets()
                messagebox.showinfo("Success", f"Dataset '{dataset_name}' removed")
            else:
                messagebox.showerror("Error", "Failed to remove dataset")
    
    def _show_info(self, text):
        """Show text in info panel"""
        self.widgets['info_text'].delete("1.0", "end")
        self.widgets['info_text'].insert("end", text)
    
    def get_selected_dataset(self):
        """Get currently selected dataset name - called by training tab"""
        name = self.widgets['dataset_var'].get()
        return name if name != "No datasets" else None
