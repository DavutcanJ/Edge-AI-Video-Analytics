"""
Dataset Preview Window
Shows sample images from dataset with bounding boxes
"""

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class DatasetPreviewWindow(ctk.CTkToplevel):
    """Window to preview dataset samples with annotations."""
    
    def __init__(self, parent, dataset_info: dict):
        super().__init__(parent)
        
        self.dataset_info = dataset_info
        self.dataset_root = Path(dataset_info["root"])
        self.class_names = dataset_info["class_names"]
        
        self.title(f"Dataset Preview: {dataset_info['name']}")
        self.geometry("1000x700")
        
        self.setup_ui()
        self.load_samples()
    
    def setup_ui(self):
        """Setup the preview UI."""
        # Top controls
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(control_frame, text="Dataset Preview", font=("Arial", 16, "bold")).pack(side="left", padx=10)
        
        # Split selector
        ctk.CTkLabel(control_frame, text="Split:", font=("Arial", 12)).pack(side="left", padx=5)
        self.split_var = ctk.StringVar(value="train")
        splits = list(self.dataset_info["metadata"]["splits"].keys())
        ctk.CTkOptionMenu(
            control_frame,
            variable=self.split_var,
            values=splits,
            command=self.on_split_changed,
            width=100
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            control_frame, text="🔄 Refresh",
            command=self.load_samples, width=100
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            control_frame, text="❌ Close",
            command=self.destroy, width=100, fg_color="#8B0000"
        ).pack(side="right", padx=10)
        
        # Info label
        self.info_label = ctk.CTkLabel(self, text="", font=("Arial", 11))
        self.info_label.pack(pady=5)
        
        # Scrollable frame for images
        self.scroll_frame = ctk.CTkScrollableFrame(self, width=960, height=580)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def on_split_changed(self, value):
        """Handle split selection change."""
        self.load_samples()
    
    def load_samples(self):
        """Load and display sample images."""
        # Clear existing widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        split = self.split_var.get()
        images_dir = self.dataset_root / split / "images"
        labels_dir = self.dataset_root / split / "labels"
        
        if not images_dir.exists():
            ctk.CTkLabel(
                self.scroll_frame,
                text=f"Split '{split}' not found",
                font=("Arial", 14)
            ).pack(pady=20)
            return
        
        # Get sample images
        image_files = self._get_sample_images(images_dir, count=6)
        
        if not image_files:
            ctk.CTkLabel(
                self.scroll_frame,
                text="No images found in this split",
                font=("Arial", 14)
            ).pack(pady=20)
            return
        
        # Update info
        split_info = self.dataset_info["metadata"]["splits"][split]
        self.info_label.configure(
            text=f"Split: {split} | Images: {split_info['image_count']} | "
                 f"Labels: {split_info['label_count']} | Showing {len(image_files)} samples"
        )
        
        # Display images in grid (2 columns)
        for idx, img_path in enumerate(image_files):
            row = idx // 2
            col = idx % 2
            
            # Create frame for this image
            img_frame = ctk.CTkFrame(self.scroll_frame)
            img_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Configure grid weights
            self.scroll_frame.grid_columnconfigure(col, weight=1)
            
            # Load and display image with annotations
            try:
                annotated_img = self._load_annotated_image(img_path, labels_dir)
                
                if annotated_img:
                    # Display image
                    ctk_img = ctk.CTkImage(
                        light_image=annotated_img,
                        dark_image=annotated_img,
                        size=(450, 350)
                    )
                    img_label = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                    img_label.pack(pady=5)
                    
                    # Image name
                    ctk.CTkLabel(
                        img_frame,
                        text=img_path.name,
                        font=("Arial", 10)
                    ).pack(pady=2)
                
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                ctk.CTkLabel(
                    img_frame,
                    text=f"Error loading {img_path.name}",
                    font=("Arial", 10),
                    text_color="red"
                ).pack(pady=10)
    
    def _get_sample_images(self, images_dir: Path, count: int = 6) -> List[Path]:
        """Get sample image paths."""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
            if len(image_files) >= count:
                break
        
        return image_files[:count]
    
    def _load_annotated_image(self, img_path: Path, labels_dir: Path) -> Image.Image:
        """Load image and draw bounding boxes from label file."""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load corresponding label
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert YOLO format to pixel coordinates
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            # Get class name
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            # Draw bounding box
                            color = self._get_class_color(class_id)
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label background
                            label_text = f"{class_name}"
                            (text_width, text_height), _ = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            cv2.rectangle(
                                img,
                                (x1, y1 - text_height - 5),
                                (x1 + text_width, y1),
                                color,
                                -1
                            )
                            
                            # Draw label text
                            cv2.putText(
                                img, label_text,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1
                            )
            except Exception as e:
                logger.error(f"Error reading label {label_path}: {e}")
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        return pil_img
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID."""
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring green
            (255, 0, 128),    # Rose
        ]
        return colors[class_id % len(colors)]
