"""
Advanced Object Detection Training Script
Supports YOLOv8/v11 with strong augmentations, EMA, AMP, and comprehensive logging.
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb
from datetime import datetime


class AdvancedTrainer:
    """Advanced training pipeline with custom augmentations and monitoring."""
    
    def __init__(self, config_path: str, use_wandb: bool = True):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to dataset.yaml
            use_wandb: Enable Weights & Biases logging
        """
        self.config_path = config_path
        self.use_wandb = use_wandb
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "training" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project="cv-advanced-assessment",
                name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
    
    def get_augmentations(self):
        """
        Create strong augmentation pipeline using Albumentations.
        
        Returns:
            Albumentations composition
        """
        return A.Compose([
            # Geometric transforms
            A.RandomResizedCrop(
                height=640, width=640,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=0,
                p=0.5
            ),
            
            # Color augmentations
            A.OneOf([
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=25,
                    g_shift_limit=25,
                    b_shift_limit=25,
                    p=1.0
                ),
            ], p=0.5),
            
            # Blur and noise
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
            ], p=0.2),
            
            # Weather and lighting
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.3),
            
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),
            
            # Cutout/Coarse Dropout
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=3,
                fill_value=0,
                p=0.3
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def train(
        self,
        model_name: str = "yolov8n.pt",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
        workers: int = 8,
        patience: int = 50,
        save_period: int = 10,
        **kwargs
    ):
        """
        Train the model with advanced settings.
        
        Args:
            model_name: Model architecture (yolov8n.pt, yolov8s.pt, etc.)
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            device: CUDA device
            workers: Number of data loading workers
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional training arguments
        """
        print(f"[INFO] Initializing model: {model_name}")
        model = YOLO(model_name)
        
        # Training arguments
        train_args = {
            'data': self.config_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'workers': workers,
            'patience': patience,
            'save_period': save_period,
            
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': 0.001,  # Initial learning rate
            'lrf': 0.01,   # Final learning rate factor (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation settings (YOLO native)
            'mosaic': 1.0,      # Mosaic augmentation
            'mixup': 0.1,       # MixUp augmentation
            'copy_paste': 0.1,  # Copy-paste augmentation
            'degrees': 15.0,    # Rotation
            'translate': 0.1,   # Translation
            'scale': 0.5,       # Scale
            'shear': 0.0,       # Shear
            'perspective': 0.0, # Perspective
            'flipud': 0.1,      # Vertical flip
            'fliplr': 0.5,      # Horizontal flip
            'hsv_h': 0.015,     # HSV-Hue augmentation
            'hsv_s': 0.7,       # HSV-Saturation augmentation
            'hsv_v': 0.4,       # HSV-Value augmentation
            
            # Advanced training settings
            'amp': True,        # Automatic Mixed Precision
            'cos_lr': True,     # Cosine learning rate scheduler
            'close_mosaic': 10, # Disable mosaic in last N epochs
            
            # Logging
            'project': str(self.logs_dir),
            'name': f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'plots': True,
            'save': True,
            'save_json': True,
            'val': True,
            
            # Multi-scale training
            'rect': False,  # Rectangular training (set True for faster, False for multi-scale)
        }
        
        # Update with any additional kwargs
        train_args.update(kwargs)
        
        print(f"[INFO] Starting training with config:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # Train the model
        results = model.train(**train_args)
        
        # Save best model to models directory
        models_dir = self.project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = Path(model.trainer.best) if hasattr(model, 'trainer') else None
        if best_model_path and best_model_path.exists():
            import shutil
            dest_path = models_dir / "latest.pt"
            shutil.copy(str(best_model_path), str(dest_path))
            print(f"[INFO] Best model saved to: {dest_path}")
        
        # Validate and print metrics
        print("\n[INFO] Validating model...")
        metrics = model.val()
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE - FINAL METRICS")
        print(f"{'='*60}")
        print(f"mAP@0.5:     {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision:   {metrics.box.mp:.4f}")
        print(f"Recall:      {metrics.box.mr:.4f}")
        print(f"{'='*60}\n")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'final_map50': metrics.box.map50,
                'final_map': metrics.box.map,
                'final_precision': metrics.box.mp,
                'final_recall': metrics.box.mr,
            })
            wandb.finish()
        
        return results, metrics


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='training/dataset.yaml',
                        help='Path to dataset configuration')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AdvancedTrainer(
        config_path=args.config,
        use_wandb=not args.no_wandb
    )
    
    # Train
    trainer.train(
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
