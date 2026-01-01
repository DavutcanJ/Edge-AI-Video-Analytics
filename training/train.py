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
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb
from datetime import datetime
from tqdm import tqdm
import json

# Import custom augmentation pipeline
from training.augmentations import create_detection_augmentation_pipeline


class AdvancedTrainer:
    """Advanced training pipeline with custom augmentations and monitoring."""
    
    def __init__(self, config_path: str, use_wandb: bool = True, augment_level: str = "strong"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to dataset.yaml
            use_wandb: Enable Weights & Biases logging
            augment_level: Augmentation intensity ('light', 'medium', 'strong')
        """
        self.config_path = config_path
        self.use_wandb = use_wandb
        self.augment_level = augment_level
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
                config={
                    **self.config,
                    "augment_level": augment_level
                }
            )
        
        # Create augmentation pipeline
        self.augmentation_pipeline = create_detection_augmentation_pipeline(
            img_size=640,
            augment_level=augment_level
        )
        print(f"[INFO] Augmentation pipeline created with level: {augment_level}")
    
    def get_augmentation_config(self):
        """
        Get YOLO-native augmentation config based on custom pipeline level.
        This maps our custom augmentation intensity to YOLO's native parameters.
        
        Returns:
            Dictionary of YOLO augmentation parameters
        """
        if self.augment_level == "light":
            return {
                'mosaic': 0.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'degrees': 0.0,
                'translate': 0.05,
                'scale': 0.1,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'hsv_h': 0.01,
                'hsv_s': 0.3,
                'hsv_v': 0.2,
            }
        elif self.augment_level == "medium":
            return {
                'mosaic': 0.3,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'degrees': 5.0,
                'translate': 0.1,
                'scale': 0.2,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.1,
                'fliplr': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.5,
                'hsv_v': 0.3,
            }
        else:  # strong
            return {
                'mosaic': 0.5,
                'mixup': 0.1,
                'copy_paste': 0.1,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.3,
                'shear': 2.0,
                'perspective': 0.0001,
                'flipud': 0.1,
                'fliplr': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
            }
    
    def train(
        self,
        model_name: str = "yolov8n.pt",
        epochs: int = 50,
        imgsz: int = 480,
        batch: int = 8,
        device: str = "0",
        workers: int = 4,
        patience: int = 20,
        save_period: int = 5,
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
        print(f"\n{'='*70}")
        print(f"[INFO] Initializing model: {model_name}")
        print(f"{'='*70}")
        
        model = YOLO(model_name)
        
        # Get augmentation config based on level
        augment_config = self.get_augmentation_config()
        
        # Training arguments - optimized for RTX 3050 Ti (4GB VRAM)
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
            'warmup_epochs': 2.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Advanced training settings - memory optimized
            'amp': True,        # Automatic Mixed Precision (IMPORTANT for memory)
            'cos_lr': True,     # Cosine learning rate scheduler
            'close_mosaic': 5,  # Disable mosaic in last N epochs
            'cache': False,     # Don't cache images in RAM
            'rect': True,       # Rectangular training (faster, less memory)
            
            # Logging
            'project': str(self.logs_dir),
            'name': f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,    # Show progress
            'plots': True,
            'save': True,
            'save_json': True,
            'val': True,
        }
        
        # Apply custom augmentation config
        train_args.update(augment_config)
        
        # Update with any additional kwargs
        train_args.update(kwargs)
        
        print(f"\n📊 Training Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch}")
        print(f"  Image Size: {imgsz}")
        print(f"  Device: {device}")
        print(f"  Workers: {workers}")
        print(f"  Learning Rate: {train_args['lr0']}")
        print(f"  Patience (Early Stopping): {patience}")
        print(f"  Augmentation Level: {self.augment_level}")
        print(f"\n🎨 Augmentation Settings:")
        for key, value in augment_config.items():
            print(f"  {key}: {value}")
        print(f"\n{'='*70}\n")
        
        # Train the model
        try:
            results = model.train(**train_args)
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
            return None, None
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            return None, None
        
        # Save best model to models directory
        models_dir = self.project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = Path(model.trainer.best) if hasattr(model, 'trainer') else None
        if best_model_path and best_model_path.exists():
            import shutil
            dest_path = models_dir / "latest.pt"
            shutil.copy(str(best_model_path), str(dest_path))
            print(f"✅ Best model saved to: {dest_path}")
        
        # Validate and print metrics
        print("\n" + "="*70)
        print("📈 Validating model...")
        print("="*70)
        metrics = model.val()
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE - FINAL METRICS")
        print(f"{'='*70}")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        print(f"{'='*70}\n")
        
        # Save metrics to JSON
        metrics_path = models_dir / "latest_metrics.json"
        metrics_dict = {
            'map50': float(metrics.box.map50),
            'map': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'timestamp': datetime.now().isoformat()
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
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
    parser.add_argument('--augment', type=str, default='strong',
                        choices=['light', 'medium', 'strong'],
                        help='Augmentation level (light/medium/strong)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🚀 Advanced Object Detection Training")
    print(f"{'='*70}")
    print(f"Augmentation Level: {args.augment}")
    print(f"W&B Logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"{'='*70}\n")
    
    # Initialize trainer
    trainer = AdvancedTrainer(
        config_path=args.config,
        use_wandb=not args.no_wandb,
        augment_level=args.augment
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
