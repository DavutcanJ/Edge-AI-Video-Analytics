"""
Advanced augmentation strategies for object detection.
Includes Mosaic, MixUp, CutOut implementations and custom transformations.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_detection_augmentation_pipeline(
    img_size: int = 640,
    augment_level: str = "strong"
) -> A.Compose:
    """
    Create a comprehensive augmentation pipeline for object detection.
        Based on the  complexity of augmentation it is divided into three parts
        Horizontal Flip and Contrast change
        Flip + Shift + Blurr
        + Color Jitter + Noise + Weather effects + Cutout etc.

        Normalization is always applied at the end.
        
    Args:
        img_size: Target image size
        augment_level: 'light', 'medium', or 'strong'
    
    Returns:
        Albumentations Compose object
    """
    
    if augment_level == "light":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Blur(blur_limit=3, p=0.1),
        ]
    
    elif augment_level == "medium":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.2),
            A.CoarseDropout(max_holes=5, max_height=32, max_width=32, p=0.2),
        ]
    
    else:  # strong
        transforms = [
            # Geometric
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
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
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.3
            ),
            
            # Color augmentations
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            ], p=0.5),
            
            # Blur
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),
            
            # Lighting
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
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
            
            # Cutout
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=3,
                fill_value=0,
                p=0.3
            ),
            
            # Weather
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=5,
                    brightness_coefficient=0.7,
                    p=1.0
                ),
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.3,
                    alpha_coef=0.08,
                    p=1.0
                ),
            ], p=0.1),
        ]
    
    # Always normalize at the end
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        )
    )
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
            min_area=0.0
        )
    )


class MosaicAugmentation:
    """
    Mosaic augmentation: combines 4 images into one.
    Used in YOLOv4, YOLOv5, and later versions.
    """
    
    def __init__(self, img_size: int = 640):
        """
        Initialize Mosaic augmentation.
        
        Args:
            img_size: Output image size
        """
        self.img_size = img_size
    
    def __call__(
        self,
        images: List[np.ndarray],
        bboxes: List[np.ndarray],
        labels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation to 4 images.
        
        Args:
            images: List of 4 images (H, W, 3)
            bboxes: List of 4 bbox arrays (N, 4) in YOLO format (cx, cy, w, h)
            labels: List of 4 label arrays (N,)
        
        Returns:
            Tuple of (mosaic_image, mosaic_bboxes, mosaic_labels)
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        
        s = self.img_size
        yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        for i, (img, boxes, lbls) in enumerate(zip(images, bboxes, labels)):
            h, w = img.shape[:2]
            
            # Place images in mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Adjust bboxes
            if len(boxes) > 0:
                boxes_copy = boxes.copy()
                boxes_copy[:, 0] = w * boxes[:, 0] + padw  # cx
                boxes_copy[:, 1] = h * boxes[:, 1] + padh  # cy
                boxes_copy[:, 2] = w * boxes[:, 2]  # width
                boxes_copy[:, 3] = h * boxes[:, 3]  # height
                
                mosaic_bboxes.append(boxes_copy)
                mosaic_labels.append(lbls)
        
        # Concatenate all boxes and labels
        if len(mosaic_bboxes) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            
            # Clip to mosaic boundaries and convert back to YOLO format
            np.clip(mosaic_bboxes[:, 0], 0, s * 2, out=mosaic_bboxes[:, 0])
            np.clip(mosaic_bboxes[:, 1], 0, s * 2, out=mosaic_bboxes[:, 1])
            
            # Normalize to [0, 1]
            mosaic_bboxes[:, [0, 2]] /= (s * 2)
            mosaic_bboxes[:, [1, 3]] /= (s * 2)
        else:
            mosaic_bboxes = np.array([])
            mosaic_labels = np.array([])
        
        # Resize to target size
        mosaic_img = cv2.resize(mosaic_img, (s, s))
        
        return mosaic_img, mosaic_bboxes, mosaic_labels


class MixUpAugmentation:
    """
    MixUp augmentation: blends two images together.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: Blending factor (0.0 to 1.0)
        """
        self.alpha = alpha
    
    def __call__(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        bboxes1: np.ndarray,
        bboxes2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply MixUp augmentation.
        
        Args:
            img1, img2: Images to blend
            bboxes1, bboxes2: Bounding boxes
            labels1, labels2: Labels
        
        Returns:
            Tuple of (mixed_image, mixed_bboxes, mixed_labels)
        """
        r = np.random.beta(self.alpha, self.alpha)
        mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        # Combine bboxes and labels
        mixed_bboxes = np.vstack([bboxes1, bboxes2]) if len(bboxes1) > 0 and len(bboxes2) > 0 else np.array([])
        mixed_labels = np.concatenate([labels1, labels2]) if len(labels1) > 0 and len(labels2) > 0 else np.array([])
        
        return mixed_img, mixed_bboxes, mixed_labels


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """Get training transforms."""
    return create_detection_augmentation_pipeline(img_size, augment_level="strong")


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))
