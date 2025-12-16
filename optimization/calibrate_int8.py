"""
INT8 Calibration for TensorRT using entropy-based calibration.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import List
import os


class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator for TensorRT.
    Uses entropy-based calibration algorithm.
    """
    
    def __init__(
        self,
        calibration_images: List[str],
        batch_size: int = 8,
        input_shape: tuple = (3, 640, 640),
        cache_file: str = 'calibration.cache'
    ):
        """
        Initialize INT8 calibrator.
        
        Args:
            calibration_images: List of image paths for calibration
            batch_size: Batch size for calibration
            input_shape: Input shape (C, H, W)
            cache_file: Path to save/load calibration cache
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_index = 0
        
        # Load and preprocess images
        print(f"[INFO] Loading {len(calibration_images)} calibration images...")
        self.image_paths = calibration_images[:len(calibration_images) // batch_size * batch_size]
        self.num_batches = len(self.image_paths) // batch_size
        
        print(f"[INFO] Using {len(self.image_paths)} images in {self.num_batches} batches")
        
        # Allocate device memory
        self.device_input = cuda.mem_alloc(int(
            self.batch_size * np.prod(input_shape) * np.dtype(np.float32).itemsize)
        )
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for calibration.
        
        Args:
            image_path: Path to image
        
        Returns:
            Preprocessed image array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW
        img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def get_batch(self, names: List[str]) -> List[int]:
        """
        Get next batch of calibration data.
        
        Args:
            names: List of input names (unused but required by interface)
        
        Returns:
            List of device memory pointers
        """
        if self.current_index >= self.num_batches:
            return None
        
        # Load batch of images
        batch_images = []
        start_idx = self.current_index * self.batch_size
        end_idx = start_idx + self.batch_size
        
        for img_path in self.image_paths[start_idx:end_idx]:
            img = self.preprocess_image(img_path)
            batch_images.append(img)
        
        # Stack into batch
        batch = np.stack(batch_images, axis=0)
        batch = np.ascontiguousarray(batch)
        
        # Copy to device
        cuda.memcpy_htod(self.device_input, batch)
        
        self.current_index += 1
        
        if self.current_index % 10 == 0:
            print(f"[INFO] Calibration progress: {self.current_index}/{self.num_batches} batches")
        
        return [int(self.device_input)]
    
    def get_batch_size(self) -> int:
        """Get batch size."""
        return self.batch_size
    
    def read_calibration_cache(self):
        """Read calibration cache if it exists."""
        if os.path.exists(self.cache_file):
            print(f"[INFO] Loading calibration cache from: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file."""
        print(f"[INFO] Writing calibration cache to: {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def collect_calibration_images(
    data_dir: str,
    num_images: int = 500,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
) -> List[str]:
    """
    Collect images for calibration from a directory.
    
    Args:
        data_dir: Directory containing images
        num_images: Number of images to collect
        extensions: Valid image extensions
    
    Returns:
        List of image paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"[INFO] Collecting images from: {data_dir}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(data_path.rglob(f"*{ext}")))
    
    print(f"[INFO] Found {len(image_paths)} images")
    
    # Randomly sample if we have more than needed
    if len(image_paths) > num_images:
        import random
        random.shuffle(image_paths)
        image_paths = image_paths[:num_images]
        print(f"[INFO] Randomly sampled {num_images} images")
    
    return [str(p) for p in image_paths]


def main():
    """Main calibration entry point."""
    parser = argparse.ArgumentParser(description='Generate INT8 calibration cache')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing calibration images')
    parser.add_argument('--onnx', type=str, default='models/model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='models/calibration.cache',
                        help='Output calibration cache path')
    parser.add_argument('--num-images', type=int, default=500,
                        help='Number of calibration images')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Calibration batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("INT8 CALIBRATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"ONNX model:     {args.onnx}")
    print(f"Output cache:   {args.output}")
    print(f"Num images:     {args.num_images}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Image size:     {args.imgsz}")
    print(f"{'='*60}\n")
    
    # Collect calibration images
    calibration_images = collect_calibration_images(
        args.data_dir,
        num_images=args.num_images
    )
    
    if len(calibration_images) == 0:
        print("[ERROR] No calibration images found")
        return
    
    # Create calibrator
    calibrator = Int8EntropyCalibrator(
        calibration_images=calibration_images,
        batch_size=args.batch_size,
        input_shape=(3, args.imgsz, args.imgsz),
        cache_file=args.output
    )
    
    print("\n[INFO] Calibration setup complete")
    print("[INFO] Run build_trt_engine.py with --precision fp16 and --calibration-cache {} to build INT8 engine".format(args.output))


if __name__ == "__main__":
    main()
