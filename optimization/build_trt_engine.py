"""
Build TensorRT engines from ONNX models.
Supports FP32, FP16, and INT8 precision with optimization profiles.
"""

import tensorrt as trt
import numpy as np
import argparse
from pathlib import Path
import pycuda.driver as cuda
import pycuda.autoinit
import sys


class TRTEngineBuilder:
    """Build TensorRT engines from ONNX models."""
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    def __init__(self, onnx_path: str):
        """
        Initialize TensorRT engine builder.
        
        Args:
            onnx_path: Path to ONNX model
        """
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        print(f"[INFO] Loading ONNX model: {self.onnx_path}")
    
    def build_engine(
        self,
        output_path: str = None,
        precision: str = 'fp16',
        batch_size: int = 1,
        workspace_size: int = 4,
        min_shapes: tuple = (1, 3, 320, 320),
        opt_shapes: tuple = (1, 3, 640, 640),
        max_shapes: tuple = (8, 3, 1280, 1280),
        calibration_cache: str = None,
        verbose: bool = False
    ):
        """
        Build TensorRT engine with optimization profiles.
        
        Args:
            output_path: Output engine path
            precision: 'fp32', 'fp16', or 'int8'
            batch_size: Maximum batch size
            workspace_size: Workspace size in GB
            min_shapes: Minimum input shapes (batch, channels, height, width)
            opt_shapes: Optimal input shapes
            max_shapes: Maximum input shapes
            calibration_cache: Path to INT8 calibration cache
            verbose: Enable verbose logging
        
        Returns:
            Path to built engine
        """
        if verbose:
            self.TRT_LOGGER.min_severity = trt.Logger.VERBOSE
        
        # Set output path
        if output_path is None:
            suffix = f"_{precision}.engine"
            output_path = self.onnx_path.with_suffix(suffix)
        else:
            output_path = Path(output_path)
        
        print(f"\n{'='*60}")
        print("TENSORRT ENGINE BUILD CONFIGURATION")
        print(f"{'='*60}")
        print(f"ONNX model:     {self.onnx_path}")
        print(f"Output engine:  {output_path}")
        print(f"Precision:      {precision.upper()}")
        print(f"Max batch size: {batch_size}")
        print(f"Workspace:      {workspace_size} GB")
        print(f"Min shapes:     {min_shapes}")
        print(f"Opt shapes:     {opt_shapes}")
        print(f"Max shapes:     {max_shapes}")
        if precision == 'int8':
            print(f"Calib cache:    {calibration_cache}")
        print(f"{'='*60}\n")
        
        # Create builder and network
        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.TRT_LOGGER)
        
        # Parse ONNX model
        print("[INFO] Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("[ERROR] Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
        
        print("[OK] ONNX model parsed successfully")
        
        # Create builder config
        config = builder.create_builder_config()
        
        # Set workspace size (in bytes)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size * (1 << 30)
        )
        
        # Set precision
        if precision == 'fp16':
            if not builder.platform_has_fast_fp16:
                print("[WARNING] FP16 not supported on this platform")
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 mode enabled")
        
        elif precision == 'int8':
            if not builder.platform_has_fast_int8:
                print("[WARNING] INT8 not supported on this platform")
            config.set_flag(trt.BuilderFlag.INT8)
            
            # Set calibration cache
            if calibration_cache and Path(calibration_cache).exists():
                print(f"[INFO] Loading calibration cache: {calibration_cache}")
                # You would implement a calibrator class here
                # For now, we'll just enable INT8 without calibration
                print("[WARNING] Calibration not implemented in this example")
            else:
                print("[WARNING] No calibration cache provided for INT8")
            
            print("[INFO] INT8 mode enabled")
        
        # Create optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        
        profile.set_shape(
            input_name,
            min_shapes,  # min
            opt_shapes,  # opt
            max_shapes   # max
        )
        config.add_optimization_profile(profile)
        
        print(f"[INFO] Optimization profile created for input: {input_name}")
        
        # Build engine
        print("\n[INFO] Building TensorRT engine (this may take a while)...")
        try:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("[ERROR] Failed to build engine")
                sys.exit(1)
            
            # Save engine
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)
            
            print(f"[SUCCESS] Engine saved to: {output_path}")
            
            # Print engine info
            self._print_engine_info(serialized_engine)
            
            return str(output_path)
            
        except Exception as e:
            print(f"[ERROR] Engine build failed: {e}")
            raise
    
    def _print_engine_info(self, serialized_engine: bytes):
        """Print information about the built engine."""
        runtime = trt.Runtime(self.TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        print(f"\n{'='*60}")
        print("ENGINE INFORMATION")
        print(f"{'='*60}")
        binding_count = getattr(engine, 'num_io_tensors', None)
        if binding_count is None:
            binding_count = len(engine)
        print(f"Number of bindings: {binding_count}")
        for i in range(binding_count):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)
            print(f"\nBinding {i}:")
            print(f"  Name:   {name}")
            print(f"  Type:   {'INPUT' if is_input else 'OUTPUT'}")
            print(f"  Shape:  {shape}")
            print(f"  Dtype:  {dtype}")
        
        print(f"\n{'='*60}\n")


def main():
    """Main build entry point."""
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX')
    parser.add_argument('--onnx', type=str, default='models/model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output engine path')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Precision mode')
    parser.add_argument('--batch', type=int, default=8,
                        help='Maximum batch size')
    parser.add_argument('--workspace', type=int, default=4,
                        help='Workspace size in GB')
    parser.add_argument('--calibration-cache', type=str, default=None,
                        help='Path to INT8 calibration cache')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # Optimization profile shapes
    parser.add_argument('--min-batch', type=int, default=1)
    parser.add_argument('--opt-batch', type=int, default=1)
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--min-size', type=int, default=320)
    parser.add_argument('--opt-size', type=int, default=640)
    parser.add_argument('--max-size', type=int, default=1280)
    
    args = parser.parse_args()
    
    # Build shapes
    min_shapes = (args.min_batch, 3, args.min_size, args.min_size)
    opt_shapes = (args.opt_batch, 3, args.opt_size, args.opt_size)
    max_shapes = (args.max_batch, 3, args.max_size, args.max_size)
    
    # Initialize builder
    builder = TRTEngineBuilder(args.onnx)
    
    # Build engine
    builder.build_engine(
        output_path=args.output,
        precision=args.precision,
        batch_size=args.batch,
        workspace_size=args.workspace,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        calibration_cache=args.calibration_cache,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
