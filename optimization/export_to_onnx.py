"""
Export PyTorch model to ONNX format with dynamic axes support.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import sys


class ONNXExporter:
    """Export PyTorch models to ONNX format."""
    
    def __init__(self, model_path: str, output_path: str = None):
        """
        Initialize ONNX exporter.
        
        Args:
            model_path: Path to PyTorch model (.pt)
            output_path: Output path for ONNX model
        """
        self.model_path = Path(model_path)
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.model_path.with_suffix('.onnx')
        
        print(f"[INFO] Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
    
    def export(
        self,
        imgsz: int = 640,
        batch_size: int = 1,
        opset: int = 12,
        dynamic: bool = True,
        simplify: bool = True,
        half: bool = False
    ):
        """
        Export model to ONNX.
        
        Args:
            imgsz: Input image size
            batch_size: Batch size
            opset: ONNX opset version
            dynamic: Enable dynamic axes
            simplify: Simplify ONNX model
            half: Export in FP16
        """
        print(f"\n{'='*60}")
        print("ONNX EXPORT CONFIGURATION")
        print(f"{'='*60}")
        print(f"Input size:     {imgsz}x{imgsz}")
        print(f"Batch size:     {batch_size}")
        print(f"Opset version:  {opset}")
        print(f"Dynamic axes:   {dynamic}")
        print(f"Simplify:       {simplify}")
        print(f"Half precision: {half}")
        print(f"Output path:    {self.output_path}")
        print(f"{'='*60}\n")
        
        # Export using Ultralytics YOLO
        try:
            export_path = self.model.export(
                format='onnx',
                imgsz=imgsz,
                dynamic=dynamic,
                simplify=simplify,
                opset=opset,
                half=half
            )
            
            print(f"[SUCCESS] Model exported to: {export_path}")
            
            # Validate ONNX model
            print("\n[INFO] Validating ONNX model...")
            self.validate_onnx(export_path, imgsz, batch_size, dynamic)
            
            return export_path
            
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            raise
    
    def validate_onnx(
        self,
        onnx_path: str,
        imgsz: int,
        batch_size: int,
        dynamic: bool
    ):
        """
        Validate ONNX model by checking outputs match PyTorch.
        
        Args:
            onnx_path: Path to ONNX model
            imgsz: Input image size
            batch_size: Batch size
            dynamic: Whether dynamic axes were used
        """
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model structure is valid")
        
        # Check dynamic axes
        if dynamic:
            print("\n[INFO] Checking dynamic axes...")
            input_shape = onnx_model.graph.input[0].type.tensor_type.shape
            dynamic_dims = []
            for i, dim in enumerate(input_shape.dim):
                if dim.dim_param:
                    dynamic_dims.append((i, dim.dim_param))
            
            if dynamic_dims:
                print(f"[OK] Dynamic dimensions found: {dynamic_dims}")
            else:
                print("[WARNING] No dynamic dimensions found despite dynamic=True")
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names and shapes
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"\n[INFO] ONNX Model I/O:")
        print(f"  Input: {input_name}")
        print(f"  Outputs: {output_names}")
        
        # Test inference with dummy data
        print("\n[INFO] Testing inference with dummy data...")
        dummy_input = np.random.randn(batch_size, 3, imgsz, imgsz).astype(np.float32)
        
        try:
            onnx_outputs = session.run(output_names, {input_name: dummy_input})
            print(f"[OK] ONNX inference successful")
            print(f"  Output shapes: {[out.shape for out in onnx_outputs]}")
        except Exception as e:
            print(f"[ERROR] ONNX inference failed: {e}")
            raise
        
        # Test with different batch sizes if dynamic
        if dynamic:
            print("\n[INFO] Testing dynamic batch sizes...")
            for test_batch in [1, 2, 4]:
                test_input = np.random.randn(test_batch, 3, imgsz, imgsz).astype(np.float32)
                try:
                    outputs = session.run(output_names, {input_name: test_input})
                    print(f"[OK] Batch size {test_batch}: output shape {outputs[0].shape}")
                except Exception as e:
                    print(f"[ERROR] Batch size {test_batch} failed: {e}")
        
        # Compare with PyTorch (if GPU available)
        if torch.cuda.is_available():
            print("\n[INFO] Comparing ONNX vs PyTorch outputs...")
            try:
                # PyTorch inference
                device = torch.device('cuda:0')
                pt_model = self.model.model.to(device)
                pt_model.eval()
                
                with torch.no_grad():
                    pt_input = torch.from_numpy(dummy_input).to(device)
                    pt_output = pt_model(pt_input)
                
                # Get predictions from both
                if isinstance(pt_output, (list, tuple)):
                    pt_pred = pt_output[0].cpu().numpy()
                else:
                    pt_pred = pt_output.cpu().numpy()
                
                onnx_pred = onnx_outputs[0]
                
                # Calculate difference
                max_diff = np.abs(pt_pred - onnx_pred).max()
                mean_diff = np.abs(pt_pred - onnx_pred).mean()
                
                print(f"  Max difference:  {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                
                if max_diff < 1e-3:
                    print("[OK] PyTorch and ONNX outputs match closely")
                else:
                    print("[WARNING] Outputs differ significantly")
                    
            except Exception as e:
                print(f"[WARNING] Could not compare outputs: {e}")
        
        print(f"\n{'='*60}")
        print("VALIDATION COMPLETE")
        print(f"{'='*60}\n")


def main():
    """Main export entry point."""
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model', type=str, default='models/latest.pt',
                        help='Path to PyTorch model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    parser.add_argument('--no-dynamic', action='store_true',
                        help='Disable dynamic axes')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Disable simplification')
    parser.add_argument('--half', action='store_true',
                        help='Export in FP16')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = ONNXExporter(args.model, args.output)
    
    # Export
    exporter.export(
        imgsz=args.imgsz,
        batch_size=args.batch,
        opset=args.opset,
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
        half=args.half
    )


if __name__ == "__main__":
    main()
