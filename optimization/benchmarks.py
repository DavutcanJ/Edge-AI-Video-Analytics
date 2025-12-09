"""
Comprehensive benchmarking for different inference backends.
Measures latency, throughput, GPU utilization, and memory usage.
"""

import time
import json
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pynvml
from collections import defaultdict
import statistics


class GPUMonitor:
    """Monitor GPU metrics using pynvml."""
    
    def __init__(self):
        """Initialize GPU monitoring."""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except:
            print("[WARNING] GPU monitoring not available")
            self.available = False
    
    def get_metrics(self) -> Dict:
        """Get current GPU metrics."""
        if not self.available:
            return {}
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                'gpu_utilization': util.gpu,
                'memory_used_mb': mem_info.used / (1024 ** 2),
                'memory_total_mb': mem_info.total / (1024 ** 2),
                'memory_utilization': (mem_info.used / mem_info.total) * 100
            }
        except Exception as e:
            print(f"[WARNING] Error getting GPU metrics: {e}")
            return {}
    
    def __del__(self):
        """Cleanup."""
        if self.available:
            pynvml.nvmlShutdown()


class BenchmarkRunner:
    """Run comprehensive benchmarks for different inference backends."""
    
    def __init__(self, warmup_iterations: int = 10):
        """
        Initialize benchmark runner.
        
        Args:
            warmup_iterations: Number of warmup iterations
        """
        self.warmup_iterations = warmup_iterations
        self.gpu_monitor = GPUMonitor()
    
    def warmup(self, inference_fn, dummy_input):
        """
        Warmup the model.
        
        Args:
            inference_fn: Inference function
            dummy_input: Dummy input data
        """
        print(f"[INFO] Warming up ({self.warmup_iterations} iterations)...")
        for _ in range(self.warmup_iterations):
            inference_fn(dummy_input)
    
    def benchmark_latency(
        self,
        inference_fn,
        dummy_input,
        num_iterations: int = 100
    ) -> Dict:
        """
        Benchmark latency statistics.
        
        Args:
            inference_fn: Inference function
            dummy_input: Input data
            num_iterations: Number of iterations
        
        Returns:
            Dictionary with latency statistics
        """
        print(f"[INFO] Running latency benchmark ({num_iterations} iterations)...")
        
        latencies = []
        gpu_utils = []
        
        for i in range(num_iterations):
            # Record start time
            start = time.perf_counter()
            
            # Run inference
            inference_fn(dummy_input)
            
            # Record end time
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            # Get GPU metrics
            if i % 10 == 0:
                metrics = self.gpu_monitor.get_metrics()
                if metrics:
                    gpu_utils.append(metrics['gpu_utilization'])
        
        # Calculate statistics
        latencies.sort()
        results = {
            'latency_avg_ms': statistics.mean(latencies),
            'latency_median_ms': statistics.median(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p90_ms': np.percentile(latencies, 90),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'latency_min_ms': min(latencies),
            'latency_max_ms': max(latencies),
            'latency_std_ms': statistics.stdev(latencies),
        }
        
        if gpu_utils:
            results['gpu_utilization_avg'] = statistics.mean(gpu_utils)
        
        return results
    
    def benchmark_throughput(
        self,
        inference_fn,
        dummy_input,
        duration_seconds: int = 10
    ) -> Dict:
        """
        Benchmark throughput (FPS).
        
        Args:
            inference_fn: Inference function
            dummy_input: Input data
            duration_seconds: Duration to run benchmark
        
        Returns:
            Dictionary with throughput statistics
        """
        print(f"[INFO] Running throughput benchmark ({duration_seconds}s)...")
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration_seconds:
            inference_fn(dummy_input)
            iterations += 1
        
        elapsed = time.time() - start_time
        fps = iterations / elapsed
        
        # Get final GPU metrics
        gpu_metrics = self.gpu_monitor.get_metrics()
        
        results = {
            'throughput_fps': fps,
            'total_iterations': iterations,
            'duration_seconds': elapsed,
        }
        
        results.update(gpu_metrics)
        
        return results
    
    def benchmark_pytorch(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        device: str = 'cuda:0'
    ) -> Dict:
        """Benchmark PyTorch model."""
        print(f"\n{'='*60}")
        print("BENCHMARKING PYTORCH")
        print(f"{'='*60}\n")
        
        # Load model
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.model.to(device)
        model.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Define inference function
        def inference_fn(x):
            with torch.no_grad():
                return model.model(x)
        
        # Warmup
        self.warmup(inference_fn, dummy_input)
        
        # Benchmark
        latency_results = self.benchmark_latency(inference_fn, dummy_input)
        throughput_results = self.benchmark_throughput(inference_fn, dummy_input)
        
        return {
            'backend': 'pytorch',
            **latency_results,
            **throughput_results
        }
    
    def benchmark_onnx(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        providers: List[str] = None
    ) -> Dict:
        """Benchmark ONNX Runtime."""
        print(f"\n{'='*60}")
        print("BENCHMARKING ONNX RUNTIME")
        print(f"{'='*60}\n")
        
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load model
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Define inference function
        def inference_fn(x):
            return session.run(None, {input_name: x})
        
        # Warmup
        self.warmup(inference_fn, dummy_input)
        
        # Benchmark
        latency_results = self.benchmark_latency(inference_fn, dummy_input)
        throughput_results = self.benchmark_throughput(inference_fn, dummy_input)
        
        return {
            'backend': 'onnx',
            'providers': providers,
            **latency_results,
            **throughput_results
        }
    
    def benchmark_tensorrt(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...]
    ) -> Dict:
        """Benchmark TensorRT engine."""
        print(f"\n{'='*60}")
        print("BENCHMARKING TENSORRT")
        print(f"{'='*60}\n")
        
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Allocate buffers
        input_binding = engine.get_binding_index(engine.get_binding_name(0))
        output_binding = engine.get_binding_index(engine.get_binding_name(1))
        
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_shape = engine.get_binding_shape(output_binding)
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
        
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Define inference function
        def inference_fn(x):
            cuda.memcpy_htod_async(d_input, x, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(np.empty(output_shape, dtype=np.float32), d_output, stream)
            stream.synchronize()
        
        # Warmup
        self.warmup(inference_fn, dummy_input)
        
        # Benchmark
        latency_results = self.benchmark_latency(inference_fn, dummy_input)
        throughput_results = self.benchmark_throughput(inference_fn, dummy_input)
        
        return {
            'backend': 'tensorrt',
            **latency_results,
            **throughput_results
        }


def main():
    """Main benchmarking entry point."""
    parser = argparse.ArgumentParser(description='Benchmark inference backends')
    parser.add_argument('--pytorch', type=str, help='Path to PyTorch model')
    parser.add_argument('--onnx', type=str, help='Path to ONNX model')
    parser.add_argument('--tensorrt', type=str, help='Path to TensorRT engine')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations')
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(warmup_iterations=args.warmup)
    
    input_shape = (args.batch_size, 3, args.imgsz, args.imgsz)
    results = []
    
    # Benchmark PyTorch
    if args.pytorch and Path(args.pytorch).exists():
        try:
            result = runner.benchmark_pytorch(args.pytorch, input_shape)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] PyTorch benchmark failed: {e}")
    
    # Benchmark ONNX
    if args.onnx and Path(args.onnx).exists():
        try:
            result = runner.benchmark_onnx(args.onnx, input_shape)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] ONNX benchmark failed: {e}")
    
    # Benchmark TensorRT
    if args.tensorrt and Path(args.tensorrt).exists():
        try:
            result = runner.benchmark_tensorrt(args.tensorrt, input_shape)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] TensorRT benchmark failed: {e}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    
    for result in results:
        print(f"\nBackend: {result['backend'].upper()}")
        print(f"  Latency (avg):  {result['latency_avg_ms']:.2f} ms")
        print(f"  Latency (p95):  {result['latency_p95_ms']:.2f} ms")
        print(f"  Throughput:     {result['throughput_fps']:.2f} FPS")
        if 'gpu_utilization_avg' in result:
            print(f"  GPU Util:       {result['gpu_utilization_avg']:.1f}%")
    
    print(f"\n[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
