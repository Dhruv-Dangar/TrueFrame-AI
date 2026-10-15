import time
import tracemalloc
import numpy as np
import torch
from PIL import Image

from src.services.face_detector import face_detector_service
from src.services.model_engine import model_engine
from src.services.explainability import explainability_service
from src.services.forensic_pipeline import forensic_pipeline


def run_cpu_benchmark(iterations: int = 10):
    """
    Executes a formal CPU benchmark on the host Intel Core i5-1235U CPU.
    Measures per-component latency and peak RAM usage.
    """
    print(f"=== TrueFrame AI: CPU Benchmark (Host: Intel Core i5-1235U, {iterations} iterations) ===")

    # Create synthetic high-resolution image in memory
    img = Image.fromarray(np.uint8(np.random.randint(50, 200, (600, 600, 3))))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    # Track memory
    tracemalloc.start()

    # 1. Warmup pass
    _ = forensic_pipeline.analyze_image_bytes(raw_bytes, "warmup.jpg")

    # 2. Benchmark Model Inference Alone
    dummy_face = Image.fromarray(np.uint8(np.random.randint(50, 200, (224, 224, 3))))
    model_latencies = []
    cam_latencies = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        score, feat_maps = model_engine.predict(dummy_face)
        t1 = time.perf_counter()
        model_latencies.append((t1 - t0) * 1000.0)

        t2 = time.perf_counter()
        _ = explainability_service.generate_eigen_cam(feat_maps, dummy_face)
        t3 = time.perf_counter()
        cam_latencies.append((t3 - t2) * 1000.0)

    # 3. Benchmark End-to-End Pipeline
    pipeline_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = forensic_pipeline.analyze_image_bytes(raw_bytes, "benchmark_sample.jpg")
        t1 = time.perf_counter()
        pipeline_latencies.append((t1 - t0) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_model_ms = np.mean(model_latencies)
    std_model_ms = np.std(model_latencies)
    avg_cam_ms = np.mean(cam_latencies)
    avg_pipe_ms = np.mean(pipeline_latencies)
    std_pipe_ms = np.std(pipeline_latencies)
    peak_mb = peak_mem / (1024 * 1024)

    print("\n--- Benchmark Results ---")
    print(f"EfficientNet-B0 CPU Inference: {avg_model_ms:.2f} ms ± {std_model_ms:.2f} ms")
    print(f"Eigen-CAM Saliency Generation: {avg_cam_ms:.2f} ms")
    print(f"End-to-End Pipeline Latency:  {avg_pipe_ms:.2f} ms ± {std_pipe_ms:.2f} ms")
    print(f"Peak Traced Memory:           {peak_mb:.2f} MB")
    print("=========================================================================\n")

    return {
        "avg_model_ms": round(float(avg_model_ms), 2),
        "std_model_ms": round(float(std_model_ms), 2),
        "avg_cam_ms": round(float(avg_cam_ms), 2),
        "avg_pipe_ms": round(float(avg_pipe_ms), 2),
        "std_pipe_ms": round(float(std_pipe_ms), 2),
        "peak_mb": round(float(peak_mb), 2),
    }


if __name__ == "__main__":
    run_cpu_benchmark()
