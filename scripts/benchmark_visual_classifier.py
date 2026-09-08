"""Phase 4C — CPU Inference Latency & Memory Benchmark.

# ruff: noqa: N806, N803
"""

import gc
import json
import os
import time
import tracemalloc

import numpy as np

from dta.perception.visual_classifier import VisualCandidateClassifier


def measure_memory_mb() -> float:
    """Return memory allocated in megabytes using tracemalloc."""
    current, peak = tracemalloc.get_traced_memory()
    return float(peak / (1024.0 * 1024.0))


def benchmark_batch_size(
    classifier: VisualCandidateClassifier,
    crops: list[np.ndarray],
    num_iterations: int = 100,
) -> dict[str, float]:
    """Benchmark feature extraction latency and throughput for a specific crop batch size."""
    # Warmup
    for _ in range(10):
        _ = classifier.extract_batch_features(crops)

    latencies_ms = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = classifier.extract_batch_features(crops)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms)
    total_crops = len(crops)

    per_crop_latencies = lat_arr / float(total_crops)

    return {
        "batch_size": total_crops,
        "iterations": num_iterations,
        "batch_latency_mean_ms": float(np.mean(lat_arr)),
        "batch_latency_median_ms": float(np.median(lat_arr)),
        "batch_latency_p95_ms": float(np.percentile(lat_arr, 95)),
        "batch_latency_max_ms": float(np.max(lat_arr)),
        "per_crop_latency_mean_ms": float(np.mean(per_crop_latencies)),
        "per_crop_latency_median_ms": float(np.median(per_crop_latencies)),
        "per_crop_latency_p95_ms": float(np.percentile(per_crop_latencies, 95)),
        "throughput_crops_per_sec": float(total_crops / (np.mean(lat_arr) / 1000.0)),
    }


def main() -> None:
    """Execute CPU latency and memory benchmark for VisualCandidateClassifier."""
    tracemalloc.start()
    os.makedirs("audit", exist_ok=True)
    gc.collect()

    initial_mem_mb = measure_memory_mb()

    model_path = os.path.join("models", "entity_classifier.pkl")
    classifier = VisualCandidateClassifier(model_path=model_path)

    after_load_mem_mb = measure_memory_mb()
    model_mem_delta_mb = round(after_load_mem_mb - initial_mem_mb, 2)

    # Synthetic candidate crops (BGR 40x80)
    crop_b1 = [np.random.randint(0, 255, (80, 40, 3), dtype=np.uint8)]
    crop_b8 = [np.random.randint(0, 255, (80, 40, 3), dtype=np.uint8) for _ in range(8)]

    print("\n--- Visual Classifier CPU Benchmark ---")
    print(f"RAM Memory Footprint (Model + MobileNetV3): {model_mem_delta_mb:.2f} MB (Total RSS: {after_load_mem_mb:.2f} MB)")

    res_b1 = benchmark_batch_size(classifier, crop_b1, num_iterations=100)
    print("\n[Batch Size 1 (Single Crop)]")
    print(f"  Mean Latency: {res_b1['batch_latency_mean_ms']:.2f} ms")
    print(f"  p50 Latency : {res_b1['batch_latency_median_ms']:.2f} ms")
    print(f"  p95 Latency : {res_b1['batch_latency_p95_ms']:.2f} ms")
    print(f"  Throughput  : {res_b1['throughput_crops_per_sec']:.1f} crops/sec")

    res_b8 = benchmark_batch_size(classifier, crop_b8, num_iterations=100)
    print("\n[Batch Size 8 (Candidate Batch)]")
    print(f"  Mean Batch Latency: {res_b8['batch_latency_mean_ms']:.2f} ms")
    print(f"  p50 Batch Latency : {res_b8['batch_latency_median_ms']:.2f} ms")
    print(f"  p95 Batch Latency : {res_b8['batch_latency_p95_ms']:.2f} ms")
    print(f"  Mean Per-Crop Lat : {res_b8['per_crop_latency_mean_ms']:.2f} ms")
    print(f"  Throughput        : {res_b8['throughput_crops_per_sec']:.1f} crops/sec")

    benchmark_report = {
        "device": "CPU",
        "memory_rss_mb": round(after_load_mem_mb, 2),
        "model_memory_mb": model_mem_delta_mb,
        "batch_size_1": res_b1,
        "batch_size_8": res_b8,
    }

    report_json = os.path.join("audit", "visual_classifier_benchmark.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\nBenchmark complete. Saved results to '{report_json}'.")


if __name__ == "__main__":
    main()
