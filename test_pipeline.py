import numpy as np
import time
from pipeline_api import FastAcousticPipeline


def main():
    sr = 8000
    channels = 32
    duration = 1.0  # 1 sec chunk
    num_samples = int(sr * duration)

    # 1. Setup geometry (Circular array)
    angles = np.linspace(0, 2 * np.pi, channels + 1)[:-1]
    coords = np.stack([0.1 * np.cos(angles), 0.1 * np.sin(angles)], axis=-1).astype(
        np.float32
    )

    print("Initializing FastAcousticPipeline (JIT Compiling...)")
    pipeline = FastAcousticPipeline(coords, sample_rate=sr)

    # Generate mock data
    # Scenario A: Plane (Coherent wave at 45 deg)
    t = np.arange(num_samples) / sr
    plane = np.zeros((channels, num_samples), dtype=np.float32)
    dir_x, dir_y = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
    delays = np.dot(coords, [dir_x, dir_y]) / 343.0
    base_plane = (
        np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        + np.random.randn(num_samples).astype(np.float32) * 0.1
    )
    for i in range(channels):
        # Time domain fractional shift approximation for mock creation
        shift_samples = int(delays[i] * sr)
        plane[i] = np.roll(base_plane, shift_samples)

    # Scenario B: Wind (Independent noise)
    wind = (np.random.randn(channels, num_samples) * 0.5).astype(np.float32)

    # Warmup Numba JIT (first run is slow)
    pipeline.process_chunk(wind, target_pan=45.0, target_tilt=90.0)

    print("\n--- Pipeline Benchmarking ---")

    # 100 iterations of Plane
    latencies = []
    scores_gain = []
    scores_phase = []
    for _ in range(100):
        start = time.perf_counter()
        audio_out, scores = pipeline.process_chunk(
            plane, target_pan=45.0, target_tilt=90.0
        )
        latencies.append((time.perf_counter() - start) * 1000)
        scores_gain.append(scores["array_gain"])
        scores_phase.append(scores["phase_variance_approx"])

    print("PLANE SCENARIO (Avg over 100 frames):")
    print(f"  Latency        : {np.mean(latencies):.2f} ms")
    print(f"  Array Gain     : {np.mean(scores_gain):.4f}")
    print(f"  Phase Variance : {np.mean(scores_phase):.4f}\n")

    assert np.mean(latencies) < 10.0, "Latency exceeded 10ms budget!"

    # 100 iterations of Wind
    latencies = []
    scores_gain = []
    scores_phase = []
    for _ in range(100):
        # Dynamically recreate wind to prevent caching
        w = (np.random.randn(channels, num_samples) * 0.5).astype(np.float32)
        start = time.perf_counter()
        audio_out, scores = pipeline.process_chunk(w, target_pan=45.0, target_tilt=90.0)
        latencies.append((time.perf_counter() - start) * 1000)
        scores_gain.append(scores["array_gain"])
        scores_phase.append(scores["phase_variance_approx"])

    print("WIND SCENARIO (Avg over 100 frames):")
    print(f"  Latency        : {np.mean(latencies):.2f} ms")
    print(f"  Array Gain     : {np.mean(scores_gain):.4f}")
    print(f"  Phase Variance : {np.mean(scores_phase):.4f}")


if __name__ == "__main__":
    main()
