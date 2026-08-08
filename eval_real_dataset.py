import os
import glob
import re
import argparse
import numpy as np
import soundfile as sf
import torch

from beamformer import ArrayGeometry, Beamformer


def parse_array_geometry(filepath: str) -> torch.Tensor:
    """Parses a text file with x=[...] and y=[...] lists into a coordinate tensor."""
    with open(filepath, "r") as f:
        content = f.read()

    x_match = re.search(r"x\s*=\s*\[(.*?)\]", content, re.DOTALL)
    y_match = re.search(r"y\s*=\s*\[(.*?)\]", content, re.DOTALL)

    if not x_match or not y_match:
        raise ValueError(
            f"Could not find valid x=[...] or y=[...] blocks in {filepath}"
        )

    x_str = x_match.group(1)
    y_str = y_match.group(1)

    x_coords = [float(val.strip()) for val in x_str.split(",") if val.strip()]
    y_coords = [float(val.strip()) for val in y_str.split(",") if val.strip()]

    if len(x_coords) != len(y_coords) or len(x_coords) == 0:
        raise ValueError("Geometry file contains mismatched or empty coordinates.")

    return torch.tensor(list(zip(x_coords, y_coords)), dtype=torch.float32)


def generate_circular_array(num_sensors: int, radius: float = 0.1) -> torch.Tensor:
    angles = torch.linspace(0, 2 * np.pi, num_sensors + 1)[:-1]
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)


def infer_label(filepath: str) -> str:
    """
    Infers whether a file contains 'wind' or an 'event' based on its path/filename.
    Adjust these keywords based on your specific dataset conventions.
    """
    path_lower = filepath.lower()
    wind_keywords = ["wind", "turbulence", "gust", "breeze", "outdoor_wind"]
    event_keywords = [
        "plane",
        "aircraft",
        "speech",
        "car",
        "vehicle",
        "event",
        "target",
        "demand",
        "esc50",
    ]

    if any(k in path_lower for k in wind_keywords):
        return "wind"
    if any(k in path_lower for k in event_keywords):
        return "event"
    return "unknown"


def evaluate_dataset(dataset_path: str, geom_path: str, chunk_duration: float = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating dataset at: {dataset_path} on {device}")

    wav_files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    wav_files = [
        f for f in wav_files if "_output.wav" not in f
    ]  # Ignore previously processed outputs

    if not wav_files:
        print("No .wav files found!")
        return

    # Load geometry if available
    base_coords = None
    if os.path.exists(geom_path):
        print(f"Loaded physical geometry from {geom_path}")
        base_coords = parse_array_geometry(geom_path)
    else:
        print(
            f"WARNING: Geometry file '{geom_path}' not found. Will guess dynamic circular arrays."
        )

    results = {
        "wind": {"sci": [], "gain": []},
        "event": {"sci": [], "gain": []},
        "unknown": {"sci": [], "gain": []},
    }

    for filepath in wav_files:
        label = infer_label(filepath)

        try:
            audio_np, sr = sf.read(filepath)
            if len(audio_np.shape) == 1:
                continue  # Skip mono files, beamforming requires multi-channel

            num_samples, num_channels = audio_np.shape

            # Match Geometry
            if base_coords is not None:
                if base_coords.shape[0] != num_channels:
                    # Truncate if audio has more channels than geometry
                    if num_channels > base_coords.shape[0]:
                        audio_np = audio_np[:, : base_coords.shape[0]]
                        num_channels = base_coords.shape[0]
                    else:
                        print(f"Skipping {filepath}: not enough channels for geometry.")
                        continue
                coords = base_coords
            else:
                coords = generate_circular_array(num_channels)

            geom = ArrayGeometry(coords.to(device))
            # Use MVDR for wind feature extraction, fast N_FFT for quick scanning
            bf = Beamformer(
                geom,
                sample_rate=sr,
                method="MVDR",
                diag_loading=1e-1,
                n_fft=256,
                hop_length=128,
            )

            chunk_size = int(sr * chunk_duration)
            num_chunks = num_samples // chunk_size

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio_np[start:end, :]

                # Skip silent chunks
                if np.max(np.abs(chunk)) < 1e-4:
                    continue

                signal = (
                    torch.from_numpy(chunk).float().T.to(device)
                )  # [channels, time]

                _, _, _, _, scores = bf.forward(
                    signal, target_pan=None, target_tilt=None
                )

                results[label]["sci"].append(scores["sci"])
                results[label]["gain"].append(scores["array_gain"])

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print("\n" + "=" * 50)
    print("DATASET EVALUATION RESULTS")
    print("=" * 50)

    for cls in ["wind", "event"]:
        sci_vals = results[cls]["sci"]
        gain_vals = results[cls]["gain"]

        n_chunks = len(sci_vals)
        if n_chunks == 0:
            print(f"Class: {cls.upper()} -> No data found.")
            continue

        sci_mean, sci_std = np.mean(sci_vals), np.std(sci_vals)
        gain_mean, gain_std = np.mean(gain_vals), np.std(gain_vals)

        print(f"Class: {cls.upper()} ({n_chunks} chunks analyzed)")
        print(f"  SCI        : {sci_mean:.4f} ± {sci_std:.4f}")
        print(f"  Array Gain : {gain_mean:.4f} ± {gain_std:.4f}")
        print("-" * 50)

    # Threshold Suggestion (if both classes exist)
    if len(results["wind"]["sci"]) > 0 and len(results["event"]["sci"]) > 0:
        wind_sci_mean = np.mean(results["wind"]["sci"])
        event_sci_mean = np.mean(results["event"]["sci"])
        suggested_threshold = (wind_sci_mean + event_sci_mean) / 2

        print("\n[Threshold Analysis]")
        print(f"Suggested SCI Threshold for Wind Rejection: {suggested_threshold:.4f}")

        wind_rejected = sum(
            1 for x in results["wind"]["sci"] if x < suggested_threshold
        )
        event_kept = sum(1 for x in results["event"]["sci"] if x >= suggested_threshold)

        wind_acc = (wind_rejected / len(results["wind"]["sci"])) * 100
        event_acc = (event_kept / len(results["event"]["sci"])) * 100

        print(f"  Wind successfully rejected : {wind_acc:.1f}%")
        print(f"  Events correctly preserved : {event_acc:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Beamformer Wind Rejection on Real Datasets"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Path to folder containing .wav files",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default="array_geometry.txt",
        help="Path to array_geometry.txt",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=1.0,
        help="Duration of chunks to analyze in seconds",
    )

    args = parser.parse_args()
    evaluate_dataset(args.dataset_path, args.geometry, args.chunk_duration)
