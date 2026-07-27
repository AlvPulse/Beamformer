import os
import glob
import time
import math
import re
import torch
import soundfile as sf
import numpy as np
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

    if len(x_coords) != len(y_coords):
        raise ValueError(
            f"Mismatched number of x ({len(x_coords)}) and y ({len(y_coords)}) coordinates."
        )
    if len(x_coords) == 0:
        raise ValueError("Geometry file contains empty coordinates.")

    return torch.tensor(list(zip(x_coords, y_coords)), dtype=torch.float32)


def generate_circular_array(num_sensors: int, radius: float = 0.1) -> torch.Tensor:
    """Generates a uniform circular array of sensors."""
    angles = torch.linspace(0, 2 * math.pi, num_sensors + 1)[:-1]
    coords = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1
    )
    return coords


def process_file(file_path: str, device: torch.device):
    print(f"\nProcessing: {file_path}")

    # Load audio
    # audio is typically returned as [num_samples, channels]
    audio_np, sr = sf.read(file_path)

    # Handle single channel just in case, though beamforming needs multi
    if len(audio_np.shape) == 1:
        audio_np = audio_np[:, np.newaxis]

    num_samples, num_channels = audio_np.shape

    # Convert to torch tensor and shape [channels, num_samples]
    signal = torch.from_numpy(audio_np).float().T.to(device)

    # Initialize array geometry
    geom_file = "array_geometry.txt"
    if os.path.exists(geom_file):
        print(f"  Found {geom_file}, parsing real coordinates...")
        coords = parse_array_geometry(geom_file)
        if coords.shape[0] != num_channels:
            print(
                f"  WARNING: Geometry file has {coords.shape[0]} sensors but audio has {num_channels} channels."
            )
            if num_channels > coords.shape[0]:
                print(
                    f"  -> Truncating audio to match geometry ({coords.shape[0]} channels)"
                )
                signal = signal[: coords.shape[0], :]
                num_channels = coords.shape[0]
            else:
                print(
                    f"  -> Skipping file due to missing coordinates for {num_channels} channels."
                )
                return
    else:
        print(
            f"  No {geom_file} found, using dynamic circular array for {num_channels} channels."
        )
        coords = generate_circular_array(num_channels)
    geom = ArrayGeometry(coords)

    # --- Benchmark DAS ---
    das_bf = Beamformer(geom, sample_rate=sr, method="DAS")

    # Warmup
    das_bf.forward(signal, target_pan=None, target_tilt=None)

    start_time = time.perf_counter()
    das_audio, _, das_pan, das_tilt = das_bf.forward(
        signal, target_pan=None, target_tilt=None
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    das_latency = (time.perf_counter() - start_time) * 1000

    # --- Benchmark MVDR ---
    mvdr_bf = Beamformer(geom, sample_rate=sr, method="MVDR")

    # Warmup
    mvdr_bf.forward(signal, target_pan=None, target_tilt=None)

    start_time = time.perf_counter()
    mvdr_audio, _, mvdr_pan, mvdr_tilt = mvdr_bf.forward(
        signal, target_pan=None, target_tilt=None
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    mvdr_latency = (time.perf_counter() - start_time) * 1000

    print(f"  Channels: {num_channels} | SR: {sr} | Samples: {num_samples}")
    print(
        f"  DAS Latency : {das_latency:.2f} ms | Auto-steered to Pan: {das_pan:.1f}, Tilt: {das_tilt:.1f}"
    )
    print(
        f"  MVDR Latency: {mvdr_latency:.2f} ms | Auto-steered to Pan: {mvdr_pan:.1f}, Tilt: {mvdr_tilt:.1f}"
    )

    # Save outputs
    # Need to convert back to CPU and numpy, and to [num_samples] or [num_samples, 1]
    das_out_np = das_audio.detach().cpu().numpy()
    mvdr_out_np = mvdr_audio.detach().cpu().numpy()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.dirname(file_path)

    sf.write(os.path.join(out_dir, f"{base_name}_DAS_output.wav"), das_out_np, sr)
    sf.write(os.path.join(out_dir, f"{base_name}_MVDR_output.wav"), mvdr_out_np, sr)
    print(f"  Saved output audio to {out_dir}/")


def main():
    data_dir = "data"
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))

    # Filter out files that we just generated as output in previous runs
    wav_files = [f for f in wav_files if "_output.wav" not in f]

    if not wav_files:
        print("No .wav files found in data/ folder!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")

    for wav_file in wav_files:
        process_file(wav_file, device)


if __name__ == "__main__":
    main()
