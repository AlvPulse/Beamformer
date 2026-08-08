import os
import glob
import argparse
import numpy as np
import soundfile as sf
import torch
import re

from beamformer import ArrayGeometry, Beamformer


def parse_array_geometry(filepath: str) -> torch.Tensor:
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        content = f.read()
    x_match = re.search(r"x\s*=\s*\[(.*?)\]", content, re.DOTALL)
    y_match = re.search(r"y\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not x_match or not y_match:
        return None
    x_coords = [
        float(val.strip()) for val in x_match.group(1).split(",") if val.strip()
    ]
    y_coords = [
        float(val.strip()) for val in y_match.group(1).split(",") if val.strip()
    ]
    if len(x_coords) != len(y_coords) or len(x_coords) == 0:
        return None
    return torch.tensor(list(zip(x_coords, y_coords)), dtype=torch.float32)


def generate_circular_array(num_sensors: int, radius: float = 0.1) -> torch.Tensor:
    angles = torch.linspace(0, 2 * np.pi, num_sensors + 1)[:-1]
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)


def main(input_dir, output_dir, geom_path, chunk_duration, sci_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Auto-Labeler on {device}...")

    wind_dir = os.path.join(output_dir, "wind")
    event_dir = os.path.join(output_dir, "event")
    os.makedirs(wind_dir, exist_ok=True)
    os.makedirs(event_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    base_coords = parse_array_geometry(geom_path)

    total_chunks = 0
    wind_count = 0
    event_count = 0

    for filepath in wav_files:
        print(f"Processing: {filepath}")
        try:
            audio_np, sr = sf.read(filepath)
            if len(audio_np.shape) == 1:
                print("  Skipping mono file.")
                continue

            num_samples, num_channels = audio_np.shape

            if base_coords is not None:
                if base_coords.shape[0] != num_channels:
                    if num_channels > base_coords.shape[0]:
                        audio_np = audio_np[:, : base_coords.shape[0]]
                        num_channels = base_coords.shape[0]
                    else:
                        print("  Skipping: not enough channels.")
                        continue
                coords = base_coords
            else:
                coords = generate_circular_array(num_channels)

            geom = ArrayGeometry(coords.to(device))
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

            file_basename = os.path.splitext(os.path.basename(filepath))[0]

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio_np[start:end, :]

                # Skip absolute silence
                if np.max(np.abs(chunk)) < 1e-4:
                    continue

                signal = torch.from_numpy(chunk).float().T.to(device)

                # Auto-steer to the loudest sound in this specific chunk
                audio_out, _, best_pan, best_tilt, scores = bf.forward(
                    signal, target_pan=None, target_tilt=None
                )
                sci = scores["sci"]

                out_np = audio_out.detach().cpu().numpy()

                # Classification Rule
                if sci < sci_threshold:
                    category = "wind"
                    wind_count += 1
                    target_dir = wind_dir
                else:
                    category = "event"
                    event_count += 1
                    target_dir = event_dir

                total_chunks += 1

                # Save the 1D beamformed clip so the user can listen to the classification
                out_name = f"{file_basename}_chunk{i:03d}_{category}_SCI{sci:.1f}_pan{int(best_pan)}.wav"
                sf.write(os.path.join(target_dir, out_name), out_np, sr)

        except Exception as e:
            print(f"  Error: {e}")

    print("\n--- Auto-Labeling Complete ---")
    print(f"Total Chunks Processed : {total_chunks}")
    print(f"Chunks labeled WIND    : {wind_count}")
    print(f"Chunks labeled EVENT   : {event_count}")
    print(f"Outputs saved to       : {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-Label and sort unlabelled multichannel wav files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./test",
        help="Folder with unlabelled .wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./organized_output",
        help="Folder to save sorted 1D chunks",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default="array_geometry.txt",
        help="Path to geometry txt file",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=1.0,
        help="Duration of chunks in seconds",
    )
    parser.add_argument(
        "--sci_threshold",
        type=float,
        default=3.0,
        help="SCI threshold to classify Wind vs Event",
    )

    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.geometry,
        args.chunk_duration,
        args.sci_threshold,
    )
