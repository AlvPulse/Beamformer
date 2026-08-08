import torch
import numpy as np
import time
from beamformer import ArrayGeometry, Beamformer

torch.set_num_threads(4)


def generate_circular_array(channels, radius=0.1):
    angles = np.linspace(0, 2 * np.pi, channels + 1)[:-1]
    coords = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
    return coords, angles


def generate_scenario(
    scenario_type, sr=8000, channels=16, duration=1.0, pan=45.0, tilt=90.0
):
    num_samples = int(sr * duration)
    t = np.arange(num_samples) / sr
    coords, _ = generate_circular_array(channels)

    signal = np.zeros((channels, num_samples))

    if scenario_type in ["plane", "distant_plane"]:
        # Generate base plane sound (mix of engine tones and broadband roar)
        base = np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)
        base += np.random.randn(num_samples) * 0.5

        # Distant plane: heavily low-pass filter it (simulating atmospheric absorption)
        if scenario_type == "distant_plane":
            # Simple low pass via FFT
            S = np.fft.rfft(base)
            freqs = np.fft.rfftfreq(num_samples, 1 / sr)
            S[freqs > 400] *= 0.01  # crush high frequencies
            base = np.fft.irfft(S, num_samples)

        # Apply TDOA plane wave delays
        pan_rad = np.deg2rad(pan)
        tilt_rad = np.deg2rad(tilt)
        dir_x = np.sin(tilt_rad) * np.cos(pan_rad)
        dir_y = np.sin(tilt_rad) * np.sin(pan_rad)
        delays = np.dot(coords, np.array([dir_x, dir_y])) / 343.0

        for i in range(channels):
            S = np.fft.rfft(base)
            freqs = np.fft.rfftfreq(num_samples, 1 / sr)
            S_delayed = S * np.exp(-1j * 2 * np.pi * freqs * delays[i])
            signal[i] = np.fft.irfft(S_delayed, num_samples)

        # Add ambient array noise
        snr_noise = 0.1 if scenario_type == "plane" else 0.8
        signal += np.random.randn(channels, num_samples) * snr_noise

    elif scenario_type == "wind":
        # Wind is local turbulent pressure fluctuations -> independent noise per microphone
        # usually dominant at low frequencies.
        for i in range(channels):
            noise = np.random.randn(num_samples)
            # Low pass the noise to simulate wind rumble
            S = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(num_samples, 1 / sr)
            S[freqs > 300] *= 0.1
            signal[i] = np.fft.irfft(S, num_samples)

    return torch.from_numpy(signal).float(), torch.from_numpy(coords).float()


def main():
    sr = 8000
    channels = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Wind Rejection Scoring Evaluation ---")
    print(f"Device: {device} | Channels: {channels} | SR: {sr}\n")

    scenarios = ["plane", "distant_plane", "wind"]

    for scenario in scenarios:
        print(f"Evaluating Scenario: [{scenario.upper()}]")
        signal, coords = generate_scenario(scenario, sr=sr, channels=channels)
        signal = signal.to(device)
        geom = ArrayGeometry(coords.to(device))

        # We test MVDR since we want to see how the wind scoring metrics behave directly from SCM
        bf = Beamformer(
            geom,
            sample_rate=sr,
            method="MVDR",
            diag_loading=1e-1,
            n_fft=128,
            hop_length=64,
        )

        # Warmup
        bf.forward(signal, target_pan=None, target_tilt=None)

        # Timed execution
        start = time.perf_counter()
        audio, spec, best_pan, best_tilt, scores = bf.forward(
            signal, target_pan=None, target_tilt=None
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

        print(f"  Latency    : {latency:.2f} ms")
        print(f"  Auto-Steer : Pan {best_pan:.1f}, Tilt {best_tilt:.1f}")
        print(f"  SCI Score  : {scores['sci']:.4f}")
        print(f"  Array Gain : {scores['array_gain']:.4f}\n")


if __name__ == "__main__":
    main()
