import torch
import os
import random
import soundfile as sf
import numpy as np

os.makedirs('data', exist_ok=True)

configs = [
    (16, 8000, 1.5),
    (32, 16000, 2.0),
    (8, 48000, 1.0),
    (64, 8000, 0.5)
]

for i, (channels, sr, duration) in enumerate(configs):
    num_samples = int(sr * duration)
    t = np.arange(num_samples) / sr
    freq = 440.0 + random.uniform(-100, 100)

    # [channels, num_samples]
    signal = np.sin(2 * np.pi * freq * t)[np.newaxis, :]
    signal = np.repeat(signal, channels, axis=0)

    noise = np.random.randn(channels, num_samples) * 0.1
    signal = signal + noise

    # soundfile expects [num_samples, channels]
    signal = signal.T

    file_path = f"data/dummy_{i}_ch{channels}_sr{sr}.wav"
    sf.write(file_path, signal, sr)
    print(f"Saved {file_path}")
