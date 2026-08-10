import numpy as np
import re
import os
from typing import Union
from numba_backend import NumbaDASBeamformer

def load_geometry(filepath: str) -> np.ndarray:
    """Read x=[...] y=[...] and return (N,2) float32 array in metres."""
    with open(filepath, 'r') as f:
        content = f.read()
    xm = re.search(r'x\s*=\s*\[(.*?)\]', content, re.DOTALL)
    ym = re.search(r'y\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not xm or not ym:
        raise ValueError("File must contain x=[...] and y=[...]")
    x = [float(v.strip()) for v in xm.group(1).split(',') if v.strip()]
    y = [float(v.strip()) for v in ym.group(1).split(',') if v.strip()]
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Coordinate lists empty or mismatched")
    return np.column_stack((x, y)).astype(np.float32)

class FastAcousticPipeline:
    """
    Ultra-low latency (<10ms) acoustic beamforming and physics scoring pipeline
    designed for one-line integration into complex systems.

    Uses CPU-optimized Numba backend for Fractional Delay DAS.
    """

    def __init__(self, geometry_source: Union[np.ndarray, str], sample_rate: int = 8000):
        """
        Initialize the pipeline once.

        Args:
            geometry_source: Either an [N, 2] numpy array, or a filepath string to array_geometry.txt
            sample_rate: Sampling rate of the audio (default 8000).
        """
        if isinstance(geometry_source, str):
            if not os.path.exists(geometry_source):
                raise FileNotFoundError(f"Geometry file not found: {geometry_source}")
            sensor_coords = load_geometry(geometry_source)
        else:
            sensor_coords = geometry_source

        # Instantiate the fast Numba DAS engine
        self.beamformer = NumbaDASBeamformer(sensor_coords, sample_rate)
        self.N = sensor_coords.shape[0]

    def process_chunk(
        self, audio_chunk: np.ndarray, target_pan: float, target_tilt: float
    ):
        """
        Processes a raw multichannel audio chunk in real-time.

        Args:
            audio_chunk: [num_channels, num_samples] float32 numpy array.
            target_pan: Target azimuth angle in degrees.
            target_tilt: Target polar angle in degrees.

        Returns:
            beamformed_audio: [num_samples] float32 1D numpy array.
            scores: Dictionary of physics-based acoustic features for wind rejection.
        """
        # 1. Update steering delays (extremely fast math calculation)
        self.beamformer.set_direction(target_pan, target_tilt)

        # 2. Run Numba parallel fractional-delay DAS
        # Returns [num_samples]
        beamformed_audio = self.beamformer.process_chunk(audio_chunk)

        # 3. Fast Physics Feature Extraction (Time-Domain)
        # To guarantee <10ms, we compute Array Gain directly in the time domain.
        # Wind cannot be coherently aligned by DAS, so its gain is very low.
        p_raw = np.mean(audio_chunk**2)
        p_beam = np.mean(beamformed_audio**2)
        array_gain = float(p_beam / max(p_raw, 1e-9))

        # Fast Phase Variance (approximated via time-domain cross-correlation with the beam)
        # If signals are perfectly aligned (plane wave), they correlate heavily with the mean beam.
        # If they are random wind, correlation is near zero.
        # We use a normalized dot product.
        cross_corrs = np.zeros(self.N, dtype=np.float32)
        beam_norm = np.sqrt(p_beam) + 1e-9
        for i in range(self.N):
            mic_p = np.mean(audio_chunk[i] ** 2)
            mic_norm = np.sqrt(mic_p) + 1e-9
            # Align the mic (approximate by just comparing to the output beam which is already aligned)
            # A true plane wave implies all shifted mics equal the beam.
            cross_corrs[i] = np.mean(audio_chunk[i] * beamformed_audio) / (
                mic_norm * beam_norm
            )

        # Variance of the correlations
        # 1.0 means perfect alignment, 0.0 means total noise
        alignment_score = float(np.mean(cross_corrs))
        phase_variance_approx = 1.0 - abs(alignment_score)

        scores = {
            "array_gain": array_gain,
            "phase_variance_approx": phase_variance_approx,
        }

        return beamformed_audio, scores
