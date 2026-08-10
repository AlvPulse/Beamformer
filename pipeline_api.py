import numpy as np
from numba_backend import NumbaDASBeamformer


class FastAcousticPipeline:
    """
    Ultra-low latency (<10ms) acoustic beamforming and physics scoring pipeline
    designed for one-line integration into complex systems.

    Uses CPU-optimized Numba backend for Fractional Delay DAS.
    """

    def __init__(self, sensor_coords: np.ndarray, sample_rate: int = 8000):
        """
        Initialize the pipeline once.

        Args:
            sensor_coords: [N, 2] numpy array of (x,y) microphone coordinates in meters.
            sample_rate: Sampling rate of the audio (default 8000).
        """
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
