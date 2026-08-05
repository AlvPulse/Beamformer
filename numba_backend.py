import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _das_parallel(signal, delay_samples):
    N, n_samp = signal.shape
    # Hardcoded max threads or dynamic allocation
    # Numba prange will distribute iterations
    out = np.zeros(n_samp, dtype=np.float32)
    # Using atomic operations or thread-local accumulation is tricky in pure Numba without lock
    # An easier parallelization strategy for DAS is to parallelize over time or sensors.
    # Parallelizing over sensors is fast:

    max_th = min(N, 128)
    local = np.zeros((max_th, n_samp), dtype=np.float32)
    for i in prange(N):
        tid = i % max_th
        shift = delay_samples[i]
        sig_i = signal[i]
        for t in range(n_samp):
            orig = t + shift
            if orig < 0 or orig >= n_samp - 1:
                continue
            floor = int(orig)
            frac = orig - floor
            local[tid, t] += (1.0 - frac) * sig_i[floor] + frac * sig_i[floor + 1]

    # Reduction
    for tid in range(max_th):
        for t in range(n_samp):
            out[t] += local[tid, t]

    return out / N


class NumbaDASBeamformer:
    """
    Low-latency / offline Delay-and-Sum beamformer using Numba.
    """

    def __init__(self, sensor_coords: np.ndarray, sample_rate: int):
        self.coords = np.asarray(sensor_coords, dtype=np.float32)
        self.N = self.coords.shape[0]
        self.sr = sample_rate
        self.c = 343.0
        self._delay_samples = None

    def set_direction(self, pan: float, tilt: float):
        pan_r = math.radians(pan)
        tilt_r = math.radians(tilt)
        dir_x = math.sin(tilt_r) * math.cos(pan_r)
        dir_y = math.sin(tilt_r) * math.sin(pan_r)
        direction = np.array([dir_x, dir_y], dtype=np.float32)
        # distances = +dot for incoming wave from direction, but standard convention
        distances = np.dot(self.coords, direction)

        # Calculate delays relative to origin (like PyTorch implementation)
        self._delay_samples = (distances / self.c * self.sr).astype(np.float32)

    def process_chunk(self, signal: np.ndarray) -> np.ndarray:
        """
        signal : ndarray (channels, chunk_size) float32
        Returns beamformed chunk (chunk_size,) float32.
        """
        if self._delay_samples is None:
            raise RuntimeError("Call set_direction() first")
        # Ensure contiguous arrays for Numba
        sig = np.ascontiguousarray(signal.astype(np.float32))
        out = _das_parallel(sig, self._delay_samples)
        return out
