import torch
import time
import math
from typing import Tuple, Optional


class ArrayGeometry:
    """
    Handles the 2D sensor array coordinates and computes the Time Difference of Arrival (TDOA)
    and complex steering vectors for a given set of (pan, tilt) angles.
    """

    def __init__(self, sensor_coords: torch.Tensor, sound_speed: float = 343.0):
        """
        Initialize the array geometry.

        Args:
            sensor_coords: [N, 2] tensor containing the (x, y) coordinates of N sensors in meters.
            sound_speed: Speed of sound in meters per second (default: 343.0 for air).
        """
        self.sensor_coords = sensor_coords
        self.num_sensors = sensor_coords.shape[0]
        self.sound_speed = sound_speed

    def compute_steering_vectors(
        self, pan: torch.Tensor, tilt: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute complex steering vectors for given angles and frequencies.

        Args:
            pan: [num_angles] tensor of pan angles (azimuth) in degrees.
            tilt: [num_angles] tensor of tilt angles (polar angle) in degrees.
            freqs: [num_freqs] tensor of frequencies in Hz.

        Returns:
            steering_vectors: [num_angles, num_freqs, N] complex tensor.
        """
        pan_rad = torch.deg2rad(pan)
        tilt_rad = torch.deg2rad(tilt)

        # Ensure sensor coordinates are on the correct device
        device = pan.device
        if self.sensor_coords.device != device:
            self.sensor_coords = self.sensor_coords.to(device)

        dir_x = torch.sin(tilt_rad) * torch.cos(pan_rad)
        dir_y = torch.sin(tilt_rad) * torch.sin(pan_rad)

        directions = torch.stack([dir_x, dir_y], dim=-1)

        # delays = distances / speed of sound
        distances = torch.matmul(directions, self.sensor_coords.T)
        delays = distances / self.sound_speed

        omega = 2 * math.pi * freqs

        phase_args = delays.unsqueeze(1) * omega.unsqueeze(0).unsqueeze(2)
        steering_vectors = torch.exp(-1j * phase_args)

        return steering_vectors


class Beamformer:
    """
    Implements Delay-and-Sum (DAS) and Minimum Variance Distortionless Response (MVDR)
    beamforming in the frequency domain.
    """

    def __init__(
        self,
        array_geometry: ArrayGeometry,
        sample_rate: int = 8000,
        n_fft: int = 256,
        hop_length: int = 128,
        method: str = "DAS",
        scan_pan: Optional[torch.Tensor] = None,
        scan_tilt: Optional[torch.Tensor] = None,
        diag_loading: float = 1e-5,
    ):
        """
        Args:
            array_geometry: Instantiated ArrayGeometry object.
            sample_rate: Sampling rate of the audio (default: 8000).
            n_fft: STFT window size (default: 256 for ~32ms at 8kHz).
            hop_length: STFT hop length (default: 128 for ~16ms).
            method: 'DAS' or 'MVDR'.
            scan_pan: [num_scan_angles] tensor of pan angles for the spatial spectrum grid.
            scan_tilt: [num_scan_angles] tensor of tilt angles for the spatial spectrum grid.
            diag_loading: Diagonal loading factor for MVDR SCM stability.
        """
        self.array_geometry = array_geometry
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.method = method.upper()
        self.diag_loading = diag_loading

        if self.method not in ["DAS", "MVDR"]:
            raise ValueError("Method must be 'DAS' or 'MVDR'")

        # Create default scanning grid if not provided
        if scan_pan is None or scan_tilt is None:
            # Simple grid: pan from 0 to 90, tilt fixed to 90
            scan_pan = torch.linspace(0, 90, 91)
            scan_tilt = torch.full_like(scan_pan, 90.0)

        self.scan_pan = scan_pan
        self.scan_tilt = scan_tilt
        self.num_scan_angles = len(scan_pan)

        # Frequencies corresponding to rfft
        self.freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        # Precompute steering vectors for the scanning grid
        # shape: [num_scan_angles, num_freqs, N]
        self.scan_steering_vectors = self.array_geometry.compute_steering_vectors(
            self.scan_pan, self.scan_tilt, self.freqs
        )

    def _compute_scm(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the Spatial Covariance Matrix (SCM) from STFT.

        Args:
            X: [num_freqs, N, num_frames] complex tensor.

        Returns:
            R: [num_freqs, N, N] complex tensor.
        """
        num_frames = X.shape[-1]
        # shape: [num_freqs, N, N]
        R = torch.einsum("fnt,fmt->fnm", X, X.conj()) / num_frames

        # Diagonal loading for stability
        N = R.shape[-1]
        device = R.device
        eye_mat = torch.eye(N, dtype=R.dtype, device=device).unsqueeze(0)

        # We can scale the loading by the trace of R to make it scale-invariant
        trace_R = torch.einsum("fnn->f", R).real
        loading = self.diag_loading * trace_R.unsqueeze(-1).unsqueeze(-1) * eye_mat

        # If trace is 0 (e.g. silence), fallback to fixed epsilon
        fallback_loading = self.diag_loading * eye_mat
        loading = torch.where(
            trace_R.unsqueeze(-1).unsqueeze(-1) > 0, loading, fallback_loading
        )

        R_loaded = R + loading
        return R_loaded, R

    def _compute_weights(
        self, steering_vectors: torch.Tensor, R: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute beamformer weights.

        Args:
            steering_vectors: [num_angles, num_freqs, N] complex tensor.
            R: [num_freqs, N, N] complex SCM (required for MVDR).

        Returns:
            weights: [num_angles, num_freqs, N] complex tensor.
        """
        if self.method == "DAS":
            # Normalized DAS weights
            N = self.array_geometry.num_sensors
            weights = steering_vectors / N
            return weights

        elif self.method == "MVDR":
            assert R is not None, "SCM (R) must be provided for MVDR"

            # steering_vectors: [num_angles, num_freqs, N]
            # V: [num_freqs, N, num_angles]
            V = steering_vectors.permute(1, 2, 0)

            # Solve R * W_unnormalized = V
            # R: [num_freqs, N, N]
            # inv_R_V: [num_freqs, N, num_angles]
            inv_R_V = torch.linalg.solve(R, V)

            # Compute denominator: V^H * R^-1 * V
            # denom: [num_freqs, num_angles]
            denom = torch.einsum("fna,fna->fa", V.conj(), inv_R_V).real

            # Avoid division by zero or extremely small numbers
            denom = torch.clamp(denom, min=1e-9)

            # W = (R^-1 * V) / (V^H * R^-1 * V)
            # W_mvdr: [num_freqs, N, num_angles]
            W_mvdr = inv_R_V / denom.unsqueeze(1)

            # Permute back to [num_angles, num_freqs, N]
            weights = W_mvdr.permute(2, 0, 1)
            return weights

    def forward(
        self,
        signal: torch.Tensor,
        target_pan: Optional[float] = None,
        target_tilt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Applies beamforming to the input signal.

        Args:
            signal: [N, num_samples] tensor of multichannel audio data.
            target_pan: Target azimuth angle in degrees.
            target_tilt: Target polar angle in degrees.

        Returns:
            output_audio: [num_samples] tensor of single-channel reconstructed audio.
            spatial_spectrum: [num_scan_angles] tensor of broadband spatial power for all scan angles.
        """
        device = signal.device

        # Ensure our precomputed tensors are on the same device as the signal
        if self.scan_steering_vectors.device != device:
            self.scan_steering_vectors = self.scan_steering_vectors.to(device)
            self.freqs = self.freqs.to(device)

        # 1. STFT
        window = torch.hann_window(self.n_fft, device=device)
        # stft output shape: [N, num_freqs, num_frames]
        X = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        # Permute for easier batch processing: [num_freqs, N, num_frames]
        X = X.permute(1, 0, 2)

        # 2. Compute SCM (needed for MVDR, and useful for computing spatial spectrum power)
        R, R_raw = self._compute_scm(X)

        # 3. Compute Spatial Spectrum for the scanning grid
        # Weights for the scanning grid: [num_scan_angles, num_freqs, N]
        scan_weights = self._compute_weights(self.scan_steering_vectors, R)

        # Power spectrum P = W^H * R * W
        # scan_weights.conj(): [num_scan_angles, num_freqs, N]
        # R: [num_freqs, N, N]
        # scan_power: [num_scan_angles, num_freqs]
        scan_power = torch.einsum(
            "afn,fnm,afm->af", scan_weights.conj(), R, scan_weights
        ).real

        # Broadband power (sum across frequencies)
        broadband_spatial_spectrum = scan_power.sum(dim=1)

        # 4. Select Target Angle
        if target_pan is None or target_tilt is None:
            best_idx = torch.argmax(broadband_spatial_spectrum)
            target_pan = self.scan_pan[best_idx].item()
            target_tilt = self.scan_tilt[best_idx].item()

        target_pan_tensor = torch.tensor(
            [target_pan], dtype=torch.float32, device=device
        )
        target_tilt_tensor = torch.tensor(
            [target_tilt], dtype=torch.float32, device=device
        )

        # Target steering vector: [1, num_freqs, N]
        target_sv = self.array_geometry.compute_steering_vectors(
            target_pan_tensor, target_tilt_tensor, self.freqs
        ).to(device)

        # Target weights: [1, num_freqs, N]
        target_weights = self._compute_weights(target_sv, R)

        # Apply weights to the STFT signal
        # target_weights: [1, num_freqs, N] -> squeeze(0) -> [num_freqs, N]
        # X: [num_freqs, N, num_frames]
        # Y: [num_freqs, num_frames]
        w_target = target_weights.squeeze(0)
        Y = torch.einsum("fn,fnt->ft", w_target.conj(), X)

        # ISTFT to recover time-domain audio
        # ISTFT expects [channels, num_freqs, num_frames], so we unsqueeze
        output_audio = torch.istft(
            Y.unsqueeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=signal.shape[-1],
        ).squeeze(
            0
        )  # back to [num_samples]

        # --- Wind Rejection Features ---
        # 1. Spatial Coherence Index (SCI): Mean off-diagonal of the normalized SCM
        # R shape: [num_freqs, N, N].
        # Normalize R to coherence matrix C: C_ij = |R_ij|^2 / (R_ii * R_jj)

        # Power per channel per freq: [num_freqs, N]
        P_ch = torch.diagonal(R_raw, dim1=-2, dim2=-1).real.clamp(min=1e-9)

        # Denominator matrix for coherence: (P_i * P_j)
        # [num_freqs, N, 1] * [num_freqs, 1, N] -> [num_freqs, N, N]
        denom_coh = P_ch.unsqueeze(-1) * P_ch.unsqueeze(-2)

        # Magnitude Squared Coherence (MSC): [num_freqs, N, N]
        msc = (R_raw.abs() ** 2) / denom_coh.clamp(min=1e-9)

        # SNR-weighted SCI: Compute average coherence only over frequencies that contain significant energy.
        # This prevents high-frequency microphone hiss or out-of-band noise from dragging the score to zero.
        power_spectrum = torch.mean(P_ch, dim=1)  # [num_freqs]
        max_power = torch.max(power_spectrum)

        # Consider bins within 20dB (0.01 power ratio) of the peak energy
        valid_bins = power_spectrum > (max_power * 0.01)

        # Also enforce a minimum bound to ignore ultra-low sub-bass wind rumble (e.g. < 50Hz)
        valid_bins = valid_bins & (self.freqs > 50)

        if valid_bins.any():
            msc_mean = msc[valid_bins].mean(dim=0)
        else:
            msc_mean = msc.mean(dim=0)  # Fallback

        # Extract mean of off-diagonal elements (exclude self-coherence which is 1.0)
        N_ch = msc_mean.shape[0]
        off_diag_mask = ~torch.eye(N_ch, dtype=torch.bool, device=device)
        spatial_coherence_index = msc_mean[off_diag_mask].mean().item()

        # 2. Beamformer Consistency (Array Gain G = P_beam / P_raw)
        # P_raw is the average power of the raw microphones
        p_raw = torch.mean(signal**2).item()
        p_beam = torch.mean(output_audio**2).item()

        array_gain = p_beam / max(p_raw, 1e-9)

        # 3. Phase Variance
        w_t = target_weights.squeeze(0)
        X_aligned = X * w_t.conj().unsqueeze(-1)
        if valid_bins.any():
            X_active = X_aligned[valid_bins]
        else:
            X_active = X_aligned
        phase_vectors = X_active / (X_active.abs() + 1e-9)
        mean_phase_vector = phase_vectors.mean(dim=1)
        phase_variance = 1.0 - mean_phase_vector.abs().mean().item()

        # 4. Spectral Persistence
        P_frames = X_active.abs() ** 2
        P_frames_mean = P_frames.mean(dim=1)
        if P_frames_mean.shape[1] > 1:
            frame_t0 = P_frames_mean[:, :-1]
            frame_t1 = P_frames_mean[:, 1:]
            mu_0 = frame_t0.mean(dim=0, keepdim=True)
            mu_1 = frame_t1.mean(dim=0, keepdim=True)
            num = ((frame_t0 - mu_0) * (frame_t1 - mu_1)).sum(dim=0)
            den = torch.sqrt(
                ((frame_t0 - mu_0) ** 2).sum(dim=0)
                * ((frame_t1 - mu_1) ** 2).sum(dim=0)
            )
            persistence = (num / (den + 1e-9)).mean().item()
        else:
            persistence = 0.0

        # 5. Nullspace Energy
        target_v = target_sv.squeeze(0)
        if valid_bins.any():
            R_active = R_raw[valid_bins]
            v_active = target_v[valid_bins]
        else:
            R_active = R_raw
            v_active = target_v

        total_energy = torch.diagonal(R_active, dim1=-2, dim2=-1).real.sum(dim=-1)
        v_norm = (v_active.abs() ** 2).sum(dim=-1)
        manifold_energy = torch.einsum(
            "fn,fnm,fm->f", v_active.conj(), R_active, v_active
        ).real / (v_norm + 1e-9)
        nullspace_energy = (
            ((total_energy - manifold_energy) / (total_energy + 1e-9))
            .clamp(min=0.0, max=1.0)
            .mean()
            .item()
        )

        # Package scores
        wind_scores = {
            "sci": spatial_coherence_index,
            "array_gain": array_gain,
            "phase_variance": phase_variance,
            "persistence": persistence,
            "nullspace": nullspace_energy,
        }

        return (
            output_audio,
            broadband_spatial_spectrum,
            target_pan,
            target_tilt,
            wind_scores,
        )


def main():
    print("Initializing geometry and generating dummy data...")
    # 16 sensors distributed in a circle of radius 0.1m
    N = 16
    angles = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    r = 0.1
    coords = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)

    geom = ArrayGeometry(coords)

    sample_rate = 8000
    chunk_size = 8000  # 1 second
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy data [N, chunk_size]
    signal = torch.randn(N, chunk_size, device=device)

    # Target pan and tilt
    target_pan = 45.0
    target_tilt = 90.0

    # Setup scanning grid (0 to 90 degrees pan, 90 degrees tilt)
    scan_pan = torch.linspace(0, 90, 91)
    scan_tilt = torch.full_like(scan_pan, 90.0)

    print("\n--- Benchmarking DAS ---")
    das_bf = Beamformer(
        geom,
        sample_rate=sample_rate,
        method="DAS",
        scan_pan=scan_pan,
        scan_tilt=scan_tilt,
    )

    # Warmup
    for _ in range(5):
        das_bf.forward(signal, target_pan, target_tilt)

    start_time = time.perf_counter()
    num_runs = 50
    for _ in range(num_runs):
        audio_out, spectrum, best_pan, best_tilt, _ = das_bf.forward(
            signal, target_pan, target_tilt
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()

    das_latency_ms = ((end_time - start_time) / num_runs) * 1000
    print(f"DAS Average Latency: {das_latency_ms:.2f} ms")

    print("\n--- Benchmarking MVDR ---")
    mvdr_bf = Beamformer(
        geom,
        sample_rate=sample_rate,
        method="MVDR",
        scan_pan=scan_pan,
        scan_tilt=scan_tilt,
    )

    # Warmup
    for _ in range(5):
        mvdr_bf.forward(signal, target_pan, target_tilt)

    start_time = time.perf_counter()
    for _ in range(num_runs):
        audio_out, spectrum, best_pan, best_tilt, _ = mvdr_bf.forward(
            signal, target_pan, target_tilt
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()

    mvdr_latency_ms = ((end_time - start_time) / num_runs) * 1000
    print(f"MVDR Average Latency: {mvdr_latency_ms:.2f} ms")

    print(
        f"\nOutputs shape -> Audio: {audio_out.shape}, Spatial Spectrum: {spectrum.shape}"
    )


if __name__ == "__main__":
    main()


def get_optimal_das_beamformer(
    geom: ArrayGeometry, sample_rate: int, num_channels: int, device: torch.device
):
    """
    Smart dispatcher for DAS beamforming.
    - If CUDA is available, always use PyTorch Frequency Domain.
    - If CPU only and num_channels is large (e.g., >32), use Numba Time Domain for lower latency.
    - Otherwise, use PyTorch Frequency Domain.
    """
    if device.type == "cuda":
        print("  -> Smart Dispatch: Using PyTorch (CUDA)")
        return Beamformer(geom, sample_rate=sample_rate, method="DAS")

    if num_channels >= 32:
        try:
            from numba_backend import NumbaDASBeamformer

            print("  -> Smart Dispatch: Using Numba (CPU Optimized)")
            return NumbaDASBeamformer(geom.sensor_coords.cpu().numpy(), sample_rate)
        except ImportError:
            print("  -> Smart Dispatch: Numba not found, falling back to PyTorch")
            pass

    print("  -> Smart Dispatch: Using PyTorch (CPU)")
    return Beamformer(geom, sample_rate=sample_rate, method="DAS")
