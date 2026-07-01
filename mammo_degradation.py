"""
mammo_degradation.py
====================
Blind degradation module for medical imaging (mammographies).

Implements the Med-BSR pipeline (Shao et al., IET Image Processing 2023)
adapted for 2D grayscale images, featuring the 6 downsampling methods
from the MST framework.

Full HR → LR Degradation:
  1. Blur   : Isotropic Gaussian (B_iso) OR Anisotropic Gaussian (B_aniso)
  2. Noise  : Additive Gaussian with 3 variants (grayscale / per-channel / generalized)
  3. Down   : One of the 6 MST methods (Nearest, Bilinear, Bicubic, Lanczos, Box, Hamming)

These three factors are combined in a RANDOM ORDER (random select/combine strategy),
generating a wide degradation space that prevents bias towards a single type.

Scale parameters following the paper:
  ×2: sigma_iso ∈ [0.1, 2.4]  |  aniso axes ∈ [0.5, 6.0]
  ×4: sigma_iso ∈ [0.1, 2.8]  |  aniso axes ∈ [0.5, 8.0]
  kernel sizes: 7×7, 9×9, 11×11, 13×13, 15×15, 17×17, 19×19, 21×21
  angle ∈ [0, pi]
  noise sigma  : σ ∈ {1/255, 2/255, …, 25/255}

STANDALONE USAGE (for external validation scripts):
  from mammo_degradation import MammoDegradation

  deg = MammoDegradation(scale=4, seed=42)
  lr = deg(hr_patch)                    # np.ndarray float32 [H,W,3] → [H/4,W/4,3]
  lr = deg(hr_patch, method_idx=2)      # force Bicubic (idx 0-5)
  lr = deg(hr_patch, order=[1,2,0])     # force order: noise→down→blur

  # Iterate with a fixed method (round-robin validation):
  deg_rr = MammoDegradation(scale=4, seed=0)
  for i, patch in enumerate(patches):
      lr = deg_rr(patch, method_idx=i % 6)

INTEGRATION IN tf.data PIPELINE:
  # See finetune_mammo_SR.py — used via tf.py_function.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
# MST Constants (6 downsampling methods)
# ─────────────────────────────────────────────────────────────────────────────
MST_FILTERS = [
    Image.NEAREST,    # 0 — Nearest Neighbor
    Image.BILINEAR,   # 1 — Bilinear
    Image.BICUBIC,    # 2 — Bicubic
    Image.LANCZOS,    # 3 — Lanczos (antialiased sinc)
    Image.BOX,        # 4 — Box (area average)
    Image.HAMMING,    # 5 — Hamming
]
MST_NAMES = ["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"]
N_MST = len(MST_FILTERS)

# Valid kernel sizes (paper: 7×7 … 21×21, odd integers)
KERNEL_SIZES = [7, 9, 11, 13, 15, 17, 19, 21]

# Sigma levels for noise: 1/255 … 25/255
NOISE_SIGMAS = [i / 255.0 for i in range(1, 26)]

# Probabilities for the 3 types of Gaussian noise (paper: 2/5, 2/5, 1/5)
NOISE_PROBS = [2 / 5, 2 / 5, 1 / 5]


# ─────────────────────────────────────────────────────────────────────────────
# Scale-Dependent Parameters (paper, Section 3.2.1)
# ─────────────────────────────────────────────────────────────────────────────
_SCALE_PARAMS = {
    2: dict(sigma_iso_range=(0.1, 2.4), axis_range=(0.5, 6.0)),
    4: dict(sigma_iso_range=(0.1, 2.8), axis_range=(0.5, 8.0)),
}


def _get_scale_params(scale: int) -> dict:
    """Retrieves degradation configuration ranges for a given scale factor."""
    if scale not in _SCALE_PARAMS:
        # For undefined scales, linearly interpolate/default towards ×4
        return _SCALE_PARAMS[4]
    return _SCALE_PARAMS[scale]


# ─────────────────────────────────────────────────────────────────────────────
# Low-Level Functions (also exportable)
# ─────────────────────────────────────────────────────────────────────────────

def blur_isotropic(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Isotropic Gaussian blur (B_iso).
    Applies scipy's gaussian_filter channel by channel.
    
    img  : float32 [H, W, 3]  ∈ [0, 1]
    sigma: standard deviation of the Gaussian kernel
    """
    out = np.stack(
        [gaussian_filter(img[:, :, c], sigma=sigma, mode="reflect") for c in range(3)],
        axis=-1,
    )
    return out.clip(0.0, 1.0).astype(np.float32)


def _gaussian_kernel_2d(
    ksize: int, sigma_x: float, sigma_y: float, angle_deg: float
) -> np.ndarray:
    """
    Generates a 2D anisotropic Gaussian kernel of size ksize×ksize.
    The kernel is rotated by 'angle_deg' degrees.
    
    Returns a normalized float32 array (sum = 1).
    """
    k = ksize // 2
    xs = np.arange(-k, k + 1, dtype=np.float64)
    ys = np.arange(-k, k + 1, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, ys)
    angle = np.deg2rad(angle_deg)
    
    # Rotate coordinates
    x_rot =  x_grid * np.cos(angle) + y_grid * np.sin(angle)
    y_rot = -x_grid * np.sin(angle) + y_grid * np.cos(angle)
    
    kernel = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    kernel = (kernel / kernel.sum()).astype(np.float32)
    return kernel


def blur_anisotropic(
    img: np.ndarray, ksize: int, sigma_x: float, sigma_y: float, angle_deg: float
) -> np.ndarray:
    """
    Anisotropic Gaussian blur (B_aniso).
    
    img      : float32 [H, W, 3]  ∈ [0, 1]
    ksize    : kernel size (one of KERNEL_SIZES)
    sigma_x  : principal axis standard deviation
    sigma_y  : secondary axis standard deviation
    angle_deg: rotation angle in degrees [0, 180)
    """
    kernel = _gaussian_kernel_2d(ksize, sigma_x, sigma_y, angle_deg)
    out = np.stack(
        [convolve(img[:, :, c], kernel, mode="reflect") for c in range(3)],
        axis=-1,
    )
    return out.clip(0.0, 1.0).astype(np.float32)


def add_gaussian_noise(
    img: np.ndarray, sigma: float, noise_type: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Adds additive Gaussian noise (N_g).

    noise_type (paper, probabilities 2/5, 2/5, 1/5):
      0 → Greyscale additive: same noise map replicated across all 3 channels (Σ = σ²·1)
      1 → Additive per-channel: independent noise per channel             (Σ = σ²·I)
      2 → Generalized: variable sigma per channel                          (alternative Σ)

    img        : float32 [H, W, 3]  ∈ [0, 1]
    sigma      : noise level (in [1/255, 25/255])
    noise_type : 0, 1, or 2
    rng        : numpy Generator for reproducibility
    """
    h, w, c = img.shape
    if noise_type == 0:
        # Replicate identical scalar noise across all 3 channels
        n_map = rng.normal(0.0, sigma, (h, w)).astype(np.float32)
        noise = np.stack([n_map] * c, axis=-1)
    elif noise_type == 1:
        # Independent noise map per channel
        noise = rng.normal(0.0, sigma, (h, w, c)).astype(np.float32)
    else:
        # Generalized noise: different sigma per channel (scaled by U[0.5, 1.5])
        noise = np.stack(
            [rng.normal(0.0, sigma * rng.uniform(0.5, 1.5), (h, w)) for _ in range(c)],
            axis=-1,
        ).astype(np.float32)
    return (img + noise).clip(0.0, 1.0).astype(np.float32)


def mst_downsample(
    img: np.ndarray, lr_h: int, lr_w: int, method_idx: int
) -> np.ndarray:
    """
    Downsamples img to target dimensions (lr_h, lr_w) using one of the 6 MST methods via PIL.
    
    img        : float32 [H, W, 3]  ∈ [0, 1]
    method_idx : 0=nearest | 1=bilinear | 2=bicubic | 3=lanczos | 4=box | 5=hamming
    Returns    : float32 [lr_h, lr_w, 3]  ∈ [0, 1]
    """
    if method_idx < 0 or method_idx >= N_MST:
        raise ValueError(f"method_idx must be in [0, {N_MST - 1}], received: {method_idx}")
    uint8_img = (img * 255.0).clip(0, 255).astype(np.uint8)
    pil_hr = Image.fromarray(uint8_img)
    pil_lr = pil_hr.resize((lr_w, lr_h), resample=MST_FILTERS[method_idx])
    return np.asarray(pil_lr, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Main Class Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MammoDegradation:
    """
    Blind degradation pipeline for mammography images.

    Implements the Med-BSR model (Shao et al. 2023) featuring 3 components:
      Blur (isotropic or anisotropic) → Noise (Gaussian) → Downsampling (6 MST methods)
    executed in a RANDOM ORDER (random select/combine strategy).

    Parameters
    ----------
    scale : int
        Scale factor (2 or 4). Determines sigma and axis bounds.
    seed  : int | None
        Seed for numpy.random.Generator and random.Random.
        None allows non-deterministic behavior.

    Example
    -------
    >>> deg = MammoDegradation(scale=4, seed=0)
    >>> lr = deg(hr_patch)                    # fully randomized degradation
    >>> lr = deg(hr_patch, method_idx=2)      # force Bicubic downsampling
    >>> lr = deg(hr_patch, order=[2, 0, 1])   # force sequence: down→blur→noise
    """

    def __init__(self, scale: int = 4, seed: int | None = None):
        if scale not in (2, 4):
            raise ValueError(f"scale must be either 2 or 4, received: {scale}")
        self.scale = scale
        self._params = _get_scale_params(scale)
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    # ── Hyperparameter Random Sampling Helpers ────────────────────────────

    def _sample_blur_iso(self) -> dict:
        lo, hi = self._params["sigma_iso_range"]
        sigma = self._rng.uniform(lo, hi)
        return {"type": "iso", "sigma": float(sigma)}

    def _sample_blur_aniso(self) -> dict:
        lo, hi = self._params["axis_range"]
        sigma_x = self._rng.uniform(lo, hi)
        sigma_y = self._rng.uniform(lo, hi)
        angle = self._rng.uniform(0.0, 180.0)
        ksize = self._py_rng.choice(KERNEL_SIZES)
        return {
            "type": "aniso",
            "ksize": ksize,
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "angle_deg": float(angle),
        }

    def _sample_noise(self) -> dict:
        sigma = self._py_rng.choice(NOISE_SIGMAS)
        noise_type = self._rng.choice([0, 1, 2], p=NOISE_PROBS)
        return {"sigma": float(sigma), "noise_type": int(noise_type)}

    def _sample_downsample(self) -> dict:
        method_idx = self._py_rng.randint(0, N_MST - 1)
        return {"method_idx": method_idx}

    # ── Individual Component Applicators ──────────────────────────────────

    def _apply_blur(self, img: np.ndarray, params: dict) -> np.ndarray:
        if params["type"] == "iso":
            return blur_isotropic(img, params["sigma"])
        else:
            return blur_anisotropic(
                img,
                ksize=params["ksize"],
                sigma_x=params["sigma_x"],
                sigma_y=params["sigma_y"],
                angle_deg=params["angle_deg"],
            )

    def _apply_noise(self, img: np.ndarray, params: dict) -> np.ndarray:
        return add_gaussian_noise(
            img,
            sigma=params["sigma"],
            noise_type=params["noise_type"],
            rng=self._rng,
        )

    def _apply_downsample(self, img: np.ndarray, params: dict) -> np.ndarray:
        lr_h = img.shape[0] // self.scale
        lr_w = img.shape[1] // self.scale
        return mst_downsample(img, lr_h, lr_w, params["method_idx"])

    # ── Public API ────────────────────────────────────────────────────────

    def __call__(
        self,
        hr: np.ndarray,
        method_idx: int | None = None,
        order: Sequence[int] | None = None,
    ) -> np.ndarray:
        """
        Degrades an HR patch down to an LR patch.

        Parameters
        ----------
        hr         : np.ndarray float32 [H, W, 3]  ∈ [0, 1]
        method_idx : int | None
            If provided, forces a specific downsampling method (0–5).
            If None, the choice is selected randomly.
        order      : Sequence[int] | None
            Permutation of [0, 1, 2] specifying the execution sequence:
              0 = blur  |  1 = noise  |  2 = downsample
            Example: [2, 0, 1] → downsample → blur → noise
            If None, the order is randomized.

        Returns
        -------
        lr : np.ndarray float32 [H//scale, W//scale, 3]  ∈ [0, 1]
        """
        if hr.ndim != 3 or hr.shape[2] != 3:
            raise ValueError(f"hr must have shape [H, W, 3], received: {hr.shape}")

        # Sample degradation hyperparameters
        blur_type = self._py_rng.choice(["iso", "aniso"])
        blur_p = self._sample_blur_iso() if blur_type == "iso" else self._sample_blur_aniso()
        noise_p = self._sample_noise()
        down_p = self._sample_downsample()

        # Enforce specific downsampling method if requested
        if method_idx is not None:
            if not (0 <= method_idx < N_MST):
                raise ValueError(f"method_idx must fall within [0, {N_MST-1}]")
            down_p["method_idx"] = method_idx

        # Randomize or validate execution order
        if order is None:
            order_list = list(range(3))
            self._py_rng.shuffle(order_list)
        else:
            order_list = list(order)
            if sorted(order_list) != [0, 1, 2]:
                raise ValueError("order must be a valid permutation of [0, 1, 2]")

        # Execute degradation pipeline in scheduled sequence
        img = hr.astype(np.float32)
        for step in order_list:
            if step == 0:
                img = self._apply_blur(img, blur_p)
            elif step == 1:
                img = self._apply_noise(img, noise_p)
            else:
                img = self._apply_downsample(img, down_p)

        return img.clip(0.0, 1.0).astype(np.float32)

    def degrade_batch(
        self,
        hr_batch: np.ndarray,
        method_idx: int | None = None,
        order: Sequence[int] | None = None,
    ) -> np.ndarray:
        """
        Applies the degradation pipeline to an entire batch of patches.
        
        hr_batch : float32 [N, H, W, 3]
        Returns  : float32 [N, H//scale, W//scale, 3]

        Note: Each patch receives INDEPENDENT degradation profiles
        (unique blur params, varying noise levels, and distinct MST selections).
        Provide a constant 'method_idx' if uniform scaling is preferred.
        """
        return np.stack(
            [self(hr_batch[i], method_idx=method_idx, order=order)
             for i in range(hr_batch.shape[0])],
            axis=0,
        )

    # ── Metadata Property Accessors ───────────────────────────────────────

    @staticmethod
    def method_name(idx: int) -> str:
        """Returns the text string identifier of an MST filter index (0–5)."""
        return MST_NAMES[idx]

    @property
    def n_methods(self) -> int:
        """Returns total number of supported downsampling filters."""
        return N_MST

    def __repr__(self) -> str:
        return (
            f"MammoDegradation(scale={self.scale}, "
            f"sigma_iso={self._params['sigma_iso_range']}, "
            f"axis={self._params['axis_range']})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper for tf.py_function Wrapper Integration (used by finetune_mammo_SR.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_degradation_fn(scale: int, seed: int):
    """
    Factory function wrapping MammoDegradation for direct execution in tf.py_function.
    Returns an executable callable with signature: (hr_np, method_idx) -> lr_np.

    Usage inside tf.data:
        _degrade = make_degradation_fn(scale=4, seed=42)

        def _py(hr_t, method_t):
            return _degrade(hr_t.numpy(), int(method_t.numpy()))

        lr = tf.py_function(_py, [hr, method_seed], tf.float32)
    """
    deg = MammoDegradation(scale=scale, seed=seed)

    def _fn(hr_np: np.ndarray, method_idx: int) -> np.ndarray:
        return deg(hr_np, method_idx=method_idx if method_idx >= 0 else None)

    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# Fast Execution Smoke-Test Module
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    target_scale = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"\n{'='*55}")
    print(f"  MammoDegradation — Smoke Test  (scale={target_scale})")
    print(f"{'='*55}")

    rng_np = np.random.default_rng(0)
    hr_mock = rng_np.random((128, 128, 3)).astype(np.float32)
    print(f"  HR patch: {hr_mock.shape}  min={hr_mock.min():.4f}  max={hr_mock.max():.4f}")

    deg_test = MammoDegradation(scale=target_scale, seed=99)
    print(f"  {deg_test}")

    # Fully randomized run
    lr_mock = deg_test(hr_mock)
    print(f"\n  [randomized]  LR: {lr_mock.shape}  min={lr_mock.min():.4f}  max={lr_mock.max():.4f}")

    # Test all fixed MST downsampling methods explicitly
    print(f"\n  [fixed method testing]")
    for idx_m in range(N_MST):
        lr_m = deg_test(hr_mock, method_idx=idx_m)
        print(f"    {MST_NAMES[idx_m]:10s} (idx={idx_m}): {lr_m.shape}  "
              f"min={lr_m.min():.4f}  max={lr_m.max():.4f}")

    # Validate Round-Robin distribution strategy simulation
    print(f"\n  [round-robin simulation — 12 patches]")
    for idx_rr in range(12):
        lr_rr = deg_test(hr_mock, method_idx=idx_rr % N_MST)
        print(f"    patch {idx_rr:2d} → {MST_NAMES[idx_rr % N_MST]:10s}  LR={lr_rr.shape}")

    # Forced procedural ordering
    print(f"\n  [enforced order pipeline: down→blur→noise]")
    lr_ordered = deg_test(hr_mock, order=[2, 0, 1])
    print(f"    LR: {lr_ordered.shape}  min={lr_ordered.min():.4f}  max={lr_ordered.max():.4f}")

    # Sequential batch verification
    hr_batch_mock = np.stack([hr_mock] * 4, axis=0)
    lr_batch_mock = deg_test.degrade_batch(hr_batch_mock)
    print(f"\n  [batch process]  HR={hr_batch_mock.shape}  LR={lr_batch_mock.shape}")

    print(f"\n  ✓ All basic smoke tests passed successfully.\n")