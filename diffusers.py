"""
Author: Tomer Porat

Measurement matrix generators (diffusers) for ghost imaging
Dataset-agnostic: Works with CIFAR10, MNIST, Fashion-MNIST
Supports: Learned, DCT (dct/dct2d), Hadamard (hadamard/hadamard2d), Random
"""

import torch
import numpy as np
from scipy.fftpack import dct as scipy_dct
import os
from config import DEVICE, IMAGE_SIZE, INPUT_CHANNELS


class DiffuserFactory:
    """Factory for creating measurement matrices"""

    @staticmethod
    def create(mode, M, D=3072, device=None):
        """
        Create measurement matrix

        Args:
            mode: "learned", "dct", "dct2d", "hadamard", "hadamard2d", or "random"
            M: number of measurements
            D: input dimension (auto-detected from config for DCT2D/Hadamard2D)
            device: torch device (default: DEVICE from config)

        Returns:
            torch.Tensor of shape (M, D)
        """
        if device is None:
            device = DEVICE

        if mode == "learned":
            return LearnedDiffuser.load(M, device)
        elif mode == "dct":
            return DCTDiffuser.create(M, D, device)
        elif mode == "dct2d":
            # DCT2D uses spatial dimensions from config (IMAGE_SIZE, INPUT_CHANNELS)
            return DCT2DDiffuser.create(M, C=INPUT_CHANNELS, H=IMAGE_SIZE, W=IMAGE_SIZE, device=device)
        elif mode == "hadamard":
            return HadamardDiffuser.create(M, D, device)
        elif mode == "hadamard2d":
            # Sequency-ordered 2D Hadamard, fixes the low-frequency-coverage
            # bug of natural-order HadamardDiffuser (see HadamardDiffuser2D
            # docstring). Uses spatial dims from config, like dct2d.
            return HadamardDiffuser2D.create(M, C=INPUT_CHANNELS, H=IMAGE_SIZE, W=IMAGE_SIZE, device=device)
        elif mode == "random":
            return RandomDiffuser.create(M, D, device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

class DCTDiffuser:
    """Discrete Cosine Transform measurement matrix"""

    @staticmethod
    def create(M, D=3072, device=None):
        """
        Create orthonormal DCT-II measurement matrix.

        Fix 1: norm='ortho', row norms are now exactly 1.0.
                Without this, rows had norm ~78, causing gradient explosion
                and preventing the classifier from learning effectively.
        Fix 2: Vectorized construction, applies scipy_dct to the full
                identity matrix at once (C speed) instead of a Python loop
                over D rows (was taking 10+ minutes for D=3072).

        Args:
            M: number of measurements (rows to keep)
            D: input dimension (e.g. 3072 for CIFAR10)
            device: torch device (default: DEVICE from config)

        Returns:
            torch.Tensor: (M, D) orthonormal DCT matrix, row norms = 1.0
        """
        if device is None:
            device = DEVICE
        dct_basis = scipy_dct(np.eye(D, dtype=np.float32), norm='ortho', axis=1)
        return torch.from_numpy(dct_basis[:M, :]).to(device)


class DCT2DDiffuser:
    """
    2D per-channel DCT measurement matrix, best performing DCT mode.

    Unlike DCTDiffuser (1D: treats the full 3072-dim flat vector as one signal),
    this applies a separate 2D DCT to each colour channel and selects the
    lowest-frequency coefficients per channel in zigzag order (JPEG-style).

    Result on CIFAR10, M=128, 50 epochs: 64.30% vs 49.98% baseline.
    Row norms = exactly 1.0 (orthonormal).
    """

    @staticmethod
    def create(M, C=3, H=32, W=32, device=None):
        """
        Args:
            M: total number of measurements
            C: colour channels (3 for CIFAR10, 1 for MNIST)
            H, W: spatial size of each channel (32×32 for CIFAR10)
            device: torch device

        Returns:
            torch.Tensor: (M, C*H*W) orthonormal DCT2D matrix
        """
        if device is None:
            device = DEVICE

        D_h = scipy_dct(np.eye(H, dtype=np.float32), norm='ortho', axis=1)
        D_w = scipy_dct(np.eye(W, dtype=np.float32), norm='ortho', axis=1)
        D_2d = np.kron(D_h, D_w)  # (H*W, H*W)

        m_base  = M // C
        m_extra = M % C
        m_per_ch = [m_base + (1 if c < m_extra else 0) for c in range(C)]

        D_full = np.zeros((M, C * H * W), dtype=np.float32)
        row_ptr = 0
        for c in range(C):
            mc   = m_per_ch[c]
            zidx = DCT2DDiffuser._zigzag(H, W, mc)
            D_full[row_ptr:row_ptr + mc, c * H * W:(c + 1) * H * W] = D_2d[zidx]
            row_ptr += mc

        return torch.from_numpy(D_full).to(device)

    @staticmethod
    def _zigzag(rows, cols, n):
        """Return indices of the n lowest-frequency entries in zigzag order."""
        idx = []
        for s in range(rows + cols - 1):
            if s % 2 == 0:
                ky_range = range(max(0, s - cols + 1), min(rows, s + 1))
            else:
                ky_range = range(min(rows - 1, s), max(-1, s - cols), -1)
            for ky in ky_range:
                idx.append(ky * cols + (s - ky))
                if len(idx) == n:
                    return idx
        return idx


class HadamardDiffuser:
    """Walsh-Hadamard Transform measurement matrix"""

    @staticmethod
    def create(M, D=3072, device=None):
        """
        Create normalized Walsh-Hadamard measurement matrix.

        Fix: Normalize by 1/sqrt(D) so each row has unit norm (same scale
             as orthonormal DCT). The original was unnormalized, each row
             had norm sqrt(D) ≈ 55, causing severe gradient scaling issues.
             Note: cast to float32 AFTER division to avoid NumPy upcasting
             to float64 (incompatible with MPS devices).

        Args:
            M: number of measurements (rows to keep)
            D: input dimension (e.g. 3072 for CIFAR10)
            device: torch device (default: DEVICE from config)

        Returns:
            torch.Tensor: (M, D) Hadamard matrix, row norms = 1.0
        """
        if device is None:
            device = DEVICE
        from scipy.linalg import hadamard
        size = 2 ** int(np.ceil(np.log2(D)))
        H = hadamard(size)
        H_sub = (H[:M, :D] / np.sqrt(D)).astype(np.float32)
        return torch.from_numpy(H_sub).to(device)


class HadamardDiffuser2D:
    """
    2D per-channel Hadamard measurement matrix in sequency (Walsh) order.
    Fixes a low-frequency-coverage bug in the natural-order HadamardDiffuser.

    scipy.linalg.hadamard() returns rows in NATURAL (Sylvester) order, where
    the row index does not correspond to spatial frequency. For a 32x32 image,
    H_1024 = H_32 (kron) H_32, and row k = 32a + b of H_1024 equals
    outer(H_32[a], H_32[b]) reshaped flat. HadamardDiffuser (legacy) takes the
    first M rows in this natural order: for M=128 that means a = k // 32
    only ranges over {0,1,2,3} while b sweeps the full 0..31, i.e. only 4 of
    32 possible vertical frequencies are ever sampled, and the other 28 are
    skipped entirely regardless of M. This is the root cause of the weak and
    non-monotonic Hadamard accuracy observed throughout this project at small
    M (e.g. MNIST M=16 < M=8, see CLAUDE.md / thesis result tables).

    Fix (reference implementation reviewed at
    only_code_12_10_24_Mor_Hale_ORIGINAL/fixes/reconstruction_v2.py,
    functions walsh_matrix + select_hadamard_masks_v2): reorder each 1D
    Hadamard basis by "sequency" (the number of sign changes along a row,
    its true frequency, low to high), then build 2D masks as
    kron(W[a], W[b]) and pick (a, b) pairs by ascending 2D frequency
    (max(a,b) first, tie-broken by a+b), a balanced low-frequency grid over
    BOTH spatial axes at once. This is the exact Hadamard analogue of the
    zigzag low-frequency selection DCT2DDiffuser already uses.
    """

    @staticmethod
    def _walsh_matrix(n):
        """1D Hadamard basis reordered by sequency (row i has i sign changes)."""
        from scipy.linalg import hadamard
        H = hadamard(n)
        sequency = [int(np.sum(np.abs(np.diff(row)) > 0)) for row in H]
        return H[np.argsort(sequency)]

    @staticmethod
    def _select_pairs(pic_width, n):
        """First n (a, b) frequency pairs in ascending 2D-sequency order."""
        pairs = [(a, b) for a in range(pic_width) for b in range(pic_width)]
        pairs.sort(key=lambda ab: (max(ab), ab[0] + ab[1], ab))
        return pairs[:n]

    @staticmethod
    def create(M, C=3, H=32, W=32, device=None):
        """
        Args:
            M: total number of measurements
            C: colour channels (3 for CIFAR10, 1 for MNIST/FMNIST/RIAWELC)
            H, W: spatial size of each channel, must be equal and a power
                  of 2 (scipy.linalg.hadamard requirement); 32 for this
                  project's unified measurement grid
            device: torch device

        Returns:
            torch.Tensor: (M, C*H*W) Hadamard matrix in sequency order,
                          row norms = 1.0 (entries are +-1/sqrt(H*W))
        """
        if device is None:
            device = DEVICE
        if H != W:
            raise ValueError(f"HadamardDiffuser2D requires square images, got H={H}, W={W}")

        Wm = HadamardDiffuser2D._walsh_matrix(H)  # (H, H) rows, sequency order

        m_base  = M // C
        m_extra = M % C
        m_per_ch = [m_base + (1 if c < m_extra else 0) for c in range(C)]

        D_full = np.zeros((M, C * H * W), dtype=np.float32)
        row_ptr = 0
        for c in range(C):
            mc = m_per_ch[c]
            pairs = HadamardDiffuser2D._select_pairs(H, mc)
            rows = np.array([np.kron(Wm[a], Wm[b]) for a, b in pairs], dtype=np.float32)
            rows /= np.sqrt(H * W)  # unit row norm (entries are +-1 before scaling)
            D_full[row_ptr:row_ptr + mc, c * H * W:(c + 1) * H * W] = rows
            row_ptr += mc

        return torch.from_numpy(D_full).to(device)


class RandomDiffuser:
    """Random Gaussian measurement matrix"""

    @staticmethod
    def create(M, D=3072, device=None, seed=None):
        """
        Create random Gaussian measurement matrix

        Args:
            M: number of measurements
            D: input dimension
            device: torch device (default: DEVICE from config)
            seed: random seed for reproducibility

        Returns:
            torch.Tensor: (M, D) random matrix
        """
        if device is None:
            device = DEVICE
        if seed is not None:
            torch.manual_seed(seed)
        
        D_tensor = torch.randn(M, D).float() / np.sqrt(M)
        return D_tensor.to(device)


def compute_measurements(diffuser, images):
    """
    Compute measurements from images using diffuser
    
    Args:
        diffuser: measurement matrix (M, D)
        images: batch of images (batch, C, H, W)
        
    Returns:
        torch.Tensor: measurements (batch, M)
    """
    batch_size = images.size(0)
    x_flat = images.view(batch_size, -1)  # (batch, D)
    measurements = (diffuser @ x_flat.T).T  # (batch, M)
    return measurements
