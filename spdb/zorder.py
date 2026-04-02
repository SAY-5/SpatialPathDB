"""Z-order (Morton) curve encoding for controlled comparison with Hilbert."""

import numpy as np


def xy2z(x: int, y: int, p: int) -> int:
    """Interleave bits of x and y to produce a Z-order (Morton) code."""
    z = 0
    for i in range(p):
        z |= ((x >> i) & 1) << (2 * i + 1)
        z |= ((y >> i) & 1) << (2 * i)
    return z


def z2xy(z: int, p: int):
    """De-interleave a Z-order code back to (x, y)."""
    x = y = 0
    for i in range(p):
        x |= ((z >> (2 * i + 1)) & 1) << i
        y |= ((z >> (2 * i)) & 1) << i
    return x, y


def encode_batch(xs: np.ndarray, ys: np.ndarray, p: int) -> np.ndarray:
    """Vectorized Z-order encoding via bit-interleaving."""
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    z = np.zeros(len(xs), dtype=np.int64)
    for i in range(p):
        z |= ((xs >> i) & 1) << (2 * i + 1)
        z |= ((ys >> i) & 1) << (2 * i)
    return z


def normalize_coords(x: np.ndarray, y: np.ndarray,
                     w: float, h: float, p: int) -> tuple:
    """Map slide coordinates to [0, 2^p) grid cells."""
    n = 1 << p
    gx = np.clip((x * n / w).astype(np.int64), 0, n - 1)
    gy = np.clip((y * n / h).astype(np.int64), 0, n - 1)
    return gx, gy


def bucket_from_zorder(z: np.ndarray, p: int, num_buckets: int) -> np.ndarray:
    """Map Z-order codes to bucket IDs."""
    total_cells = np.int64(1) << (2 * p)
    return np.clip((z * num_buckets // total_cells).astype(np.int64), 0, num_buckets - 1)
