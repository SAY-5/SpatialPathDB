"""Hilbert curve encoding: 2D coordinates to 1D Hilbert index and back.

Uses the standard bit-manipulation algorithm (Lawder 2000).  The vectorized
``encode_batch`` processes millions of points via numpy without Python loops.
"""

import numpy as np


def _rot(n: int, x: int, y: int, rx: int, ry: int):
    """Rotate/flip quadrant."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def xy2d(p: int, x: int, y: int) -> int:
    """Convert (x, y) in a 2^p grid to Hilbert index d."""
    n = 1 << p
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s >>= 1
    return d


def d2xy(p: int, d: int):
    """Convert Hilbert index d back to (x, y) in a 2^p grid."""
    n = 1 << p
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if (d & 1) ^ rx else 0  # (d & 1) XOR rx
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def encode_batch(xs: np.ndarray, ys: np.ndarray, p: int) -> np.ndarray:
    """Vectorized Hilbert encoding -- processes full arrays without Python loops."""
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    n = np.int64(1 << p)
    d = np.zeros(len(xs), dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((xs & s) > 0).astype(np.int64)
        ry = ((ys & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # Rotation: when ry == 0
        mask_ry0 = ry == 0
        mask_flip = mask_ry0 & (rx == 1)
        xs_new = xs.copy()
        ys_new = ys.copy()
        xs_new[mask_flip] = (s - 1) - xs[mask_flip]
        ys_new[mask_flip] = (s - 1) - ys[mask_flip]
        swap = mask_ry0
        tmp_x = xs_new.copy()
        xs_new[swap] = ys_new[swap]
        ys_new[swap] = tmp_x[swap]
        xs = xs_new
        ys = ys_new
        s >>= 1
    return d


def normalize_coords(x: np.ndarray, y: np.ndarray,
                     w: float, h: float, p: int) -> tuple:
    """Map slide coordinates to [0, 2^p) grid cells."""
    n = 1 << p
    gx = np.clip((x * n / w).astype(np.int64), 0, n - 1)
    gy = np.clip((y * n / h).astype(np.int64), 0, n - 1)
    return gx, gy


def bucket_from_hilbert(h: np.ndarray, p: int, num_buckets: int) -> np.ndarray:
    """Map Hilbert indices to bucket IDs."""
    total_cells = np.int64(1) << (2 * p)
    return np.clip((h * num_buckets // total_cells).astype(np.int64), 0, num_buckets - 1)


def hilbert_range_for_bbox(x_min, y_min, x_max, y_max, w, h, p):
    """Return the set of Hilbert keys for corners of a bounding box.

    Used by the query planner to identify candidate Hilbert buckets.
    Returns (h_min, h_max) spanning the range of all corner keys.
    """
    n = 1 << p
    corners_x = [x_min, x_min, x_max, x_max]
    corners_y = [y_min, y_max, y_min, y_max]
    gxs = [max(0, min(n - 1, int(cx * n / w))) for cx in corners_x]
    gys = [max(0, min(n - 1, int(cy * n / h))) for cy in corners_y]
    h_vals = [xy2d(p, gx, gy) for gx, gy in zip(gxs, gys)]
    return min(h_vals), max(h_vals)


def candidate_buckets_for_bbox(x_min, y_min, x_max, y_max,
                                w, h, p, num_buckets):
    """Return set of bucket IDs that a bounding box may touch.

    Scans all grid cells covered by the bbox to find exact bucket set.
    For small viewports (5% of slide) this is fast.
    """
    n = 1 << p
    gx_lo = max(0, int(x_min * n / w))
    gx_hi = min(n - 1, int(x_max * n / w))
    gy_lo = max(0, int(y_min * n / h))
    gy_hi = min(n - 1, int(y_max * n / h))

    total_cells = 1 << (2 * p)
    buckets = set()
    for gx in range(gx_lo, gx_hi + 1):
        for gy in range(gy_lo, gy_hi + 1):
            h = xy2d(p, gx, gy)
            b = min(num_buckets - 1, h * num_buckets // total_cells)
            buckets.add(b)
    return sorted(buckets)
