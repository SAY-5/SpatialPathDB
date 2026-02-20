"""
Density estimation for spatial cell distributions.
Supports grid-based aggregation and kernel density estimation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter


class DensityEstimator:
    """
    Compute density maps and heatmaps from cell coordinates.
    """

    def __init__(self, centroids: np.ndarray, bounds: Optional[Tuple] = None):
        """
        Initialize with cell centroids.

        Args:
            centroids: Nx2 array of (x, y) coordinates
            bounds: Optional (x_min, y_min, x_max, y_max) for the slide
        """
        self.centroids = np.asarray(centroids)

        if bounds is None and len(self.centroids) > 0:
            padding = 100
            self.bounds = (
                self.centroids[:, 0].min() - padding,
                self.centroids[:, 1].min() - padding,
                self.centroids[:, 0].max() + padding,
                self.centroids[:, 1].max() + padding
            )
        else:
            self.bounds = bounds or (0, 0, 100000, 80000)

    def compute_grid_density(
        self,
        grid_size: float = 256.0,
        labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute density using fixed-size grid cells.
        Fast and suitable for large datasets.
        """
        x_min, y_min, x_max, y_max = self.bounds

        # Compute grid dimensions
        n_x = int(np.ceil((x_max - x_min) / grid_size))
        n_y = int(np.ceil((y_max - y_min) / grid_size))

        if len(self.centroids) == 0:
            return {
                'grid': np.zeros((n_y, n_x)).tolist(),
                'grid_size': grid_size,
                'bounds': self.bounds
            }

        # Compute grid indices for each point
        grid_x = ((self.centroids[:, 0] - x_min) / grid_size).astype(int)
        grid_y = ((self.centroids[:, 1] - y_min) / grid_size).astype(int)

        # Clip to valid range
        grid_x = np.clip(grid_x, 0, n_x - 1)
        grid_y = np.clip(grid_y, 0, n_y - 1)

        # Count cells in each grid cell (note: y is rows, x is columns)
        density_grid = np.zeros((n_y, n_x))
        np.add.at(density_grid, (grid_y, grid_x), 1)

        result = {
            'grid': density_grid.tolist(),
            'grid_size': grid_size,
            'bounds': self.bounds,
            'shape': (n_y, n_x),
            'max_count': int(density_grid.max()),
            'total_count': int(density_grid.sum())
        }

        # If labels provided, compute per-label density grids
        if labels is not None:
            result['by_label'] = {}
            unique_labels = np.unique(labels)

            for label in unique_labels:
                mask = labels == label
                label_grid = np.zeros((n_y, n_x))
                np.add.at(label_grid, (grid_y[mask], grid_x[mask]), 1)
                result['by_label'][str(label)] = label_grid.tolist()

        return result

    def compute_kde(
        self,
        resolution: int = 256,
        bandwidth: Optional[float] = None
    ) -> Dict:
        """
        Compute kernel density estimation.
        Produces smooth density surfaces.
        """
        if len(self.centroids) < 10:
            return {'error': 'Insufficient data for KDE'}

        x_min, y_min, x_max, y_max = self.bounds

        # Create evaluation grid
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Compute KDE
        try:
            from scipy.stats import gaussian_kde

            # Subsample if too many points for KDE performance
            if len(self.centroids) > 50000:
                indices = np.random.choice(
                    len(self.centroids), 50000, replace=False
                )
                sample = self.centroids[indices].T
            else:
                sample = self.centroids.T

            kernel = gaussian_kde(sample, bw_method=bandwidth)
            density = kernel(positions).reshape(resolution, resolution)

            # Normalize to [0, 1]
            density = (density - density.min()) / (density.max() - density.min() + 1e-10)

        except Exception as e:
            return {'error': f'KDE computation failed: {str(e)}'}

        return {
            'density': density.tolist(),
            'resolution': resolution,
            'bounds': self.bounds,
            'x_coords': x_grid.tolist(),
            'y_coords': y_grid.tolist()
        }

    def compute_smoothed_density(
        self,
        grid_size: float = 256.0,
        sigma: float = 2.0
    ) -> Dict:
        """
        Compute grid density with Gaussian smoothing.
        Good balance between speed and visual quality.
        """
        # First compute grid density
        grid_result = self.compute_grid_density(grid_size)
        density_grid = np.array(grid_result['grid'])

        # Apply Gaussian smoothing
        smoothed = gaussian_filter(density_grid.astype(float), sigma=sigma)

        # Normalize to [0, 1]
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()

        grid_result['smoothed_grid'] = smoothed.tolist()
        grid_result['sigma'] = sigma

        return grid_result

    def compute_contours(
        self,
        grid_size: float = 512.0,
        n_levels: int = 5,
        sigma: float = 3.0
    ) -> Dict:
        """
        Compute density contour levels for visualization.
        Returns threshold values for contour rendering.
        """
        grid_result = self.compute_smoothed_density(grid_size, sigma)
        smoothed = np.array(grid_result['smoothed_grid'])

        # Compute contour levels based on percentiles of non-zero values
        non_zero = smoothed[smoothed > 0.01]
        if len(non_zero) == 0:
            return {**grid_result, 'contour_levels': []}

        percentiles = np.linspace(20, 95, n_levels)
        levels = [float(np.percentile(non_zero, p)) for p in percentiles]

        grid_result['contour_levels'] = levels
        grid_result['percentiles'] = percentiles.tolist()

        return grid_result

    def get_density_at_point(self, x: float, y: float, radius: float = 100.0) -> float:
        """Get local density around a specific point."""
        if len(self.centroids) == 0:
            return 0.0

        distances = np.sqrt(
            (self.centroids[:, 0] - x) ** 2 +
            (self.centroids[:, 1] - y) ** 2
        )
        count = np.sum(distances <= radius)
        area = np.pi * radius ** 2
        return count / area * 1e6  # per million sq pixels
