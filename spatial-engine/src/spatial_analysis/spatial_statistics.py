"""
Spatial statistics computations for pathology research.
Implements common metrics for analyzing cell distributions.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from collections import defaultdict


class SpatialStatisticsEngine:
    """
    Compute spatial statistics metrics commonly used in pathology analysis.
    """

    def __init__(self, centroids: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Initialize with cell centroids and optional labels.

        Args:
            centroids: Nx2 array of (x, y) coordinates
            labels: Optional array of cell type labels
        """
        self.centroids = np.asarray(centroids)
        self.labels = labels
        self.tree = KDTree(self.centroids) if len(self.centroids) > 0 else None

    def compute_nearest_neighbor_distribution(self) -> Dict:
        """
        Compute distribution of nearest-neighbor distances.
        Key metric for detecting clustering vs dispersion.
        """
        if self.tree is None or len(self.centroids) < 2:
            return {'error': 'Insufficient data points'}

        # k=2 because the nearest neighbor of each point is itself at distance 0
        distances, _ = self.tree.query(self.centroids, k=2)
        nn_distances = distances[:, 1]

        # Compute histogram for visualization
        hist, bin_edges = np.histogram(nn_distances, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'mean': float(np.mean(nn_distances)),
            'median': float(np.median(nn_distances)),
            'std': float(np.std(nn_distances)),
            'min': float(np.min(nn_distances)),
            'max': float(np.max(nn_distances)),
            'percentile_25': float(np.percentile(nn_distances, 25)),
            'percentile_75': float(np.percentile(nn_distances, 75)),
            'percentile_95': float(np.percentile(nn_distances, 95)),
            'histogram': {
                'counts': hist.tolist(),
                'bin_centers': bin_centers.tolist()
            }
        }

    def compute_ripleys_k(
        self,
        radii: np.ndarray,
        area: float
    ) -> Dict:
        """
        Compute Ripley's K-function for spatial clustering analysis.

        K(r) measures the expected number of points within distance r
        of a typical point, normalized by density.

        K(r) > πr² indicates clustering
        K(r) < πr² indicates dispersion
        K(r) ≈ πr² indicates random (Poisson) distribution
        """
        if self.tree is None:
            return {'error': 'No spatial data'}

        n = len(self.centroids)
        density = n / area

        k_values = []
        l_values = []  # Besag's L-function: L(r) = sqrt(K(r)/π)

        for r in radii:
            # Count pairs within distance r
            pairs = self.tree.query_pairs(r)
            count = len(pairs) * 2  # Each pair counted once, need both directions

            # K(r) = area * count / (n * (n-1))
            k_r = area * count / (n * (n - 1)) if n > 1 else 0
            k_values.append(k_r)

            # L(r) = sqrt(K(r)/π) - r
            # Positive L indicates clustering, negative indicates dispersion
            l_r = np.sqrt(k_r / np.pi) - r
            l_values.append(l_r)

        # Expected K for complete spatial randomness (CSR)
        csr_k = np.pi * radii ** 2

        return {
            'radii': radii.tolist(),
            'k_observed': k_values,
            'k_expected_csr': csr_k.tolist(),
            'l_function': l_values,
            'is_clustered': [k > csr for k, csr in zip(k_values, csr_k)]
        }

    def compute_label_colocalization(
        self,
        radius: float = 50.0
    ) -> Dict:
        """
        Compute co-localization scores between cell types.
        Measures how often cells of different types appear near each other.
        """
        if self.labels is None:
            return {'error': 'No labels provided'}

        unique_labels = np.unique(self.labels)
        n_labels = len(unique_labels)

        # Initialize co-occurrence matrix
        cooccurrence = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)

        # For each point, count neighbors of each type within radius
        indices = self.tree.query_ball_tree(self.tree, radius)

        for i, neighbors in enumerate(indices):
            label_i = self.labels[i]
            label_counts[label_i] += 1

            for j in neighbors:
                if i != j:
                    label_j = self.labels[j]
                    cooccurrence[label_i][label_j] += 1

        # Normalize to get colocalization scores
        colocalization = {}
        for label_i in unique_labels:
            colocalization[label_i] = {}
            total_neighbors = sum(cooccurrence[label_i].values())

            for label_j in unique_labels:
                if total_neighbors > 0:
                    score = cooccurrence[label_i][label_j] / total_neighbors
                else:
                    score = 0.0
                colocalization[label_i][label_j] = score

        return {
            'radius': radius,
            'colocalization_matrix': colocalization,
            'label_counts': dict(label_counts)
        }

    def compute_hotspot_detection(
        self,
        cell_size: float = 100.0,
        min_density: float = 10.0
    ) -> Dict:
        """
        Detect spatial hotspots using grid-based density analysis.
        Returns grid cells that exceed the minimum density threshold.
        """
        if len(self.centroids) == 0:
            return {'hotspots': []}

        x_min, y_min = self.centroids.min(axis=0)
        x_max, y_max = self.centroids.max(axis=0)

        # Create grid
        n_x = int(np.ceil((x_max - x_min) / cell_size))
        n_y = int(np.ceil((y_max - y_min) / cell_size))

        # Count cells in each grid cell
        grid_x = ((self.centroids[:, 0] - x_min) / cell_size).astype(int)
        grid_y = ((self.centroids[:, 1] - y_min) / cell_size).astype(int)

        # Clip to valid range
        grid_x = np.clip(grid_x, 0, n_x - 1)
        grid_y = np.clip(grid_y, 0, n_y - 1)

        # Count per grid cell
        grid_counts = np.zeros((n_x, n_y))
        np.add.at(grid_counts, (grid_x, grid_y), 1)

        # Compute density (cells per unit area)
        cell_area = cell_size ** 2
        grid_density = grid_counts / cell_area * 10000  # per 10000 sq pixels

        # Find hotspots
        hotspots = []
        for ix in range(n_x):
            for iy in range(n_y):
                if grid_density[ix, iy] >= min_density:
                    hotspots.append({
                        'grid_x': int(ix),
                        'grid_y': int(iy),
                        'center_x': float(x_min + (ix + 0.5) * cell_size),
                        'center_y': float(y_min + (iy + 0.5) * cell_size),
                        'count': int(grid_counts[ix, iy]),
                        'density': float(grid_density[ix, iy])
                    })

        # Sort by density
        hotspots.sort(key=lambda h: h['density'], reverse=True)

        return {
            'hotspots': hotspots,
            'grid_size': cell_size,
            'total_cells': int(len(self.centroids)),
            'n_hotspots': len(hotspots),
            'max_density': float(grid_density.max()),
            'mean_density': float(grid_density[grid_counts > 0].mean())
        }

    def compute_summary_statistics(self, area: float) -> Dict:
        """Compute comprehensive summary of spatial distribution."""
        n = len(self.centroids)

        if n == 0:
            return {'error': 'No data points'}

        summary = {
            'total_count': n,
            'density': n / area * 1e6,  # per million sq pixels
            'centroid': {
                'x': float(self.centroids[:, 0].mean()),
                'y': float(self.centroids[:, 1].mean())
            },
            'extent': {
                'x_min': float(self.centroids[:, 0].min()),
                'x_max': float(self.centroids[:, 0].max()),
                'y_min': float(self.centroids[:, 1].min()),
                'y_max': float(self.centroids[:, 1].max())
            }
        }

        # Add nearest neighbor stats
        summary['nearest_neighbor'] = self.compute_nearest_neighbor_distribution()

        # Add label distribution if available
        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            summary['label_distribution'] = {
                str(label): int(count) for label, count in zip(unique, counts)
            }

        return summary
