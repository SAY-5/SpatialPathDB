"""
Generates realistic synthetic spatial data mimicking whole-slide image analysis outputs.
Creates millions of cell annotations with realistic spatial distributions:
- Clustered cells in tissue regions using Gaussian mixture models
- Varied cell types with realistic proportions
- Realistic polygon geometries with morphological variation
- Properties like area, eccentricity, and staining intensity
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm

from ..utils.geometry_utils import create_cell_polygon, create_tissue_region, compute_polygon_features


@dataclass
class CellTypeConfig:
    """Configuration for a cell type."""
    name: str
    proportion: float
    avg_radius: float
    radius_std: float
    color: str
    confidence_mean: float = 0.85
    confidence_std: float = 0.1


@dataclass
class SlideConfig:
    """Configuration for synthetic slide generation."""
    width: int = 100000
    height: int = 80000
    microns_per_pixel: float = 0.25
    stain_type: str = 'H&E'
    organ: str = 'Breast'


# Biologically realistic cell type distributions
DEFAULT_CELL_TYPES = [
    CellTypeConfig('epithelial', 0.35, 12.0, 3.0, '#FF6B6B', 0.88, 0.08),
    CellTypeConfig('stromal', 0.25, 15.0, 4.0, '#4ECDC4', 0.82, 0.12),
    CellTypeConfig('lymphocyte', 0.20, 6.0, 1.5, '#45B7D1', 0.90, 0.06),
    CellTypeConfig('macrophage', 0.10, 10.0, 2.5, '#96CEB4', 0.78, 0.15),
    CellTypeConfig('necrotic', 0.10, 14.0, 5.0, '#FFEAA7', 0.65, 0.20),
]


class SyntheticSlideGenerator:
    """
    Generates synthetic spatial data for whole-slide pathology images.
    Uses Gaussian mixture models to create realistic clustered distributions.
    """

    def __init__(
        self,
        slide_config: Optional[SlideConfig] = None,
        cell_types: Optional[List[CellTypeConfig]] = None,
        seed: int = 42
    ):
        self.config = slide_config or SlideConfig()
        self.cell_types = cell_types or DEFAULT_CELL_TYPES
        self.rng = np.random.default_rng(seed)

        # Normalize proportions
        total_prop = sum(ct.proportion for ct in self.cell_types)
        for ct in self.cell_types:
            ct.proportion /= total_prop

    def generate_cluster_centers(
        self,
        n_clusters: int,
        margin: float = 0.1
    ) -> np.ndarray:
        """Generate cluster centers with spatial separation."""
        margin_x = self.config.width * margin
        margin_y = self.config.height * margin

        centers = []
        min_distance = min(self.config.width, self.config.height) / (np.sqrt(n_clusters) * 2)

        attempts = 0
        max_attempts = n_clusters * 100

        while len(centers) < n_clusters and attempts < max_attempts:
            x = self.rng.uniform(margin_x, self.config.width - margin_x)
            y = self.rng.uniform(margin_y, self.config.height - margin_y)

            # Check minimum distance from existing centers
            if len(centers) == 0:
                centers.append([x, y])
            else:
                distances = np.sqrt(np.sum((np.array(centers) - [x, y]) ** 2, axis=1))
                if np.min(distances) > min_distance:
                    centers.append([x, y])

            attempts += 1

        return np.array(centers)

    def generate_tissue_regions(self, n_regions: int = 5) -> List[Dict]:
        """Generate tissue region boundaries as large polygons."""
        regions = []
        centers = self.generate_cluster_centers(n_regions, margin=0.15)

        tissue_types = ['tumor', 'stroma', 'necrosis', 'normal', 'inflammation']

        for i, (cx, cy) in enumerate(centers):
            # Vary region size based on type
            base_radius = self.rng.uniform(5000, 15000)
            irregularity = self.rng.uniform(0.2, 0.4)

            polygon = create_tissue_region(
                cx, cy, base_radius, irregularity,
                n_vertices=32, rng=self.rng
            )

            features = compute_polygon_features(polygon)
            tissue_type = tissue_types[i % len(tissue_types)]

            regions.append({
                'object_type': 'tissue_region',
                'label': tissue_type,
                'confidence': float(self.rng.uniform(0.85, 0.98)),
                'geometry': polygon,
                'centroid': polygon.centroid,
                'area_pixels': features['area'],
                'perimeter_pixels': features['perimeter'],
                'properties': {
                    'tissue_type': tissue_type,
                    'region_id': i,
                    'circularity': features['circularity']
                }
            })

        return regions

    def _assign_cell_type(self) -> CellTypeConfig:
        """Randomly assign a cell type based on proportions."""
        r = self.rng.random()
        cumulative = 0.0
        for ct in self.cell_types:
            cumulative += ct.proportion
            if r <= cumulative:
                return ct
        return self.cell_types[-1]

    def generate_cells_gmm(
        self,
        n_cells: int,
        n_clusters: int = 20,
        cluster_std_range: Tuple[float, float] = (500, 3000)
    ) -> Iterator[Dict]:
        """
        Generate cell annotations using Gaussian mixture model.
        Yields cells one at a time for memory efficiency.
        """
        # Generate cluster parameters
        centers = self.generate_cluster_centers(n_clusters)
        cluster_stds = self.rng.uniform(
            cluster_std_range[0],
            cluster_std_range[1],
            size=(n_clusters, 2)
        )

        # Assign cells to clusters (uneven distribution makes it more realistic)
        cluster_weights = self.rng.dirichlet(np.ones(n_clusters) * 2)
        cells_per_cluster = (cluster_weights * n_cells).astype(int)
        cells_per_cluster[-1] += n_cells - cells_per_cluster.sum()

        cell_id = 0
        for cluster_idx in range(n_clusters):
            n_cluster_cells = cells_per_cluster[cluster_idx]
            cx, cy = centers[cluster_idx]
            std_x, std_y = cluster_stds[cluster_idx]

            # Generate cell positions from 2D Gaussian
            positions = self.rng.normal(
                loc=[cx, cy],
                scale=[std_x, std_y],
                size=(n_cluster_cells, 2)
            )

            for pos in positions:
                x, y = pos

                # Skip cells outside slide bounds
                if not (0 < x < self.config.width and 0 < y < self.config.height):
                    continue

                cell_type = self._assign_cell_type()

                # Generate cell radius with type-specific variation
                radius = max(3, self.rng.normal(cell_type.avg_radius, cell_type.radius_std))

                # Create cell polygon
                polygon = create_cell_polygon(x, y, radius, n_vertices=10, rng=self.rng)
                features = compute_polygon_features(polygon)

                # Generate confidence score
                confidence = np.clip(
                    self.rng.normal(cell_type.confidence_mean, cell_type.confidence_std),
                    0.1, 1.0
                )

                # Synthetic staining intensity (H&E specific)
                hematoxylin = self.rng.uniform(0.3, 0.9)
                eosin = self.rng.uniform(0.2, 0.8)

                yield {
                    'object_type': 'cell',
                    'label': cell_type.name,
                    'confidence': float(confidence),
                    'geometry': polygon,
                    'centroid': polygon.centroid,
                    'area_pixels': features['area'],
                    'perimeter_pixels': features['perimeter'],
                    'properties': {
                        'cluster_id': int(cluster_idx),
                        'circularity': features['circularity'],
                        'eccentricity': features['eccentricity'],
                        'hematoxylin_intensity': hematoxylin,
                        'eosin_intensity': eosin,
                        'color': cell_type.color
                    }
                }

                cell_id += 1

    def generate_full_slide(
        self,
        n_cells: int = 1_000_000,
        n_clusters: int = 30,
        include_tissue_regions: bool = True,
        progress: bool = True
    ) -> Tuple[Dict, List[Dict]]:
        """
        Generate complete synthetic slide with metadata and spatial objects.

        Returns:
            Tuple of (slide_metadata, list of spatial_objects)
        """
        slide_id = str(uuid.uuid4())

        slide_metadata = {
            'id': slide_id,
            'slide_name': f'synthetic_slide_{slide_id[:8]}',
            'file_path': f'/data/slides/{slide_id}.svs',
            'width_pixels': self.config.width,
            'height_pixels': self.config.height,
            'microns_per_pixel': self.config.microns_per_pixel,
            'stain_type': self.config.stain_type,
            'organ': self.config.organ,
            'metadata': {
                'synthetic': True,
                'n_clusters': n_clusters,
                'target_cells': n_cells,
                'cell_types': [ct.name for ct in self.cell_types]
            }
        }

        objects = []

        # Generate tissue regions first
        if include_tissue_regions:
            regions = self.generate_tissue_regions(n_regions=5)
            objects.extend(regions)

        # Generate cells with progress bar
        cell_generator = self.generate_cells_gmm(n_cells, n_clusters)

        if progress:
            cell_generator = tqdm(
                cell_generator,
                total=n_cells,
                desc='Generating cells',
                unit='cells'
            )

        for cell in cell_generator:
            objects.append(cell)

        return slide_metadata, objects

    def generate_cells_streaming(
        self,
        slide_id: str,
        n_cells: int,
        n_clusters: int = 30,
        batch_size: int = 10000
    ) -> Iterator[List[Dict]]:
        """
        Generate cells in batches for memory-efficient bulk loading.
        Yields batches of cell dictionaries.
        """
        batch = []

        for cell in self.generate_cells_gmm(n_cells, n_clusters):
            cell['slide_id'] = slide_id
            batch.append(cell)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def generate_multi_slide_dataset(
    n_slides: int = 10,
    cells_per_slide: int = 500_000,
    seed: int = 42
) -> Iterator[Tuple[Dict, Iterator[List[Dict]]]]:
    """
    Generate multiple synthetic slides with varying characteristics.
    Memory-efficient generator for large datasets.
    """
    organs = ['Breast', 'Lung', 'Colon', 'Liver', 'Kidney']
    stains = ['H&E', 'IHC-Ki67', 'IHC-CD3', 'IHC-CD8', 'H&E']

    for i in range(n_slides):
        # Vary slide parameters
        config = SlideConfig(
            width=np.random.randint(80000, 120000),
            height=np.random.randint(60000, 100000),
            microns_per_pixel=np.random.choice([0.25, 0.5, 0.125]),
            stain_type=stains[i % len(stains)],
            organ=organs[i % len(organs)]
        )

        generator = SyntheticSlideGenerator(config, seed=seed + i)
        slide_meta, _ = generator.generate_full_slide(
            n_cells=0,  # Just get metadata
            include_tissue_regions=False
        )

        # Create streaming cell generator
        cell_batches = generator.generate_cells_streaming(
            slide_meta['id'],
            cells_per_slide,
            n_clusters=20 + i * 2
        )

        yield slide_meta, cell_batches
