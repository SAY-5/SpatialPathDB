"""
Geometry conversion utilities for working with Shapely and PostGIS.
"""

from typing import Optional, Tuple, List
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.geometry.base import BaseGeometry


def wkt_to_shapely(wkt_string: str) -> Optional[BaseGeometry]:
    """Convert WKT string to Shapely geometry."""
    if not wkt_string:
        return None
    try:
        return wkt.loads(wkt_string)
    except Exception:
        return None


def shapely_to_wkt(geom: BaseGeometry) -> str:
    """Convert Shapely geometry to WKT string."""
    return geom.wkt


def create_cell_polygon(
    cx: float,
    cy: float,
    radius: float,
    n_vertices: int = 12,
    rng: Optional[np.random.Generator] = None
) -> Polygon:
    """
    Create an irregular polygon approximating a cell boundary.
    Adds realistic variation to vertex positions.
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    # Add radial variation (15% of radius)
    noise = rng.normal(1.0, 0.15, n_vertices)
    r = radius * np.clip(noise, 0.7, 1.3)

    # Add angular jitter
    angle_jitter = rng.normal(0, 0.05, n_vertices)
    angles = angles + angle_jitter

    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)

    return Polygon(zip(x, y))


def create_tissue_region(
    cx: float,
    cy: float,
    avg_radius: float,
    irregularity: float = 0.3,
    n_vertices: int = 24,
    rng: Optional[np.random.Generator] = None
) -> Polygon:
    """
    Create an irregular polygon representing a tissue region.
    More variation than cell polygons for organic tissue shapes.
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)

    # Use Perlin-like noise for smooth variations
    noise = 1.0 + irregularity * (
        np.sin(angles * 3) * 0.3 +
        np.sin(angles * 5) * 0.2 +
        rng.normal(0, 0.2, n_vertices)
    )
    r = avg_radius * np.clip(noise, 0.5, 1.5)

    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)

    return Polygon(zip(x, y))


def compute_polygon_features(polygon: Polygon) -> dict:
    """Compute morphological features of a polygon."""
    area = polygon.area
    perimeter = polygon.length
    centroid = polygon.centroid

    # Circularity: 4*pi*area / perimeter^2 (1.0 for perfect circle)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Get bounding box for eccentricity
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny

    # Eccentricity approximation from bounding box
    if max(width, height) > 0:
        eccentricity = min(width, height) / max(width, height)
    else:
        eccentricity = 1.0

    return {
        'area': area,
        'perimeter': perimeter,
        'centroid_x': centroid.x,
        'centroid_y': centroid.y,
        'circularity': circularity,
        'eccentricity': eccentricity,
        'bbox_width': width,
        'bbox_height': height
    }


def bbox_to_polygon(min_x: float, min_y: float, max_x: float, max_y: float) -> Polygon:
    """Convert bounding box coordinates to Polygon."""
    return Polygon([
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)
    ])


def sample_points_in_polygon(
    polygon: Polygon,
    n_points: int,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[float, float]]:
    """Sample random points uniformly within a polygon."""
    if rng is None:
        rng = np.random.default_rng()

    minx, miny, maxx, maxy = polygon.bounds
    points = []

    # Rejection sampling
    while len(points) < n_points:
        batch_size = min(n_points * 3, 10000)
        x = rng.uniform(minx, maxx, batch_size)
        y = rng.uniform(miny, maxy, batch_size)

        for px, py in zip(x, y):
            if polygon.contains(Point(px, py)):
                points.append((px, py))
                if len(points) >= n_points:
                    break

    return points
