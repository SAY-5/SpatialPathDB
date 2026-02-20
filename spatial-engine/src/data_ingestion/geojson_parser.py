"""
GeoJSON parser for importing external annotation files.
Supports QuPath, ASAP, and generic GeoJSON formats.
"""

import json
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Union
import uuid

from shapely.geometry import shape, mapping
from shapely.validation import make_valid

from ..utils.geometry_utils import compute_polygon_features


class GeoJSONParser:
    """
    Parse GeoJSON annotation files from various pathology tools.
    Handles format variations and validates geometries.
    """

    # Known property mappings from different tools
    PROPERTY_MAPPINGS = {
        'qupath': {
            'classification': 'label',
            'objectType': 'object_type',
            'measurements': 'properties'
        },
        'asap': {
            'Name': 'label',
            'Type': 'object_type',
            'Color': 'color'
        }
    }

    def __init__(self, source_format: str = 'generic'):
        self.source_format = source_format
        self.mapping = self.PROPERTY_MAPPINGS.get(source_format, {})

    def _normalize_properties(self, properties: Dict) -> Dict:
        """Normalize property names based on source format."""
        normalized = {}

        for key, value in properties.items():
            mapped_key = self.mapping.get(key, key)
            normalized[mapped_key] = value

        return normalized

    def _extract_label(self, properties: Dict) -> str:
        """Extract classification label from properties."""
        # Try common label fields
        for field in ['label', 'classification', 'name', 'class', 'Name']:
            if field in properties:
                value = properties[field]
                # Handle nested classification objects (QuPath style)
                if isinstance(value, dict) and 'name' in value:
                    return value['name']
                return str(value)
        return 'unknown'

    def _extract_object_type(self, properties: Dict, geometry_type: str) -> str:
        """Determine object type from properties or geometry."""
        for field in ['object_type', 'objectType', 'type', 'Type']:
            if field in properties:
                return str(properties[field]).lower()

        # Infer from geometry type
        if geometry_type == 'Point':
            return 'marker'
        elif geometry_type in ['Polygon', 'MultiPolygon']:
            return 'annotation'
        return 'unknown'

    def parse_feature(self, feature: Dict) -> Optional[Dict]:
        """Parse a single GeoJSON feature into spatial object format."""
        try:
            geom = shape(feature['geometry'])

            # Fix invalid geometries
            if not geom.is_valid:
                geom = make_valid(geom)
                if geom.is_empty:
                    return None

            properties = feature.get('properties', {})
            normalized_props = self._normalize_properties(properties)

            label = self._extract_label(normalized_props)
            object_type = self._extract_object_type(
                normalized_props,
                feature['geometry']['type']
            )

            # Compute polygon features
            features = compute_polygon_features(geom)

            # Extract confidence if present
            confidence = None
            for field in ['confidence', 'score', 'probability']:
                if field in normalized_props:
                    confidence = float(normalized_props[field])
                    break

            return {
                'object_type': object_type,
                'label': label,
                'confidence': confidence,
                'geometry': geom,
                'centroid': geom.centroid,
                'area_pixels': features['area'],
                'perimeter_pixels': features['perimeter'],
                'properties': {
                    k: v for k, v in normalized_props.items()
                    if k not in ['label', 'object_type', 'confidence']
                }
            }

        except Exception as e:
            # Log but don't fail on individual features
            print(f"Warning: Failed to parse feature: {e}")
            return None

    def parse_file(self, file_path: Union[str, Path]) -> Iterator[Dict]:
        """Parse a GeoJSON file and yield spatial objects."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle both FeatureCollection and single Feature
        if data.get('type') == 'FeatureCollection':
            features = data.get('features', [])
        elif data.get('type') == 'Feature':
            features = [data]
        else:
            raise ValueError(f"Unsupported GeoJSON type: {data.get('type')}")

        for feature in features:
            parsed = self.parse_feature(feature)
            if parsed:
                yield parsed

    def parse_string(self, geojson_str: str) -> Iterator[Dict]:
        """Parse a GeoJSON string and yield spatial objects."""
        data = json.loads(geojson_str)

        if data.get('type') == 'FeatureCollection':
            features = data.get('features', [])
        elif data.get('type') == 'Feature':
            features = [data]
        else:
            raise ValueError(f"Unsupported GeoJSON type: {data.get('type')}")

        for feature in features:
            parsed = self.parse_feature(feature)
            if parsed:
                yield parsed

    @staticmethod
    def export_to_geojson(spatial_objects: List[Dict]) -> Dict:
        """Export spatial objects back to GeoJSON format."""
        features = []

        for obj in spatial_objects:
            geom = obj.get('geometry')
            if geom is None:
                continue

            feature = {
                'type': 'Feature',
                'geometry': mapping(geom),
                'properties': {
                    'object_type': obj.get('object_type'),
                    'label': obj.get('label'),
                    'confidence': obj.get('confidence'),
                    'area_pixels': obj.get('area_pixels'),
                    'perimeter_pixels': obj.get('perimeter_pixels'),
                    **obj.get('properties', {})
                }
            }
            features.append(feature)

        return {
            'type': 'FeatureCollection',
            'features': features
        }
