"""
Bulk data loading utilities for PostgreSQL.
Uses COPY command for high-performance inserts (100x faster than INSERT).
"""

import io
import csv
import json
from typing import List, Dict, Iterator, Optional
from datetime import datetime

import psycopg2
from tqdm import tqdm

from ..utils.db_connection import get_db_connection


class BulkDataLoader:
    """
    High-performance bulk loader for spatial objects.
    Uses PostgreSQL COPY for maximum throughput.
    """

    SPATIAL_OBJECTS_COLUMNS = [
        'slide_id', 'object_type', 'label', 'confidence',
        'geometry', 'centroid', 'area_pixels', 'perimeter_pixels',
        'properties', 'created_at'
    ]

    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size

    def _format_geometry(self, geom) -> str:
        """Convert Shapely geometry to WKT for PostgreSQL."""
        if geom is None:
            return ''
        return geom.wkt

    def _format_row(self, obj: Dict, slide_id: str) -> List[str]:
        """Format a spatial object dict as a CSV row."""
        return [
            slide_id,
            obj.get('object_type', ''),
            obj.get('label', ''),
            str(obj.get('confidence', '')) if obj.get('confidence') is not None else '',
            self._format_geometry(obj.get('geometry')),
            self._format_geometry(obj.get('centroid')),
            str(obj.get('area_pixels', '')) if obj.get('area_pixels') is not None else '',
            str(obj.get('perimeter_pixels', '')) if obj.get('perimeter_pixels') is not None else '',
            json.dumps(obj.get('properties', {})),
            datetime.now().isoformat()
        ]

    def _create_csv_buffer(self, objects: List[Dict], slide_id: str) -> io.StringIO:
        """Create in-memory CSV buffer for COPY command."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL)

        for obj in objects:
            writer.writerow(self._format_row(obj, slide_id))

        buffer.seek(0)
        return buffer

    def insert_slide(self, slide_meta: Dict) -> str:
        """Insert slide metadata and return the slide ID."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO slides (
                        id, slide_name, file_path, width_pixels, height_pixels,
                        microns_per_pixel, stain_type, organ, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    slide_meta['id'],
                    slide_meta['slide_name'],
                    slide_meta['file_path'],
                    int(slide_meta['width_pixels']),
                    int(slide_meta['height_pixels']),
                    slide_meta.get('microns_per_pixel'),
                    slide_meta.get('stain_type'),
                    slide_meta.get('organ'),
                    json.dumps(slide_meta.get('metadata', {}))
                ))
                result = cur.fetchone()
                return str(result[0])

    def bulk_insert_objects(
        self,
        objects: List[Dict],
        slide_id: str,
        progress: bool = False
    ) -> int:
        """
        Bulk insert spatial objects using COPY.
        Returns number of rows inserted.
        """
        if not objects:
            return 0

        buffer = self._create_csv_buffer(objects, slide_id)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                columns = ', '.join(self.SPATIAL_OBJECTS_COLUMNS)
                cur.copy_expert(
                    f"""COPY spatial_objects ({columns})
                        FROM STDIN WITH (FORMAT csv, NULL '')""",
                    buffer
                )
                return len(objects)

    def bulk_insert_streaming(
        self,
        object_batches: Iterator[List[Dict]],
        slide_id: str,
        total_expected: Optional[int] = None,
        progress: bool = True
    ) -> int:
        """
        Stream-insert objects in batches for memory efficiency.
        Returns total number of rows inserted.
        """
        total_inserted = 0

        with get_db_connection() as conn:
            pbar = None
            if progress and total_expected:
                pbar = tqdm(total=total_expected, desc='Loading to database', unit='objects')

            try:
                for batch in object_batches:
                    buffer = self._create_csv_buffer(batch, slide_id)

                    with conn.cursor() as cur:
                        columns = ', '.join(self.SPATIAL_OBJECTS_COLUMNS)
                        cur.copy_expert(
                            f"""COPY spatial_objects ({columns})
                                FROM STDIN WITH (FORMAT csv, NULL '')""",
                            buffer
                        )

                    total_inserted += len(batch)
                    conn.commit()

                    if pbar:
                        pbar.update(len(batch))

            finally:
                if pbar:
                    pbar.close()

        return total_inserted

    def load_synthetic_slide(
        self,
        slide_meta: Dict,
        spatial_objects: List[Dict],
        progress: bool = True
    ) -> Dict:
        """
        Load a complete synthetic slide (metadata + objects).
        Returns summary statistics.
        """
        start_time = datetime.now()

        # Insert slide metadata
        slide_id = self.insert_slide(slide_meta)

        # Bulk insert spatial objects
        n_objects = self.bulk_insert_objects(spatial_objects, slide_id, progress)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'slide_id': slide_id,
            'objects_inserted': n_objects,
            'elapsed_seconds': elapsed,
            'objects_per_second': n_objects / elapsed if elapsed > 0 else 0
        }

    def get_slide_summary(self, slide_id: str) -> Dict:
        """Get summary statistics for a loaded slide."""
        with get_db_connection(dict_cursor=True) as conn:
            with conn.cursor() as cur:
                # Get object counts by type and label
                cur.execute("""
                    SELECT object_type, label, COUNT(*) as count,
                           AVG(confidence) as avg_confidence,
                           AVG(area_pixels) as avg_area
                    FROM spatial_objects
                    WHERE slide_id = %s
                    GROUP BY object_type, label
                    ORDER BY count DESC
                """, (slide_id,))
                type_counts = cur.fetchall()

                # Get total count
                cur.execute("""
                    SELECT COUNT(*) as total FROM spatial_objects
                    WHERE slide_id = %s
                """, (slide_id,))
                total = cur.fetchone()['total']

                return {
                    'slide_id': slide_id,
                    'total_objects': total,
                    'by_type_and_label': type_counts
                }


def verify_data_integrity(slide_id: str) -> Dict:
    """Verify that loaded data has correct geometry and indexes."""
    with get_db_connection(dict_cursor=True) as conn:
        with conn.cursor() as cur:
            # Check for valid geometries
            cur.execute("""
                SELECT COUNT(*) as invalid_count
                FROM spatial_objects
                WHERE slide_id = %s AND NOT ST_IsValid(geometry)
            """, (slide_id,))
            invalid_geom = cur.fetchone()['invalid_count']

            # Check centroid computation
            cur.execute("""
                SELECT COUNT(*) as missing_centroid
                FROM spatial_objects
                WHERE slide_id = %s AND centroid IS NULL
            """, (slide_id,))
            missing_centroid = cur.fetchone()['missing_centroid']

            # Test spatial index with sample bbox query
            cur.execute("""
                EXPLAIN (ANALYZE, FORMAT JSON)
                SELECT COUNT(*) FROM spatial_objects
                WHERE slide_id = %s
                  AND geometry && ST_MakeEnvelope(0, 0, 10000, 10000, 0)
            """, (slide_id,))
            explain_result = cur.fetchone()

            return {
                'invalid_geometries': invalid_geom,
                'missing_centroids': missing_centroid,
                'index_scan_used': 'Index' in str(explain_result)
            }
