#!/usr/bin/env python3
"""
Generate synthetic spatial data for SpatialPathDB.

Creates realistic cell and tissue region annotations simulating whole-slide
pathology image analysis outputs. Uses Gaussian mixture models for spatial
clustering.

Usage:
    python scripts/generate_synthetic_data.py --slides 5 --cells-per-slide 500000
    python scripts/generate_synthetic_data.py --slides 1 --cells-per-slide 1000000 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'spatial-engine'))

from src.data_ingestion.synthetic_generator import SyntheticSlideGenerator, SlideConfig
from src.data_ingestion.db_loader import BulkDataLoader, verify_data_integrity


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate synthetic spatial data for SpatialPathDB'
    )
    parser.add_argument(
        '--slides', type=int, default=5,
        help='Number of slides to generate (default: 5)'
    )
    parser.add_argument(
        '--cells-per-slide', type=int, default=500000,
        help='Target cells per slide (default: 500000)'
    )
    parser.add_argument(
        '--clusters', type=int, default=25,
        help='Number of cell clusters per slide (default: 25)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-tissue-regions', action='store_true',
        help='Skip generating tissue region boundaries'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify data integrity after loading'
    )
    return parser.parse_args()


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def main():
    args = parse_args()

    print("=" * 60)
    print("SpatialPathDB - Synthetic Data Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Slides to generate: {args.slides}")
    print(f"  Cells per slide:    {format_number(args.cells_per_slide)}")
    print(f"  Clusters per slide: {args.clusters}")
    print(f"  Random seed:        {args.seed}")
    print(f"  Include tissue:     {not args.no_tissue_regions}")
    print()

    loader = BulkDataLoader(batch_size=10000)
    total_objects = 0
    total_time = 0

    # Slide configuration variations
    organs = ['Breast', 'Lung', 'Colon', 'Liver', 'Kidney', 'Prostate', 'Ovary', 'Pancreas']
    stains = ['H&E', 'IHC-Ki67', 'IHC-CD3', 'IHC-CD8', 'IHC-PD-L1']

    for i in range(args.slides):
        print("-" * 60)
        print(f"Generating slide {i+1}/{args.slides}")

        # Vary slide parameters
        import numpy as np
        rng = np.random.default_rng(args.seed + i)

        config = SlideConfig(
            width=rng.integers(80000, 120000),
            height=rng.integers(60000, 100000),
            microns_per_pixel=float(rng.choice([0.25, 0.5, 0.125])),
            stain_type=stains[i % len(stains)],
            organ=organs[i % len(organs)]
        )

        print(f"  Organ: {config.organ}, Stain: {config.stain_type}")
        print(f"  Dimensions: {config.width}x{config.height} pixels")

        generator = SyntheticSlideGenerator(
            config,
            seed=args.seed + i
        )

        # Generate slide
        start_time = time.time()

        slide_meta, objects = generator.generate_full_slide(
            n_cells=args.cells_per_slide,
            n_clusters=args.clusters,
            include_tissue_regions=not args.no_tissue_regions,
            progress=True
        )

        gen_time = time.time() - start_time
        print(f"  Generated {len(objects)} objects in {gen_time:.1f}s")

        # Load to database
        print("  Loading to database...")
        load_start = time.time()

        try:
            result = loader.load_synthetic_slide(slide_meta, objects, progress=True)
            load_time = time.time() - load_start

            print(f"  Loaded {result['objects_inserted']} objects in {load_time:.1f}s")
            print(f"  Rate: {result['objects_per_second']:.0f} objects/sec")

            total_objects += result['objects_inserted']
            total_time += gen_time + load_time

            # Verify if requested
            if args.verify:
                print("  Verifying data integrity...")
                integrity = verify_data_integrity(result['slide_id'])
                print(f"    Invalid geometries: {integrity['invalid_geometries']}")
                print(f"    Missing centroids:  {integrity['missing_centroids']}")
                print(f"    Index scan used:    {integrity['index_scan_used']}")

        except Exception as e:
            print(f"  ERROR loading slide: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total slides created:  {args.slides}")
    print(f"Total objects loaded:  {format_number(total_objects)}")
    print(f"Total time:            {total_time:.1f}s")
    print(f"Average rate:          {total_objects/total_time:.0f} objects/sec")
    print()

    # Show sample query
    print("Test with a sample spatial query:")
    print("  psql -d spatialpathdb -c \"SELECT COUNT(*) FROM spatial_objects;\"")
    print()


if __name__ == '__main__':
    main()
