"""Multi-dataset support for cross-cancer and non-pathology evaluation.

Extends SPDB beyond BLCA to BRCA, LUAD, COAD cancer types from TCGA,
and adds OpenStreetMap building data as a non-pathology baseline.
"""

import os
import json
import time

import numpy as np
import pandas as pd

from spdb import config, hilbert, zorder, schema
from spdb.ingest import (
    download_patient, transform_patient, _copy_chunk_numpy,
    setup_schemas, build_indexes,
)

# ---------------------------------------------------------------------------
# TCGA cancer type metadata
# ---------------------------------------------------------------------------

CANCER_TYPES = {
    "BLCA": {"name": "Bladder Urothelial Carcinoma", "prefix": "TCGA-BLCA"},
    "BRCA": {"name": "Breast Invasive Carcinoma", "prefix": "TCGA-BRCA"},
    "LUAD": {"name": "Lung Adenocarcinoma", "prefix": "TCGA-LUAD"},
    "COAD": {"name": "Colon Adenocarcinoma", "prefix": "TCGA-COAD"},
    "LUSC": {"name": "Lung Squamous Cell Carcinoma", "prefix": "TCGA-LUSC"},
    "UCEC": {"name": "Uterine Corpus Endometrial Carcinoma", "prefix": "TCGA-UCEC"},
}


def list_patients_by_cancer(cancer_type="BLCA"):
    """List patients from HuggingFace dataset filtered by cancer type.

    The pan-cancer dataset organizes by bcr_patient_barcode which encodes
    the cancer type in the TCGA barcode (positions 5-8).
    """
    from huggingface_hub import HfApi

    if cancer_type not in CANCER_TYPES:
        raise ValueError(f"Unknown cancer type: {cancer_type}. "
                         f"Known: {list(CANCER_TYPES.keys())}")

    api = HfApi()
    items = list(api.list_repo_tree(config.HF_DATASET, repo_type="dataset"))
    prefix = CANCER_TYPES[cancer_type]["prefix"]

    patients = []
    for x in items:
        path = getattr(x, "path", "")
        if "bcr_patient_barcode=" in path:
            # Extract barcode from path
            barcode = path.split("bcr_patient_barcode=")[-1].rstrip("/")
            if barcode.startswith(prefix.split("-")[-1]):
                patients.append(path)

    return sorted(patients)


def select_slides_multi(cancer_types=None, n_per_type=50, seed=42):
    """Select slides across multiple cancer types.

    Parameters
    ----------
    cancer_types : list of str
        Cancer type codes (e.g., ["BLCA", "BRCA", "LUAD"]).
    n_per_type : int
        Number of slides per cancer type.

    Returns
    -------
    dict mapping cancer_type -> list of patient directories
    """
    if cancer_types is None:
        cancer_types = ["BLCA", "BRCA", "LUAD", "COAD"]

    rng = np.random.RandomState(seed)
    selected = {}

    for ct in cancer_types:
        patients = list_patients_by_cancer(ct)
        if len(patients) == 0:
            print(f"  WARNING: No patients found for {ct}")
            continue

        if len(patients) <= n_per_type:
            selected[ct] = patients
        else:
            step = max(1, len(patients) // n_per_type)
            chosen = patients[::step][:n_per_type]
            selected[ct] = chosen

        print(f"  {ct}: {len(selected[ct])} slides selected "
              f"(from {len(patients)} available)")

    return selected


def ingest_multi_dataset(cancer_types=None, n_per_type=50, p=None,
                          bucket_target=None, seed=42):
    """Full ingestion pipeline for multi-cancer evaluation.

    Creates separate table sets per cancer type and a combined table.
    Returns structured metadata for benchmarking.
    """
    if cancer_types is None:
        cancer_types = ["BLCA", "BRCA", "LUAD", "COAD"]
    if p is None:
        p = config.HILBERT_ORDER
    if bucket_target is None:
        bucket_target = config.BUCKET_TARGET

    selected = select_slides_multi(cancer_types, n_per_type, seed)

    all_metadata = {}
    per_cancer_stats = {}

    for ct, patient_dirs in selected.items():
        print(f"\n{'='*50}")
        print(f"  Ingesting {ct} ({len(patient_dirs)} slides)")
        print(f"{'='*50}")

        ct_dfs = []
        ct_metas = {}
        ct_object_counts = {}
        ct_total = 0

        for patient_dir in patient_dirs:
            try:
                path = download_patient(patient_dir)
                df, meta = transform_patient(path, p=p, bucket_target=bucket_target)
                ct_dfs.append(df)
                ct_metas[meta["slide_id"]] = meta
                ct_object_counts[meta["slide_id"]] = meta["num_objects"]
                ct_total += meta["num_objects"]
            except Exception as e:
                print(f"    SKIP {patient_dir}: {e}")

        per_cancer_stats[ct] = {
            "n_slides": len(ct_dfs),
            "total_objects": ct_total,
            "slide_ids": list(ct_object_counts.keys()),
            "metas": ct_metas,
            "object_counts": ct_object_counts,
        }
        all_metadata[ct] = ct_metas

        print(f"  {ct}: {ct_total:,} objects across {len(ct_dfs)} slides")

    return {
        "cancer_types": cancer_types,
        "per_cancer": per_cancer_stats,
        "all_metadata": all_metadata,
    }


# ---------------------------------------------------------------------------
# OpenStreetMap building data adapter
# ---------------------------------------------------------------------------

class OSMDatasetAdapter:
    """Adapter for loading OpenStreetMap building footprints into SPDB schema.

    Non-pathology dataset for generalization evaluation.
    """

    def __init__(self, region_name="manhattan"):
        self.region_name = region_name
        self.buildings = None

    def download_osm_buildings(self, bbox=None, timeout=180):
        """Download building footprints via Overpass API.

        Parameters
        ----------
        bbox : tuple
            (south, west, north, east) in decimal degrees.
            Default: Manhattan, NYC.
        """
        if bbox is None:
            bbox = (40.7000, -74.0200, 40.8200, -73.9300)  # Manhattan

        import urllib.request
        import json as json_mod

        south, west, north, east = bbox
        query = f"""
        [out:json][timeout:{timeout}];
        (way["building"]({south},{west},{north},{east}););
        out center;
        """

        url = "http://overpass-api.de/api/interpreter"
        data = urllib.parse.urlencode({"data": query}).encode()
        req = urllib.request.Request(url, data=data)
        resp = urllib.request.urlopen(req, timeout=timeout)
        result = json_mod.loads(resp.read())

        buildings = []
        for element in result.get("elements", []):
            center = element.get("center", {})
            if "lat" in center and "lon" in center:
                tags = element.get("tags", {})
                buildings.append({
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "building_type": tags.get("building", "yes"),
                    "osm_id": element.get("id", 0),
                    "name": tags.get("name", ""),
                })

        self.buildings = pd.DataFrame(buildings)
        print(f"  Downloaded {len(self.buildings)} buildings for {self.region_name}")
        return self.buildings

    def transform_to_spdb_schema(self, p=None, bucket_target=None):
        """Convert OSM buildings to SPDB-compatible DataFrame.

        Maps:
        - building centroid (lat, lon) -> centroid_x, centroid_y (projected)
        - building_type -> class_label
        - region_name -> slide_id
        - grid tile -> tile_id
        """
        if self.buildings is None or len(self.buildings) == 0:
            raise ValueError("No buildings loaded. Call download_osm_buildings first.")

        if p is None:
            p = config.HILBERT_ORDER
        if bucket_target is None:
            bucket_target = config.BUCKET_TARGET

        df = self.buildings.copy()

        # Simple equirectangular projection (sufficient for city scale)
        lat_center = df["lat"].mean()
        lon_scale = np.cos(np.radians(lat_center))
        cx = (df["lon"].values - df["lon"].min()) * 111320 * lon_scale  # meters
        cy = (df["lat"].values - df["lat"].min()) * 110540  # meters

        w = float(cx.max() - cx.min()) + 1.0
        h = float(cy.max() - cy.min()) + 1.0

        gx, gy = hilbert.normalize_coords(cx, cy, w, h, p)
        h_keys = hilbert.encode_batch(gx, gy, p)

        zgx, zgy = zorder.normalize_coords(cx, cy, w, h, p)
        z_keys = zorder.encode_batch(zgx, zgy, p)

        # Tile grid (256m tiles)
        tile_size = 256.0
        tile_ids = [f"{int(x // tile_size)}_{int(y // tile_size)}"
                    for x, y in zip(cx, cy)]

        # Map building types to class labels
        type_map = {
            "residential": "Residential",
            "commercial": "Commercial",
            "industrial": "Industrial",
            "yes": "Unknown",
        }
        class_labels = [type_map.get(t, "Other")
                        for t in df["building_type"]]

        result = pd.DataFrame({
            "slide_id": self.region_name,
            "centroid_x": cx,
            "centroid_y": cy,
            "class_label": class_labels,
            "tile_id": tile_ids,
            "hilbert_key": h_keys,
            "zorder_key": z_keys,
            "area": 100.0,  # placeholder
            "perimeter": 40.0,  # placeholder
            "confidence": 1.0,
            "pipeline_id": "osm_overpass",
        })

        meta = {
            "slide_id": self.region_name,
            "image_width": w,
            "image_height": h,
            "num_objects": len(result),
            "num_buckets": max(1, len(result) // bucket_target),
            "dataset_type": "osm_buildings",
        }

        return result, meta


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Tracks loaded datasets for multi-dataset benchmarking."""

    def __init__(self, registry_path=None):
        if registry_path is None:
            registry_path = os.path.join(config.RESULTS_DIR, "dataset_registry.json")
        self.path = registry_path
        self._registry = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._registry = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register(self, name, dataset_type, n_slides, n_objects, metadata=None):
        self._registry[name] = {
            "dataset_type": dataset_type,
            "n_slides": n_slides,
            "n_objects": n_objects,
            "registered_at": time.time(),
            "metadata": metadata or {},
        }
        self._save()

    def list_datasets(self):
        return dict(self._registry)

    def get_metadata(self, name):
        return self._registry.get(name, {})
