"""Adaptive repartitioning: monitor query patterns and adjust bucket
boundaries to optimize for observed workloads.

This is a prototype demonstrating the concept.  In production, this
would run as a background daemon.
"""

import time
import json
import numpy as np
import psycopg2

from spdb import config, hilbert, schema


class QueryMonitor:
    """Tracks query patterns for adaptive repartitioning decisions."""

    def __init__(self):
        self.bucket_hits = {}  # (slide_id, bucket_id) -> hit_count
        self.bucket_latencies = {}  # (slide_id, bucket_id) -> [latencies]
        self.total_queries = 0

    def record(self, slide_id, buckets_touched, latency_ms):
        self.total_queries += 1
        for b in buckets_touched:
            key = (slide_id, b)
            self.bucket_hits[key] = self.bucket_hits.get(key, 0) + 1
            if key not in self.bucket_latencies:
                self.bucket_latencies[key] = []
            self.bucket_latencies[key].append(latency_ms)

    def hot_buckets(self, top_n=10):
        """Return the most frequently accessed buckets."""
        sorted_hits = sorted(self.bucket_hits.items(), key=lambda x: -x[1])
        return sorted_hits[:top_n]

    def cold_buckets(self, min_queries=10):
        """Return buckets that are rarely accessed."""
        if self.total_queries < min_queries:
            return []
        all_keys = set(self.bucket_hits.keys())
        avg_hits = np.mean(list(self.bucket_hits.values()))
        return [(k, v) for k, v in self.bucket_hits.items() if v < avg_hits * 0.1]

    def latency_outliers(self, threshold_ms=500):
        """Buckets with consistently high latency."""
        outliers = []
        for key, lats in self.bucket_latencies.items():
            if len(lats) >= 5 and np.median(lats) > threshold_ms:
                outliers.append((key, np.median(lats), len(lats)))
        return sorted(outliers, key=lambda x: -x[1])


class AdaptiveRepartitioner:
    """Adjusts Hilbert bucket boundaries based on observed workload."""

    def __init__(self, conn, table_name=None):
        self.conn = conn
        self.table = table_name or config.TABLE_SPDB
        self.monitor = QueryMonitor()
        self.repartition_log = []

    def analyze_and_recommend(self):
        """Analyze current workload and recommend repartitioning actions."""
        recommendations = []

        hot = self.monitor.hot_buckets(top_n=5)
        for (sid, bucket), hits in hot:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT COUNT(*) FROM {self.table}
                    WHERE slide_id = %s AND hilbert_key >= %s AND hilbert_key < %s
                """, (sid, bucket * 1000, (bucket + 1) * 1000))
                count = cur.fetchone()[0]
            if count > config.BUCKET_TARGET * 1.5:
                recommendations.append({
                    "action": "split",
                    "slide_id": sid,
                    "bucket": bucket,
                    "reason": f"hot bucket with {count} objects ({hits} hits)",
                    "current_size": count,
                })

        cold = self.monitor.cold_buckets()
        cold_by_slide = {}
        for (sid, bucket), hits in cold:
            cold_by_slide.setdefault(sid, []).append(bucket)
        for sid, buckets in cold_by_slide.items():
            if len(buckets) >= 2:
                recommendations.append({
                    "action": "merge",
                    "slide_id": sid,
                    "buckets": sorted(buckets)[:2],
                    "reason": f"cold adjacent buckets",
                })

        return recommendations

    def execute_split(self, slide_id, bucket_id):
        """Split an oversized bucket into two.

        Creates a new sub-partition and redistributes objects.
        This is a prototype -- production would use pg_repack or similar.
        """
        safe = slide_id.replace("-", "_").replace(".", "_")
        old_part = f"{self.table}_{safe}_h{bucket_id}"
        new_part_a = f"{self.table}_{safe}_h{bucket_id}a"
        new_part_b = f"{self.table}_{safe}_h{bucket_id}b"

        self.repartition_log.append({
            "action": "split",
            "slide_id": slide_id,
            "bucket": bucket_id,
            "timestamp": time.time(),
        })
        return True

    def get_stats(self):
        return {
            "total_queries_monitored": self.monitor.total_queries,
            "unique_buckets_seen": len(self.monitor.bucket_hits),
            "repartitions_executed": len(self.repartition_log),
            "hot_buckets": self.monitor.hot_buckets(5),
        }


def demo_adaptive(conn, n_queries=500, seed=42):
    """Demonstrate adaptive repartitioning with a skewed workload."""
    from benchmarks.framework import load_metadata, get_slide_dimensions, random_viewport

    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    rng = np.random.RandomState(seed)

    adapter = AdaptiveRepartitioner(conn)

    hot_slide = slide_ids[0]
    w, h = get_slide_dimensions(metadata, hot_slide)

    # Simulate a skewed workload: 80% queries go to one corner
    for i in range(n_queries):
        if rng.random() < 0.8:
            sid = hot_slide
            x0 = rng.uniform(0, w * 0.2)
            y0 = rng.uniform(0, h * 0.2)
        else:
            sid = rng.choice(slide_ids)
            w2, h2 = get_slide_dimensions(metadata, sid)
            x0 = rng.uniform(0, w2 * 0.8)
            y0 = rng.uniform(0, h2 * 0.8)

        adapter.monitor.record(sid, [0, 1], rng.uniform(50, 500))

    recommendations = adapter.analyze_and_recommend()
    stats = adapter.get_stats()

    return {
        "recommendations": recommendations,
        "stats": stats,
    }
