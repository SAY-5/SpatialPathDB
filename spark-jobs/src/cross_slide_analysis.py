"""
Cross-slide analysis job for cohort-level spatial statistics.
Aggregates and compares spatial distributions across multiple slides.
"""

import os
from typing import List, Optional, Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


def create_spark_session() -> SparkSession:
    """Create optimized Spark session."""
    return SparkSession.builder \
        .appName("SpatialPathDB-CrossSlideAnalysis") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def compute_cohort_statistics(df: DataFrame) -> Dict:
    """
    Compute aggregate statistics across all slides in cohort.
    """
    stats = df.agg(
        F.count("*").alias("total_objects"),
        F.countDistinct("slide_id").alias("n_slides"),
        F.countDistinct("label").alias("unique_labels"),

        # Global averages
        F.avg("area_pixels").alias("global_avg_area"),
        F.avg("confidence").alias("global_avg_confidence"),

        # Spatial extent
        F.min("centroid_x").alias("global_min_x"),
        F.max("centroid_x").alias("global_max_x"),
        F.min("centroid_y").alias("global_min_y"),
        F.max("centroid_y").alias("global_max_y"),
    ).collect()[0]

    return {
        "total_objects": stats["total_objects"],
        "n_slides": stats["n_slides"],
        "unique_labels": stats["unique_labels"],
        "global_avg_area": stats["global_avg_area"],
        "global_avg_confidence": stats["global_avg_confidence"],
        "spatial_extent": {
            "min_x": stats["global_min_x"],
            "max_x": stats["global_max_x"],
            "min_y": stats["global_min_y"],
            "max_y": stats["global_max_y"],
        }
    }


def compute_label_prevalence(df: DataFrame) -> DataFrame:
    """
    Compute prevalence of each label across slides.
    """
    # Total objects per label
    label_totals = df.groupBy("label").agg(
        F.count("*").alias("total_count"),
        F.countDistinct("slide_id").alias("slides_with_label"),
        F.avg("confidence").alias("avg_confidence"),
        F.avg("area_pixels").alias("avg_area")
    )

    # Total objects overall
    total = df.count()

    # Compute prevalence
    label_totals = label_totals.withColumn(
        "prevalence", F.col("total_count") / F.lit(total)
    )

    return label_totals.orderBy(F.desc("total_count"))


def compute_slide_rankings(df: DataFrame) -> DataFrame:
    """
    Rank slides by various metrics for quality control.
    """
    slide_stats = df.groupBy("slide_id").agg(
        F.count("*").alias("object_count"),
        F.avg("confidence").alias("avg_confidence"),
        F.avg("area_pixels").alias("avg_area"),
        F.countDistinct("label").alias("label_diversity"),
        F.stddev("confidence").alias("confidence_std"),
    )

    # Compute percentile ranks
    from pyspark.sql.window import Window

    for metric in ["object_count", "avg_confidence", "label_diversity"]:
        window = Window.orderBy(F.col(metric))
        slide_stats = slide_stats.withColumn(
            f"{metric}_rank",
            F.percent_rank().over(window)
        )

    return slide_stats


def find_representative_slides(df: DataFrame, n_slides: int = 5) -> DataFrame:
    """
    Identify slides that are most representative of the cohort.
    Uses distance from cohort centroid in feature space.
    """
    # Compute per-slide features
    slide_features = df.groupBy("slide_id").agg(
        F.count("*").alias("n_objects"),
        F.avg("area_pixels").alias("avg_area"),
        F.avg("confidence").alias("avg_conf"),
    )

    # Compute cohort means
    cohort_means = slide_features.agg(
        F.avg("n_objects").alias("mean_n"),
        F.avg("avg_area").alias("mean_area"),
        F.avg("avg_conf").alias("mean_conf"),
        F.stddev("n_objects").alias("std_n"),
        F.stddev("avg_area").alias("std_area"),
        F.stddev("avg_conf").alias("std_conf"),
    ).collect()[0]

    # Compute z-scores for each slide
    slide_features = slide_features.withColumn(
        "z_n", (F.col("n_objects") - cohort_means["mean_n"]) /
               F.lit(cohort_means["std_n"] or 1)
    ).withColumn(
        "z_area", (F.col("avg_area") - cohort_means["mean_area"]) /
                  F.lit(cohort_means["std_area"] or 1)
    ).withColumn(
        "z_conf", (F.col("avg_conf") - cohort_means["mean_conf"]) /
                  F.lit(cohort_means["std_conf"] or 1)
    )

    # Distance from centroid (all z-scores = 0)
    slide_features = slide_features.withColumn(
        "distance_from_centroid",
        F.sqrt(F.pow("z_n", 2) + F.pow("z_area", 2) + F.pow("z_conf", 2))
    )

    # Select most representative (closest to centroid)
    return slide_features.orderBy("distance_from_centroid").limit(n_slides)


def compute_spatial_heterogeneity(df: DataFrame, grid_size: float = 2000.0) -> DataFrame:
    """
    Compute spatial heterogeneity metrics per slide.
    Measures how evenly cells are distributed spatially.
    """
    # Assign grid cells
    df_grid = df.withColumn(
        "grid_id",
        F.concat(
            F.floor(F.col("centroid_x") / grid_size).cast("string"),
            F.lit("_"),
            F.floor(F.col("centroid_y") / grid_size).cast("string")
        )
    )

    # Count per grid cell
    grid_counts = df_grid.groupBy("slide_id", "grid_id").agg(
        F.count("*").alias("cell_count")
    )

    # Compute heterogeneity metrics per slide
    heterogeneity = grid_counts.groupBy("slide_id").agg(
        F.count("*").alias("n_grids"),
        F.avg("cell_count").alias("mean_count"),
        F.stddev("cell_count").alias("std_count"),
        F.max("cell_count").alias("max_count"),
        F.min("cell_count").alias("min_count"),
    )

    # Coefficient of variation (higher = more heterogeneous)
    heterogeneity = heterogeneity.withColumn(
        "heterogeneity_cv",
        F.when(F.col("mean_count") > 0, F.col("std_count") / F.col("mean_count"))
         .otherwise(0)
    )

    # Gini-like measure
    heterogeneity = heterogeneity.withColumn(
        "heterogeneity_range",
        (F.col("max_count") - F.col("min_count")) / (F.col("max_count") + 1)
    )

    return heterogeneity


def run_cross_slide_analysis(slide_ids: Optional[List[str]] = None):
    """
    Main entry point for cross-slide analysis.
    """
    spark = create_spark_session()

    try:
        from batch_spatial_join import load_spatial_objects

        print("Loading spatial data...")
        df = load_spatial_objects(spark, slide_ids, object_type="cell")
        df.cache()

        print(f"Analyzing {df.count()} objects across slides")

        # Compute analyses
        print("\n1. Cohort Statistics:")
        cohort_stats = compute_cohort_statistics(df)
        for k, v in cohort_stats.items():
            print(f"   {k}: {v}")

        print("\n2. Label Prevalence:")
        prevalence = compute_label_prevalence(df)
        prevalence.show(10)

        print("\n3. Slide Rankings:")
        rankings = compute_slide_rankings(df)
        rankings.select(
            "slide_id", "object_count", "avg_confidence",
            "object_count_rank", "avg_confidence_rank"
        ).show(10)

        print("\n4. Representative Slides:")
        representative = find_representative_slides(df, n_slides=5)
        representative.select(
            "slide_id", "n_objects", "avg_area", "distance_from_centroid"
        ).show()

        print("\n5. Spatial Heterogeneity:")
        heterogeneity = compute_spatial_heterogeneity(df)
        heterogeneity.select(
            "slide_id", "n_grids", "heterogeneity_cv", "heterogeneity_range"
        ).orderBy(F.desc("heterogeneity_cv")).show(10)

        return {
            "cohort_stats": cohort_stats,
            "prevalence": prevalence,
            "rankings": rankings,
            "representative": representative,
            "heterogeneity": heterogeneity
        }

    finally:
        spark.stop()


if __name__ == "__main__":
    run_cross_slide_analysis()
