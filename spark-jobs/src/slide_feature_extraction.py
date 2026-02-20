"""
Spark job for extracting spatial features from cell distributions.
Used for downstream ML tasks like slide classification.
"""

import os
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors


def create_spark_session() -> SparkSession:
    """Create Spark session."""
    return SparkSession.builder \
        .appName("SpatialPathDB-FeatureExtraction") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def compute_spatial_grid_features(
    df: DataFrame,
    grid_size: float = 1000.0
) -> DataFrame:
    """
    Compute features based on spatial grid aggregation.
    Divides slide into grid cells and computes statistics per cell.
    """
    # Assign grid cell to each object
    df_with_grid = df.withColumn(
        "grid_x", F.floor(F.col("centroid_x") / grid_size).cast("int")
    ).withColumn(
        "grid_y", F.floor(F.col("centroid_y") / grid_size).cast("int")
    )

    # Compute per-grid-cell statistics
    grid_stats = df_with_grid.groupBy("slide_id", "grid_x", "grid_y").agg(
        F.count("*").alias("cell_count"),
        F.avg("confidence").alias("avg_confidence"),
        F.countDistinct("label").alias("label_diversity")
    )

    # Aggregate grid statistics per slide
    slide_grid_features = grid_stats.groupBy("slide_id").agg(
        F.count("*").alias("n_occupied_grids"),
        F.avg("cell_count").alias("mean_cells_per_grid"),
        F.stddev("cell_count").alias("std_cells_per_grid"),
        F.max("cell_count").alias("max_cells_per_grid"),
        F.avg("label_diversity").alias("mean_label_diversity"),

        # Gini coefficient approximation for cell distribution
        F.expr("percentile_approx(cell_count, 0.5)").alias("median_cells_per_grid")
    )

    # Compute spatial dispersion (coefficient of variation)
    slide_grid_features = slide_grid_features.withColumn(
        "spatial_dispersion",
        F.when(F.col("mean_cells_per_grid") > 0,
               F.col("std_cells_per_grid") / F.col("mean_cells_per_grid"))
         .otherwise(0)
    )

    return slide_grid_features


def compute_neighbor_features(df: DataFrame, radius: float = 50.0) -> DataFrame:
    """
    Compute features based on local neighborhood analysis.
    Note: Full neighbor analysis requires spatial joins which are expensive.
    This is a simplified version using grid-based approximation.
    """
    # Assign to fine grid for neighbor approximation
    fine_grid_size = radius * 2

    df_with_grid = df.withColumn(
        "fine_grid_x", F.floor(F.col("centroid_x") / fine_grid_size).cast("int")
    ).withColumn(
        "fine_grid_y", F.floor(F.col("centroid_y") / fine_grid_size).cast("int")
    )

    # Count cells in each fine grid
    grid_counts = df_with_grid.groupBy("slide_id", "fine_grid_x", "fine_grid_y").agg(
        F.count("*").alias("local_count"),
        F.collect_set("label").alias("local_labels")
    )

    # Compute statistics across grids
    neighbor_features = grid_counts.groupBy("slide_id").agg(
        F.avg("local_count").alias("avg_local_density"),
        F.stddev("local_count").alias("std_local_density"),
        F.avg(F.size("local_labels")).alias("avg_local_label_diversity")
    )

    return neighbor_features


def compute_morphology_features(df: DataFrame) -> DataFrame:
    """
    Compute morphological features from cell geometry statistics.
    """
    morph_features = df.groupBy("slide_id").agg(
        # Area statistics
        F.avg("area_pixels").alias("mean_area"),
        F.stddev("area_pixels").alias("std_area"),
        F.min("area_pixels").alias("min_area"),
        F.max("area_pixels").alias("max_area"),
        F.expr("percentile_approx(area_pixels, 0.25)").alias("area_q1"),
        F.expr("percentile_approx(area_pixels, 0.75)").alias("area_q3"),

        # Perimeter statistics
        F.avg("perimeter_pixels").alias("mean_perimeter"),
        F.stddev("perimeter_pixels").alias("std_perimeter"),
    )

    # Compute circularity-related features
    morph_features = morph_features.withColumn(
        "area_iqr", F.col("area_q3") - F.col("area_q1")
    ).withColumn(
        "area_cv",
        F.when(F.col("mean_area") > 0, F.col("std_area") / F.col("mean_area"))
         .otherwise(0)
    )

    return morph_features


def compute_label_distribution_features(df: DataFrame) -> DataFrame:
    """
    Compute features from cell type distribution.
    """
    total_counts = df.groupBy("slide_id").agg(
        F.count("*").alias("total_cells")
    )

    label_counts = df.groupBy("slide_id", "label").agg(
        F.count("*").alias("label_count")
    )

    # Pivot to get counts per label
    label_pivot = label_counts.groupBy("slide_id").pivot("label").agg(
        F.first("label_count")
    ).na.fill(0)

    # Join with total to compute ratios
    label_features = label_pivot.join(total_counts, "slide_id")

    # Get label columns (excluding slide_id and total_cells)
    label_cols = [c for c in label_features.columns if c not in ["slide_id", "total_cells"]]

    # Compute ratios
    for col in label_cols:
        label_features = label_features.withColumn(
            f"ratio_{col}",
            F.when(F.col("total_cells") > 0, F.col(col) / F.col("total_cells"))
             .otherwise(0)
        )

    # Compute entropy of label distribution
    entropy_expr = F.lit(0.0)
    for col in label_cols:
        ratio_col = f"ratio_{col}"
        entropy_expr = entropy_expr - F.when(
            F.col(ratio_col) > 0,
            F.col(ratio_col) * F.log2(F.col(ratio_col))
        ).otherwise(0)

    label_features = label_features.withColumn("label_entropy", entropy_expr)

    return label_features


def assemble_feature_vector(
    df: DataFrame,
    feature_cols: List[str],
    output_col: str = "features"
) -> DataFrame:
    """
    Assemble numeric columns into a feature vector for ML.
    """
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=output_col,
        handleInvalid="skip"
    )

    return assembler.transform(df)


def run_feature_extraction(slide_ids: Optional[List[str]] = None):
    """
    Main entry point for feature extraction job.
    """
    spark = create_spark_session()

    try:
        # Load data
        from batch_spatial_join import load_spatial_objects

        print("Loading spatial objects...")
        df = load_spatial_objects(spark, slide_ids, object_type="cell")
        df.cache()

        print(f"Processing {df.count()} objects")

        # Compute feature groups
        print("Computing spatial grid features...")
        grid_features = compute_spatial_grid_features(df)

        print("Computing morphology features...")
        morph_features = compute_morphology_features(df)

        print("Computing label distribution features...")
        label_features = compute_label_distribution_features(df)

        # Join all features
        all_features = grid_features.join(morph_features, "slide_id") \
                                    .join(label_features, "slide_id")

        # Select numeric columns for feature vector
        numeric_cols = [
            "n_occupied_grids", "mean_cells_per_grid", "spatial_dispersion",
            "mean_area", "std_area", "area_cv",
            "mean_perimeter", "label_entropy"
        ]

        # Add ratio columns
        ratio_cols = [c for c in all_features.columns if c.startswith("ratio_")]
        feature_cols = numeric_cols + ratio_cols

        # Filter to only existing columns
        existing_cols = [c for c in feature_cols if c in all_features.columns]

        print(f"Assembling {len(existing_cols)} features...")
        final_df = assemble_feature_vector(all_features, existing_cols)

        # Show results
        print("\nExtracted Features:")
        final_df.select("slide_id", "total_cells", "label_entropy", "spatial_dispersion").show()

        return final_df

    finally:
        spark.stop()


if __name__ == "__main__":
    run_feature_extraction()
