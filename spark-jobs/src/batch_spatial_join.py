"""
PySpark job for large-scale spatial join operations.
Performs distributed spatial analysis across multiple slides.
"""

import os
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    LongType, TimestampType
)


def create_spark_session(app_name: str = "SpatialPathDB-BatchJoin") -> SparkSession:
    """Create Spark session with optimized configuration."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g")) \
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "4g")) \
        .getOrCreate()


def get_jdbc_properties() -> dict:
    """Get JDBC connection properties."""
    return {
        "driver": "org.postgresql.Driver",
        "user": os.getenv("POSTGRES_USER", "pathdb_user"),
        "password": os.getenv("POSTGRES_PASSWORD", "pathdb_pass"),
    }


def get_jdbc_url() -> str:
    """Build JDBC URL from environment."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "spatialpathdb")
    return f"jdbc:postgresql://{host}:{port}/{db}"


def load_spatial_objects(
    spark: SparkSession,
    slide_ids: Optional[List[str]] = None,
    object_type: Optional[str] = None
) -> DataFrame:
    """
    Load spatial objects from PostgreSQL into Spark DataFrame.
    Uses predicate pushdown for efficient filtering.
    """
    jdbc_url = get_jdbc_url()
    props = get_jdbc_properties()

    # Build query with filters
    base_query = """
        SELECT
            id,
            slide_id,
            object_type,
            label,
            confidence,
            ST_X(centroid) as centroid_x,
            ST_Y(centroid) as centroid_y,
            area_pixels,
            perimeter_pixels,
            created_at
        FROM spatial_objects
    """

    conditions = []
    if slide_ids:
        ids_str = ", ".join(f"'{sid}'" for sid in slide_ids)
        conditions.append(f"slide_id IN ({ids_str})")
    if object_type:
        conditions.append(f"object_type = '{object_type}'")

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    return spark.read.jdbc(
        url=jdbc_url,
        table=f"({base_query}) as spatial_data",
        properties=props
    )


def extract_slide_features(df: DataFrame) -> DataFrame:
    """
    Extract spatial features from cell distribution per slide.
    Creates a feature vector for each slide.
    """
    # Define cell type columns
    cell_types = ['epithelial', 'stromal', 'lymphocyte', 'macrophage', 'necrotic']

    # Compute aggregations per slide
    features = df.groupBy("slide_id").agg(
        F.count("*").alias("total_objects"),
        F.countDistinct("label").alias("unique_labels"),
        F.avg("area_pixels").alias("avg_cell_area"),
        F.stddev("area_pixels").alias("std_cell_area"),
        F.avg("confidence").alias("avg_confidence"),
        F.stddev("confidence").alias("std_confidence"),

        # Spatial extent
        F.min("centroid_x").alias("min_x"),
        F.max("centroid_x").alias("max_x"),
        F.min("centroid_y").alias("min_y"),
        F.max("centroid_y").alias("max_y"),

        # Centroid of all cells
        F.avg("centroid_x").alias("mean_x"),
        F.avg("centroid_y").alias("mean_y"),

        # Cell type counts
        *[
            F.sum(F.when(F.col("label") == ct, 1).otherwise(0)).alias(f"count_{ct}")
            for ct in cell_types
        ]
    )

    # Compute derived features
    features = features.withColumn(
        "spatial_extent_x", F.col("max_x") - F.col("min_x")
    ).withColumn(
        "spatial_extent_y", F.col("max_y") - F.col("min_y")
    ).withColumn(
        "spatial_area", F.col("spatial_extent_x") * F.col("spatial_extent_y")
    ).withColumn(
        "cell_density",
        F.when(F.col("spatial_area") > 0, F.col("total_objects") / F.col("spatial_area"))
         .otherwise(0)
    )

    # Compute cell type ratios
    for ct in cell_types:
        features = features.withColumn(
            f"ratio_{ct}",
            F.when(F.col("total_objects") > 0, F.col(f"count_{ct}") / F.col("total_objects"))
             .otherwise(0)
        )

    return features


def compute_slide_similarity(features: DataFrame) -> DataFrame:
    """
    Compute pairwise similarity between slides based on feature vectors.
    Uses Euclidean distance on normalized features.
    """
    # Select numeric features for comparison
    feature_cols = [
        "avg_cell_area", "avg_confidence", "cell_density",
        "ratio_epithelial", "ratio_stromal", "ratio_lymphocyte",
        "ratio_macrophage", "ratio_necrotic"
    ]

    # Normalize features (simple min-max scaling)
    for col in feature_cols:
        min_val = features.agg(F.min(col)).collect()[0][0] or 0
        max_val = features.agg(F.max(col)).collect()[0][0] or 1
        range_val = max_val - min_val if max_val > min_val else 1

        features = features.withColumn(
            f"{col}_norm",
            (F.col(col) - F.lit(min_val)) / F.lit(range_val)
        )

    # Self-join to compute all pairs (for small number of slides)
    left = features.select(
        F.col("slide_id").alias("slide_id_1"),
        *[F.col(f"{c}_norm").alias(f"{c}_1") for c in feature_cols]
    )

    right = features.select(
        F.col("slide_id").alias("slide_id_2"),
        *[F.col(f"{c}_norm").alias(f"{c}_2") for c in feature_cols]
    )

    pairs = left.crossJoin(right).filter(
        F.col("slide_id_1") < F.col("slide_id_2")
    )

    # Compute Euclidean distance
    dist_expr = F.sqrt(
        sum(
            F.pow(F.col(f"{c}_1") - F.col(f"{c}_2"), 2)
            for c in feature_cols
        )
    )

    pairs = pairs.withColumn("distance", dist_expr)
    pairs = pairs.withColumn(
        "similarity",
        1 / (1 + F.col("distance"))  # Convert distance to similarity
    )

    return pairs.select("slide_id_1", "slide_id_2", "distance", "similarity")


def identify_outlier_slides(features: DataFrame, threshold: float = 2.0) -> DataFrame:
    """
    Identify slides that are statistical outliers based on features.
    Uses z-score based outlier detection.
    """
    feature_cols = ["total_objects", "avg_cell_area", "cell_density", "avg_confidence"]

    for col in feature_cols:
        mean_val = features.agg(F.avg(col)).collect()[0][0]
        std_val = features.agg(F.stddev(col)).collect()[0][0] or 1

        features = features.withColumn(
            f"{col}_zscore",
            (F.col(col) - F.lit(mean_val)) / F.lit(std_val)
        )

    # Mark outliers
    outlier_condition = F.lit(False)
    for col in feature_cols:
        outlier_condition = outlier_condition | (
            F.abs(F.col(f"{col}_zscore")) > threshold
        )

    features = features.withColumn("is_outlier", outlier_condition)

    return features


def write_results_to_db(df: DataFrame, table_name: str, mode: str = "overwrite"):
    """Write Spark DataFrame back to PostgreSQL."""
    jdbc_url = get_jdbc_url()
    props = get_jdbc_properties()

    df.write.jdbc(
        url=jdbc_url,
        table=table_name,
        mode=mode,
        properties=props
    )


def run_batch_analysis(slide_ids: Optional[List[str]] = None):
    """
    Main entry point for batch spatial analysis.
    """
    spark = create_spark_session()

    try:
        print("Loading spatial objects...")
        objects_df = load_spatial_objects(spark, slide_ids, object_type="cell")
        objects_df.cache()

        print(f"Loaded {objects_df.count()} spatial objects")

        print("Extracting slide features...")
        features_df = extract_slide_features(objects_df)
        features_df.cache()

        print("Computing slide similarity...")
        similarity_df = compute_slide_similarity(features_df)

        print("Identifying outlier slides...")
        outliers_df = identify_outlier_slides(features_df)

        # Show results
        print("\nSlide Features:")
        features_df.select(
            "slide_id", "total_objects", "avg_cell_area",
            "cell_density", "ratio_epithelial", "ratio_lymphocyte"
        ).show(10)

        print("\nMost Similar Slide Pairs:")
        similarity_df.orderBy(F.desc("similarity")).show(10)

        print("\nOutlier Slides:")
        outliers_df.filter(F.col("is_outlier")).select(
            "slide_id", "total_objects", "is_outlier"
        ).show(10)

        # Optionally write results back to database
        # write_results_to_db(features_df, "slide_features")

        return features_df, similarity_df, outliers_df

    finally:
        spark.stop()


if __name__ == "__main__":
    import sys
    slide_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    run_batch_analysis(slide_ids)
