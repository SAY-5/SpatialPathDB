#!/bin/bash
set -e

echo "SpatialPathDB Benchmark Runner"
echo "=============================="

cd "$(dirname "$0")/.."

# Check if database is running
if ! docker-compose ps postgres | grep -q "Up"; then
    echo "Starting database..."
    docker-compose up -d postgres redis
    sleep 5
fi

# Run benchmarks
echo "Running benchmarks..."
cd spatial-engine
python -m pip install -q -r requirements.txt
cd ..

python benchmarks/src/benchmark_spatial_queries.py \
    --iterations 100 \
    --output benchmarks/results/benchmark_$(date +%Y%m%d_%H%M%S).json

echo "Benchmarks complete!"
