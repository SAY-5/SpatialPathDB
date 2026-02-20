# SpatialPathDB API Documentation

Base URL: `http://localhost:8080/api/v1`

## Authentication

Currently no authentication required (development mode).

## Slides

### Create Slide

```http
POST /slides
Content-Type: application/json

{
  "slideName": "sample_slide_001",
  "filePath": "/data/slides/sample.svs",
  "widthPixels": 100000,
  "heightPixels": 80000,
  "micronsPerPixel": 0.25,
  "stainType": "H&E",
  "organ": "Breast",
  "metadata": {
    "patient_id": "P001",
    "diagnosis": "IDC"
  }
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "slideName": "sample_slide_001",
  "filePath": "/data/slides/sample.svs",
  "widthPixels": 100000,
  "heightPixels": 80000,
  "micronsPerPixel": 0.25,
  "stainType": "H&E",
  "organ": "Breast",
  "uploadedAt": "2024-01-15T10:30:00Z",
  "metadata": {...}
}
```

### List Slides

```http
GET /slides?page=0&size=20&organ=Breast
```

**Response:** `200 OK`
```json
{
  "content": [...],
  "totalElements": 50,
  "totalPages": 3,
  "size": 20,
  "number": 0
}
```

### Get Slide Details

```http
GET /slides/{slideId}
```

### Delete Slide

```http
DELETE /slides/{slideId}
```

**Response:** `204 No Content`

### Get Slide Statistics

```http
GET /slides/{slideId}/statistics
```

**Response:** `200 OK`
```json
{
  "slideId": "550e8400-e29b-41d4-a716-446655440000",
  "totalObjects": 1250000,
  "byType": [
    {
      "objectType": "cell",
      "label": "epithelial",
      "totalCount": 437500,
      "avgArea": 452.3,
      "stdArea": 85.2,
      "avgConfidence": 0.88
    },
    ...
  ]
}
```

### Get Slide Bounds

```http
GET /slides/{slideId}/bounds
```

**Response:** `200 OK`
```json
{
  "minX": 0,
  "minY": 0,
  "maxX": 99500,
  "maxY": 79800,
  "totalObjects": 1250000
}
```

---

## Spatial Queries

### Bounding Box Query

Find all objects within a rectangular region.

```http
POST /spatial/bbox
Content-Type: application/json

{
  "slideId": "550e8400-e29b-41d4-a716-446655440000",
  "minX": 10000,
  "minY": 10000,
  "maxX": 20000,
  "maxY": 20000,
  "objectType": "cell",
  "minConfidence": 0.8,
  "limit": 1000,
  "offset": 0
}
```

**Response:** `200 OK`
```json
[
  {
    "id": 12345,
    "objectType": "cell",
    "label": "epithelial",
    "confidence": 0.92,
    "geometryWkt": "POLYGON((10050 10030, 10062 10030, ...))",
    "centroidX": 10056.5,
    "centroidY": 10042.3,
    "areaPixels": 452.8,
    "properties": {
      "circularity": 0.85,
      "color": "#FF6B6B"
    }
  },
  ...
]
```

### K-Nearest Neighbors Query

Find the K closest objects to a point.

```http
POST /spatial/knn
Content-Type: application/json

{
  "slideId": "550e8400-e29b-41d4-a716-446655440000",
  "x": 50000,
  "y": 40000,
  "k": 10,
  "objectType": "cell"
}
```

**Response:** `200 OK`
```json
[
  {
    "id": 67890,
    "label": "lymphocyte",
    "confidence": 0.95,
    "centroidX": 50012.3,
    "centroidY": 39998.7,
    "distance": 15.2
  },
  ...
]
```

### Density Grid

Compute cell density heatmap.

```http
POST /spatial/density?slideId={id}&gridSize=256&objectType=cell
```

**Response:** `200 OK`
```json
[
  {
    "gridX": 0,
    "gridY": 0,
    "cellCount": 145,
    "avgConfidence": 0.87,
    "dominantLabel": "epithelial",
    "centerX": 128,
    "centerY": 128
  },
  ...
]
```

### Point-in-Polygon Query

Find objects within a polygon.

```http
POST /spatial/within?slideId={id}&polygonWkt=POLYGON((...)&limit=10000
```

### Count Objects

Lightweight count without returning data.

```http
POST /spatial/count
Content-Type: application/json

{
  "slideId": "...",
  "minX": 0,
  "minY": 0,
  "maxX": 50000,
  "maxY": 50000
}
```

**Response:** `200 OK`
```json
{
  "count": 625000
}
```

---

## Analysis Jobs

### Submit Job

```http
POST /jobs
Content-Type: application/json

{
  "slideId": "550e8400-e29b-41d4-a716-446655440000",
  "jobType": "cell_detection",
  "parameters": {
    "target_cells": 100000,
    "n_clusters": 15,
    "confidence_threshold": 0.5
  }
}
```

**Response:** `202 Accepted`
```json
{
  "id": "job-uuid",
  "slideId": "550e8400-e29b-41d4-a716-446655440000",
  "jobType": "cell_detection",
  "status": "QUEUED",
  "submittedAt": "2024-01-15T10:30:00Z"
}
```

### Job Types

| Type | Description | Parameters |
|------|-------------|------------|
| `cell_detection` | Detect and classify cells | `target_cells`, `n_clusters`, `confidence_threshold` |
| `tissue_segmentation` | Segment tissue regions | `n_regions`, `min_region_size`, `max_region_size` |
| `spatial_statistics` | Compute distribution metrics | `hotspot_grid_size`, `colocalization_radius` |

### Get Job Status

```http
GET /jobs/{jobId}
```

**Response:** `200 OK`
```json
{
  "id": "job-uuid",
  "slideId": "...",
  "jobType": "cell_detection",
  "status": "COMPLETED",
  "submittedAt": "2024-01-15T10:30:00Z",
  "startedAt": "2024-01-15T10:30:05Z",
  "completedAt": "2024-01-15T10:32:30Z",
  "durationMs": 145000,
  "resultSummary": {
    "total_cells_detected": 98500,
    "cell_type_breakdown": {...}
  }
}
```

### List Jobs

```http
GET /jobs?status=RUNNING&page=0&size=20
```

### Cancel Job

```http
DELETE /jobs/{jobId}
```

**Response:** `204 No Content`

### Get Job Counts

```http
GET /jobs/counts
```

**Response:** `200 OK`
```json
{
  "queued": 5,
  "running": 2,
  "completed": 150,
  "failed": 3,
  "cancelled": 1
}
```

---

## Benchmarks

### Get Benchmark Results

```http
GET /benchmarks/results?hoursBack=24
```

**Response:** `200 OK`
```json
{
  "period": {
    "since": "2024-01-14T10:30:00Z",
    "until": "2024-01-15T10:30:00Z"
  },
  "statsByQueryType": [
    {
      "queryType": "bbox",
      "totalQueries": 1250,
      "avgTimeMs": 45.2,
      "minTimeMs": 12.1,
      "maxTimeMs": 185.3,
      "p95TimeMs": 82.5,
      "p99TimeMs": 125.0
    },
    ...
  ],
  "hourlyStats": [...],
  "summary": {
    "totalQueries": 5000,
    "overallAvgTimeMs": 38.5
  }
}
```

---

## Health

### Health Check

```http
GET /health
```

**Response:** `200 OK`
```json
{
  "status": "UP",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": {
    "status": "UP",
    "database": "PostgreSQL",
    "version": "15.4"
  },
  "redis": {
    "status": "UP",
    "ping": "PONG"
  }
}
```

### Readiness Check

```http
GET /health/ready
```

### Liveness Check

```http
GET /health/live
```

---

## Error Responses

### 400 Bad Request
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Validation failed",
  "fieldErrors": {
    "minX": "minX is required",
    "slideId": "Slide ID is required"
  }
}
```

### 404 Not Found
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "status": 404,
  "error": "Not Found",
  "message": "Slide not found: abc123"
}
```

### 500 Internal Server Error
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "status": 500,
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
```
