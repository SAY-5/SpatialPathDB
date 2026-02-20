export interface Slide {
  id: string;
  slideName: string;
  filePath: string;
  widthPixels: number;
  heightPixels: number;
  micronsPerPixel?: number;
  stainType?: string;
  organ?: string;
  uploadedAt: string;
  metadata?: Record<string, unknown>;
  objectCount?: number;
}

export interface SpatialObject {
  id: number;
  objectType: string;
  label?: string;
  confidence?: number;
  geometryWkt?: string;
  centroidX?: number;
  centroidY?: number;
  areaPixels?: number;
  perimeterPixels?: number;
  properties?: Record<string, unknown>;
  distance?: number;
}

export interface DensityGrid {
  gridX: number;
  gridY: number;
  cellCount: number;
  avgConfidence?: number;
  dominantLabel?: string;
  centerX: number;
  centerY: number;
}

export interface SlideStatistics {
  slideId: string;
  totalObjects: number;
  byType: TypeStatistics[];
}

export interface TypeStatistics {
  objectType: string;
  label?: string;
  totalCount: number;
  avgArea?: number;
  stdArea?: number;
  avgConfidence?: number;
  spatialExtent?: string;
}

export interface AnalysisJob {
  id: string;
  slideId: string;
  jobType: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  parameters?: Record<string, unknown>;
  resultSummary?: Record<string, unknown>;
  submittedAt: string;
  startedAt?: string;
  completedAt?: string;
  errorMessage?: string;
  durationMs?: number;
}

export interface BBoxQueryRequest {
  slideId: string;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  objectType?: string;
  minConfidence?: number;
  limit?: number;
  offset?: number;
}

export interface KNNQueryRequest {
  slideId: string;
  x: number;
  y: number;
  k?: number;
  objectType?: string;
}

export interface SlideBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  totalObjects: number;
}

export interface BenchmarkResults {
  period: {
    since: string;
    until: string;
  };
  statsByQueryType: QueryTypeStats[];
  hourlyStats: HourlyStats[];
  summary?: {
    totalQueries: number;
    overallAvgTimeMs: number;
  };
}

export interface QueryTypeStats {
  queryType: string;
  totalQueries: number;
  avgTimeMs: number;
  minTimeMs: number;
  maxTimeMs: number;
  p95TimeMs: number;
  p99TimeMs: number;
  avgResults: number;
}

export interface HourlyStats {
  hour: string;
  queryCount: number;
  avgTimeMs: number;
}

export interface Page<T> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
}

export interface HealthStatus {
  status: 'UP' | 'DOWN' | 'DEGRADED';
  timestamp: string;
  database: {
    status: string;
    database?: string;
    version?: string;
    error?: string;
  };
  redis: {
    status: string;
    ping?: string;
    error?: string;
  };
}
