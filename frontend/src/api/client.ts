import axios from 'axios';
import type {
  Slide,
  SpatialObject,
  DensityGrid,
  SlideStatistics,
  AnalysisJob,
  BBoxQueryRequest,
  KNNQueryRequest,
  SlideBounds,
  BenchmarkResults,
  Page,
  HealthStatus,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Slides API
export const slidesApi = {
  list: async (page = 0, size = 20): Promise<Page<Slide>> => {
    const { data } = await api.get('/slides', { params: { page, size } });
    return data;
  },

  get: async (slideId: string): Promise<Slide> => {
    const { data } = await api.get(`/slides/${slideId}`);
    return data;
  },

  create: async (slide: Partial<Slide>): Promise<Slide> => {
    const { data } = await api.post('/slides', slide);
    return data;
  },

  delete: async (slideId: string): Promise<void> => {
    await api.delete(`/slides/${slideId}`);
  },

  getStatistics: async (slideId: string): Promise<SlideStatistics> => {
    const { data } = await api.get(`/slides/${slideId}/statistics`);
    return data;
  },

  getBounds: async (slideId: string): Promise<SlideBounds> => {
    const { data } = await api.get(`/slides/${slideId}/bounds`);
    return data;
  },
};

// Spatial Query API
export const spatialApi = {
  bboxQuery: async (request: BBoxQueryRequest): Promise<SpatialObject[]> => {
    const { data } = await api.post('/spatial/bbox', request);
    return data;
  },

  knnQuery: async (request: KNNQueryRequest): Promise<SpatialObject[]> => {
    const { data } = await api.post('/spatial/knn', request);
    return data;
  },

  densityGrid: async (
    slideId: string,
    gridSize = 256,
    objectType?: string
  ): Promise<DensityGrid[]> => {
    const { data } = await api.post('/spatial/density', null, {
      params: { slideId, gridSize, objectType },
    });
    return data;
  },

  countInBbox: async (request: BBoxQueryRequest): Promise<{ count: number }> => {
    const { data } = await api.post('/spatial/count', request);
    return data;
  },
};

// Jobs API
export const jobsApi = {
  list: async (page = 0, size = 20, status?: string): Promise<Page<AnalysisJob>> => {
    const { data } = await api.get('/jobs', { params: { page, size, status } });
    return data;
  },

  get: async (jobId: string): Promise<AnalysisJob> => {
    const { data } = await api.get(`/jobs/${jobId}`);
    return data;
  },

  submit: async (
    slideId: string,
    jobType: string,
    parameters?: Record<string, unknown>
  ): Promise<AnalysisJob> => {
    const { data } = await api.post('/jobs', { slideId, jobType, parameters });
    return data;
  },

  cancel: async (jobId: string): Promise<void> => {
    await api.delete(`/jobs/${jobId}`);
  },

  getCounts: async (): Promise<Record<string, number>> => {
    const { data } = await api.get('/jobs/counts');
    return data;
  },
};

// Benchmarks API
export const benchmarksApi = {
  getResults: async (hoursBack = 24): Promise<BenchmarkResults> => {
    const { data } = await api.get('/benchmarks/results', { params: { hoursBack } });
    return data;
  },
};

// Health API
export const healthApi = {
  check: async (): Promise<HealthStatus> => {
    const { data } = await api.get('/health');
    return data;
  },
};

export default api;
