import { useQuery, useQueryClient } from '@tanstack/react-query';
import { spatialApi } from '../api/client';
import type { BBoxQueryRequest, KNNQueryRequest } from '../types';

export function useBBoxQuery(request: BBoxQueryRequest | null) {
  return useQuery({
    queryKey: ['spatial-bbox', request],
    queryFn: () => spatialApi.bboxQuery(request!),
    enabled: !!request,
    staleTime: 60 * 1000, // 1 minute
  });
}

export function useKnnQuery(request: KNNQueryRequest | null) {
  return useQuery({
    queryKey: ['spatial-knn', request],
    queryFn: () => spatialApi.knnQuery(request!),
    enabled: !!request,
    staleTime: 60 * 1000,
  });
}

export function useDensityGrid(
  slideId: string | undefined,
  gridSize = 256,
  objectType?: string
) {
  return useQuery({
    queryKey: ['spatial-density', slideId, gridSize, objectType],
    queryFn: () => spatialApi.densityGrid(slideId!, gridSize, objectType),
    enabled: !!slideId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useCountInBbox(request: BBoxQueryRequest | null) {
  return useQuery({
    queryKey: ['spatial-count', request],
    queryFn: () => spatialApi.countInBbox(request!),
    enabled: !!request,
    staleTime: 30 * 1000,
  });
}

export function usePrefetchBBox(slideId: string) {
  const queryClient = useQueryClient();

  return (minX: number, minY: number, maxX: number, maxY: number) => {
    const request: BBoxQueryRequest = { slideId, minX, minY, maxX, maxY };
    queryClient.prefetchQuery({
      queryKey: ['spatial-bbox', request],
      queryFn: () => spatialApi.bboxQuery(request),
      staleTime: 60 * 1000,
    });
  };
}
