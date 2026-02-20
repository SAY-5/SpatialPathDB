import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { slidesApi } from '../api/client';
import type { Slide } from '../types';

export function useSlides(page = 0, size = 20) {
  return useQuery({
    queryKey: ['slides', page, size],
    queryFn: () => slidesApi.list(page, size),
  });
}

export function useSlide(slideId: string | undefined) {
  return useQuery({
    queryKey: ['slide', slideId],
    queryFn: () => slidesApi.get(slideId!),
    enabled: !!slideId,
  });
}

export function useSlideStatistics(slideId: string | undefined) {
  return useQuery({
    queryKey: ['slide-statistics', slideId],
    queryFn: () => slidesApi.getStatistics(slideId!),
    enabled: !!slideId,
  });
}

export function useSlideBounds(slideId: string | undefined) {
  return useQuery({
    queryKey: ['slide-bounds', slideId],
    queryFn: () => slidesApi.getBounds(slideId!),
    enabled: !!slideId,
  });
}

export function useCreateSlide() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (slide: Partial<Slide>) => slidesApi.create(slide),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['slides'] });
    },
  });
}

export function useDeleteSlide() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (slideId: string) => slidesApi.delete(slideId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['slides'] });
    },
  });
}
