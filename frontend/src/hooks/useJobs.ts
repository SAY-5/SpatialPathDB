import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { jobsApi } from '../api/client';

export function useJobs(page = 0, size = 20, status?: string) {
  return useQuery({
    queryKey: ['jobs', page, size, status],
    queryFn: () => jobsApi.list(page, size, status),
    refetchInterval: 5000, // Poll for job updates
  });
}

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => jobsApi.get(jobId!),
    enabled: !!jobId,
    refetchInterval: (data) => {
      // Stop polling once job is complete
      if (data?.status === 'COMPLETED' || data?.status === 'FAILED' || data?.status === 'CANCELLED') {
        return false;
      }
      return 2000;
    },
  });
}

export function useJobCounts() {
  return useQuery({
    queryKey: ['job-counts'],
    queryFn: () => jobsApi.getCounts(),
    refetchInterval: 10000,
  });
}

export function useSubmitJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      slideId,
      jobType,
      parameters,
    }: {
      slideId: string;
      jobType: string;
      parameters?: Record<string, unknown>;
    }) => jobsApi.submit(slideId, jobType, parameters),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      queryClient.invalidateQueries({ queryKey: ['job-counts'] });
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => jobsApi.cancel(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      queryClient.invalidateQueries({ queryKey: ['job-counts'] });
    },
  });
}
