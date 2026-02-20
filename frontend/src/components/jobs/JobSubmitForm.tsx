import { useState } from 'react';
import { useSubmitJob, useJobs } from '../../hooks/useJobs';
import { Zap, CheckCircle, Clock, Loader, XCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import clsx from 'clsx';

interface JobSubmitFormProps {
  slideId: string;
}

const jobTypes = [
  {
    id: 'cell_detection',
    name: 'Cell Detection',
    description: 'Detect and classify cells in the slide',
    params: [
      { key: 'target_cells', label: 'Target Cells', type: 'number', default: 100000 },
      { key: 'n_clusters', label: 'Number of Clusters', type: 'number', default: 15 },
      { key: 'confidence_threshold', label: 'Confidence Threshold', type: 'number', default: 0.5 },
    ],
  },
  {
    id: 'tissue_segmentation',
    name: 'Tissue Segmentation',
    description: 'Segment tissue regions and classify tissue types',
    params: [
      { key: 'n_regions', label: 'Number of Regions', type: 'number', default: 8 },
      { key: 'min_region_size', label: 'Min Region Size', type: 'number', default: 3000 },
      { key: 'max_region_size', label: 'Max Region Size', type: 'number', default: 20000 },
    ],
  },
  {
    id: 'spatial_statistics',
    name: 'Spatial Statistics',
    description: 'Compute spatial distribution statistics',
    params: [
      { key: 'hotspot_grid_size', label: 'Hotspot Grid Size', type: 'number', default: 500 },
      { key: 'colocalization_radius', label: 'Colocalization Radius', type: 'number', default: 100 },
      { key: 'compute_ripleys_k', label: "Compute Ripley's K", type: 'boolean', default: false },
    ],
  },
];

export default function JobSubmitForm({ slideId }: JobSubmitFormProps) {
  const [selectedType, setSelectedType] = useState(jobTypes[0]);
  const [params, setParams] = useState<Record<string, unknown>>({});

  const submitMutation = useSubmitJob();
  const { data: recentJobs } = useJobs(0, 5);

  const slideJobs = recentJobs?.content.filter((j) => j.slideId === slideId) || [];

  const handleSubmit = () => {
    const jobParams = { ...params };
    selectedType.params.forEach((p) => {
      if (jobParams[p.key] === undefined) {
        jobParams[p.key] = p.default;
      }
    });

    submitMutation.mutate({
      slideId,
      jobType: selectedType.id,
      parameters: jobParams,
    });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="space-y-6">
        <div className="card">
          <h3 className="font-semibold text-dark-100 mb-4">Submit Analysis Job</h3>

          <div className="space-y-4">
            <div>
              <label className="label">Job Type</label>
              <div className="space-y-2">
                {jobTypes.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => {
                      setSelectedType(type);
                      setParams({});
                    }}
                    className={clsx(
                      'w-full text-left p-3 rounded-lg border transition-colors',
                      selectedType.id === type.id
                        ? 'border-accent-primary bg-accent-primary/10'
                        : 'border-dark-700 hover:border-dark-600'
                    )}
                  >
                    <p className="font-medium text-dark-100">{type.name}</p>
                    <p className="text-sm text-dark-400 mt-1">{type.description}</p>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="label">Parameters</label>
              <div className="space-y-3">
                {selectedType.params.map((param) => (
                  <div key={param.key}>
                    <label className="text-xs text-dark-400 mb-1 block">
                      {param.label}
                    </label>
                    {param.type === 'boolean' ? (
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={params[param.key] as boolean ?? param.default}
                          onChange={(e) =>
                            setParams({ ...params, [param.key]: e.target.checked })
                          }
                          className="rounded border-dark-600 bg-dark-800 text-accent-primary focus:ring-accent-primary"
                        />
                        <span className="text-dark-300 text-sm">Enabled</span>
                      </label>
                    ) : (
                      <input
                        type="number"
                        value={params[param.key] as number ?? param.default}
                        onChange={(e) =>
                          setParams({ ...params, [param.key]: parseFloat(e.target.value) })
                        }
                        className="input w-full"
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={submitMutation.isPending}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {submitMutation.isPending ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Submit Job
                </>
              )}
            </button>

            {submitMutation.isSuccess && (
              <div className="p-3 bg-accent-success/10 border border-accent-success/20 rounded-lg">
                <p className="text-accent-success text-sm flex items-center gap-2">
                  <CheckCircle className="w-4 h-4" />
                  Job submitted successfully
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold text-dark-100 mb-4">Recent Jobs for This Slide</h3>

        {slideJobs.length === 0 ? (
          <p className="text-dark-500 text-center py-8">No jobs yet</p>
        ) : (
          <div className="space-y-3">
            {slideJobs.map((job) => (
              <div
                key={job.id}
                className="p-3 bg-dark-800 rounded-lg"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {job.status === 'COMPLETED' && (
                      <CheckCircle className="w-4 h-4 text-accent-success" />
                    )}
                    {job.status === 'RUNNING' && (
                      <Loader className="w-4 h-4 text-accent-warning animate-spin" />
                    )}
                    {job.status === 'QUEUED' && (
                      <Clock className="w-4 h-4 text-dark-400" />
                    )}
                    {job.status === 'FAILED' && (
                      <XCircle className="w-4 h-4 text-accent-danger" />
                    )}
                    <span className="text-dark-200 font-medium">{job.jobType}</span>
                  </div>
                  <span className="text-xs text-dark-500">
                    {formatDistanceToNow(new Date(job.submittedAt), { addSuffix: true })}
                  </span>
                </div>
                {job.errorMessage && (
                  <p className="text-sm text-accent-danger mt-2">{job.errorMessage}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
