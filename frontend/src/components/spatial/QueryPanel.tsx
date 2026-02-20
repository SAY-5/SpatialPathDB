import { useState } from 'react';
import { useKnnQuery } from '../../hooks/useSpatialQuery';
import type { SlideBounds, SpatialObject } from '../../types';
import { Search, MapPin } from 'lucide-react';

interface QueryPanelProps {
  slideId: string;
  bounds?: SlideBounds;
}

export default function QueryPanel({ slideId, bounds }: QueryPanelProps) {
  const [queryType, setQueryType] = useState<'knn'>('knn');
  const [knnParams, setKnnParams] = useState({
    x: '',
    y: '',
    k: '10',
  });

  const knnRequest = knnParams.x && knnParams.y ? {
    slideId,
    x: parseFloat(knnParams.x),
    y: parseFloat(knnParams.y),
    k: parseInt(knnParams.k) || 10,
  } : null;

  const { data: knnResults, isLoading } = useKnnQuery(knnRequest);

  const setRandomPoint = () => {
    if (bounds) {
      const x = Math.random() * (bounds.maxX - bounds.minX) + bounds.minX;
      const y = Math.random() * (bounds.maxY - bounds.minY) + bounds.minY;
      setKnnParams({
        ...knnParams,
        x: x.toFixed(0),
        y: y.toFixed(0),
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="card">
        <h3 className="font-semibold text-dark-100 mb-4 flex items-center gap-2">
          <Search className="w-4 h-4" />
          KNN Query
        </h3>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="label">X</label>
              <input
                type="number"
                value={knnParams.x}
                onChange={(e) => setKnnParams({ ...knnParams, x: e.target.value })}
                className="input w-full"
                placeholder="X coordinate"
              />
            </div>
            <div>
              <label className="label">Y</label>
              <input
                type="number"
                value={knnParams.y}
                onChange={(e) => setKnnParams({ ...knnParams, y: e.target.value })}
                className="input w-full"
                placeholder="Y coordinate"
              />
            </div>
          </div>

          <div>
            <label className="label">K (neighbors)</label>
            <input
              type="number"
              value={knnParams.k}
              onChange={(e) => setKnnParams({ ...knnParams, k: e.target.value })}
              className="input w-full"
              min="1"
              max="100"
            />
          </div>

          <button
            onClick={setRandomPoint}
            className="btn-secondary w-full flex items-center justify-center gap-2"
          >
            <MapPin className="w-4 h-4" />
            Random Point
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="card flex items-center justify-center py-8">
          <div className="spinner w-6 h-6 text-accent-primary" />
        </div>
      )}

      {knnResults && knnResults.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-dark-100 mb-3">
            {knnResults.length} Nearest Neighbors
          </h3>
          <div className="space-y-2 max-h-80 overflow-auto">
            {knnResults.map((obj, idx) => (
              <div
                key={obj.id}
                className="flex items-center justify-between p-2 bg-dark-800 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="text-dark-500 text-sm w-6">{idx + 1}</span>
                  <div>
                    <p className="text-dark-200 text-sm font-medium">
                      {obj.label || 'Unknown'}
                    </p>
                    <p className="text-dark-500 text-xs">
                      ({obj.centroidX?.toFixed(0)}, {obj.centroidY?.toFixed(0)})
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-accent-secondary text-sm font-mono">
                    {obj.distance?.toFixed(1)} px
                  </p>
                  {obj.confidence && (
                    <p className="text-dark-500 text-xs">
                      {(obj.confidence * 100).toFixed(0)}%
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
