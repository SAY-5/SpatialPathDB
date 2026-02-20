import { useEffect, useState, useCallback, useMemo } from 'react';
import { MapContainer, useMap, useMapEvents, Rectangle, CircleMarker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { useBBoxQuery, useDensityGrid } from '../../hooks/useSpatialQuery';
import type { SlideBounds, SpatialObject, DensityGrid } from '../../types';
import { Layers, Grid3X3 } from 'lucide-react';
import clsx from 'clsx';

interface SpatialViewerProps {
  slideId: string;
  bounds?: SlideBounds;
  slideWidth: number;
  slideHeight: number;
}

const CELL_COLORS: Record<string, string> = {
  epithelial: '#FF6B6B',
  stromal: '#4ECDC4',
  lymphocyte: '#45B7D1',
  macrophage: '#96CEB4',
  necrotic: '#FFEAA7',
  default: '#8e8ea0',
};

function getCellColor(label?: string): string {
  return label ? CELL_COLORS[label.toLowerCase()] || CELL_COLORS.default : CELL_COLORS.default;
}

function ViewportTracker({
  onViewportChange,
}: {
  onViewportChange: (bounds: L.LatLngBounds) => void;
}) {
  const map = useMapEvents({
    moveend: () => {
      onViewportChange(map.getBounds());
    },
    zoomend: () => {
      onViewportChange(map.getBounds());
    },
  });

  useEffect(() => {
    onViewportChange(map.getBounds());
  }, [map, onViewportChange]);

  return null;
}

function MapController({ bounds, slideWidth, slideHeight }: {
  bounds?: SlideBounds;
  slideWidth: number;
  slideHeight: number;
}) {
  const map = useMap();

  useEffect(() => {
    if (bounds && bounds.maxX > 0) {
      const latLngBounds = L.latLngBounds(
        [bounds.minY, bounds.minX],
        [bounds.maxY, bounds.maxX]
      );
      map.fitBounds(latLngBounds, { padding: [20, 20] });
    } else {
      map.fitBounds([
        [0, 0],
        [slideHeight, slideWidth],
      ]);
    }
  }, [map, bounds, slideWidth, slideHeight]);

  return null;
}

function CellMarkers({ objects, selectedId, onSelect }: {
  objects: SpatialObject[];
  selectedId?: number;
  onSelect: (obj: SpatialObject) => void;
}) {
  return (
    <>
      {objects.map((obj) => (
        <CircleMarker
          key={obj.id}
          center={[obj.centroidY || 0, obj.centroidX || 0]}
          radius={6}
          pathOptions={{
            color: selectedId === obj.id ? '#fff' : getCellColor(obj.label),
            fillColor: getCellColor(obj.label),
            fillOpacity: 0.7,
            weight: selectedId === obj.id ? 2 : 1,
          }}
          eventHandlers={{
            click: () => onSelect(obj),
          }}
        >
          <Popup>
            <div className="text-sm">
              <p className="font-semibold">{obj.label || 'Unknown'}</p>
              <p>Type: {obj.objectType}</p>
              {obj.confidence && (
                <p>Confidence: {(obj.confidence * 100).toFixed(1)}%</p>
              )}
              {obj.areaPixels && (
                <p>Area: {obj.areaPixels.toFixed(0)} px²</p>
              )}
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </>
  );
}

function DensityHeatmap({ density, gridSize }: {
  density: DensityGrid[];
  gridSize: number;
}) {
  const maxCount = Math.max(...density.map((d) => d.cellCount));

  return (
    <>
      {density.map((cell, idx) => {
        const intensity = cell.cellCount / maxCount;
        const color = `rgba(99, 102, 241, ${intensity * 0.7})`;

        return (
          <Rectangle
            key={idx}
            bounds={[
              [cell.gridY * gridSize, cell.gridX * gridSize],
              [(cell.gridY + 1) * gridSize, (cell.gridX + 1) * gridSize],
            ]}
            pathOptions={{
              color: 'transparent',
              fillColor: color,
              fillOpacity: 1,
            }}
          >
            <Popup>
              <div className="text-sm">
                <p className="font-semibold">Grid Cell</p>
                <p>Count: {cell.cellCount.toLocaleString()}</p>
                {cell.dominantLabel && (
                  <p>Dominant: {cell.dominantLabel}</p>
                )}
              </div>
            </Popup>
          </Rectangle>
        );
      })}
    </>
  );
}

export default function SpatialViewer({
  slideId,
  bounds,
  slideWidth,
  slideHeight,
}: SpatialViewerProps) {
  const [viewMode, setViewMode] = useState<'cells' | 'density'>('cells');
  const [viewportBounds, setViewportBounds] = useState<L.LatLngBounds | null>(null);
  const [selectedObject, setSelectedObject] = useState<SpatialObject | null>(null);
  const [gridSize] = useState(512);

  const bboxRequest = useMemo(() => {
    if (!viewportBounds) return null;
    return {
      slideId,
      minX: viewportBounds.getWest(),
      minY: viewportBounds.getSouth(),
      maxX: viewportBounds.getEast(),
      maxY: viewportBounds.getNorth(),
      limit: 2000,
    };
  }, [slideId, viewportBounds]);

  const { data: objects = [], isLoading: objectsLoading } = useBBoxQuery(
    viewMode === 'cells' ? bboxRequest : null
  );

  const { data: density = [] } = useDensityGrid(
    viewMode === 'density' ? slideId : undefined,
    gridSize
  );

  const handleViewportChange = useCallback((newBounds: L.LatLngBounds) => {
    setViewportBounds(newBounds);
  }, []);

  // Custom CRS for pixel coordinates
  const crs = useMemo(() => {
    return L.CRS.Simple;
  }, []);

  return (
    <div className="relative h-full">
      <div className="absolute top-4 right-4 z-[1000] flex gap-2">
        <button
          onClick={() => setViewMode('cells')}
          className={clsx(
            'flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
            viewMode === 'cells'
              ? 'bg-accent-primary text-white'
              : 'bg-dark-800 text-dark-300 hover:bg-dark-700'
          )}
        >
          <Layers className="w-4 h-4" />
          Cells
        </button>
        <button
          onClick={() => setViewMode('density')}
          className={clsx(
            'flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
            viewMode === 'density'
              ? 'bg-accent-primary text-white'
              : 'bg-dark-800 text-dark-300 hover:bg-dark-700'
          )}
        >
          <Grid3X3 className="w-4 h-4" />
          Density
        </button>
      </div>

      {objectsLoading && (
        <div className="absolute top-4 left-4 z-[1000] bg-dark-800 px-3 py-2 rounded-lg flex items-center gap-2">
          <div className="spinner w-4 h-4 text-accent-primary" />
          <span className="text-sm text-dark-300">Loading...</span>
        </div>
      )}

      {viewMode === 'cells' && objects.length > 0 && (
        <div className="absolute bottom-4 left-4 z-[1000] bg-dark-800 px-3 py-2 rounded-lg">
          <span className="text-sm text-dark-300">
            {objects.length.toLocaleString()} objects in view
          </span>
        </div>
      )}

      <MapContainer
        crs={crs}
        center={[slideHeight / 2, slideWidth / 2]}
        zoom={-2}
        minZoom={-5}
        maxZoom={2}
        style={{ height: '100%', width: '100%', background: '#1a1a24' }}
        zoomControl={true}
      >
        <MapController bounds={bounds} slideWidth={slideWidth} slideHeight={slideHeight} />
        <ViewportTracker onViewportChange={handleViewportChange} />

        {viewMode === 'cells' && (
          <CellMarkers
            objects={objects}
            selectedId={selectedObject?.id}
            onSelect={setSelectedObject}
          />
        )}

        {viewMode === 'density' && (
          <DensityHeatmap density={density} gridSize={gridSize} />
        )}
      </MapContainer>

      {/* Cell type legend */}
      <div className="absolute bottom-4 right-4 z-[1000] bg-dark-800 p-3 rounded-lg">
        <p className="text-xs text-dark-400 mb-2">Cell Types</p>
        <div className="space-y-1">
          {Object.entries(CELL_COLORS).filter(([k]) => k !== 'default').map(([label, color]) => (
            <div key={label} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-dark-300 capitalize">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
