import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { SlideStatistics } from '../../types';

interface StatisticsPanelProps {
  slideId: string;
  statistics?: SlideStatistics;
}

const COLORS = ['#6366f1', '#22d3ee', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'];

export default function StatisticsPanel({ slideId, statistics }: StatisticsPanelProps) {
  if (!statistics) {
    return (
      <div className="card flex items-center justify-center py-12">
        <div className="spinner w-8 h-8 text-accent-primary" />
      </div>
    );
  }

  const cellTypeCounts = statistics.byType
    .filter((t) => t.objectType === 'cell')
    .map((t) => ({
      name: t.label || 'Unknown',
      value: t.totalCount,
      avgArea: t.avgArea,
      avgConfidence: t.avgConfidence,
    }));

  const areaByType = statistics.byType
    .filter((t) => t.objectType === 'cell' && t.avgArea)
    .map((t) => ({
      name: t.label || 'Unknown',
      avgArea: Math.round(t.avgArea || 0),
      stdArea: Math.round(t.stdArea || 0),
    }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <p className="text-dark-400 text-sm">Total Objects</p>
          <p className="text-3xl font-bold text-dark-100 mt-1">
            {statistics.totalObjects.toLocaleString()}
          </p>
        </div>
        <div className="card">
          <p className="text-dark-400 text-sm">Cell Types</p>
          <p className="text-3xl font-bold text-dark-100 mt-1">
            {cellTypeCounts.length}
          </p>
        </div>
        <div className="card">
          <p className="text-dark-400 text-sm">Avg Confidence</p>
          <p className="text-3xl font-bold text-dark-100 mt-1">
            {(
              cellTypeCounts.reduce((sum, c) => sum + (c.avgConfidence || 0), 0) /
              cellTypeCounts.length * 100
            ).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-dark-100 mb-4">Cell Type Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={cellTypeCounts}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} (${(percent * 100).toFixed(0)}%)`
                  }
                  labelLine={false}
                >
                  {cellTypeCounts.map((_, idx) => (
                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d3a',
                    border: '1px solid #4a4a5a',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number) => value.toLocaleString()}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3 className="font-semibold text-dark-100 mb-4">Cell Area by Type</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={areaByType} layout="vertical">
                <XAxis type="number" stroke="#6e6e80" />
                <YAxis
                  dataKey="name"
                  type="category"
                  stroke="#6e6e80"
                  width={80}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d3a',
                    border: '1px solid #4a4a5a',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="avgArea" fill="#6366f1" name="Avg Area (px²)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold text-dark-100 mb-4">Detailed Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dark-700">
                <th className="text-left py-3 px-4 text-dark-400 font-medium">Type</th>
                <th className="text-left py-3 px-4 text-dark-400 font-medium">Label</th>
                <th className="text-right py-3 px-4 text-dark-400 font-medium">Count</th>
                <th className="text-right py-3 px-4 text-dark-400 font-medium">Avg Area</th>
                <th className="text-right py-3 px-4 text-dark-400 font-medium">Std Area</th>
                <th className="text-right py-3 px-4 text-dark-400 font-medium">Avg Conf</th>
              </tr>
            </thead>
            <tbody>
              {statistics.byType.map((row, idx) => (
                <tr key={idx} className="border-b border-dark-800 hover:bg-dark-800/50">
                  <td className="py-3 px-4 text-dark-200">{row.objectType}</td>
                  <td className="py-3 px-4 text-dark-200">{row.label || '—'}</td>
                  <td className="py-3 px-4 text-dark-200 text-right font-mono">
                    {row.totalCount.toLocaleString()}
                  </td>
                  <td className="py-3 px-4 text-dark-200 text-right font-mono">
                    {row.avgArea?.toFixed(1) || '—'}
                  </td>
                  <td className="py-3 px-4 text-dark-200 text-right font-mono">
                    {row.stdArea?.toFixed(1) || '—'}
                  </td>
                  <td className="py-3 px-4 text-dark-200 text-right font-mono">
                    {row.avgConfidence ? `${(row.avgConfidence * 100).toFixed(1)}%` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
