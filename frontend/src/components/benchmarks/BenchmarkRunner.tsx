import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { benchmarksApi } from '../../api/client';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';
import { BarChart3, Clock, Zap, TrendingUp } from 'lucide-react';

export default function BenchmarkRunner() {
  const [hoursBack, setHoursBack] = useState(24);

  const { data: results, isLoading } = useQuery({
    queryKey: ['benchmarks', hoursBack],
    queryFn: () => benchmarksApi.getResults(hoursBack),
    refetchInterval: 30000,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-100">Performance Benchmarks</h1>
          <p className="text-dark-400 mt-1">Query performance monitoring</p>
        </div>
        <select
          value={hoursBack}
          onChange={(e) => setHoursBack(parseInt(e.target.value))}
          className="input"
        >
          <option value={1}>Last hour</option>
          <option value={6}>Last 6 hours</option>
          <option value={24}>Last 24 hours</option>
          <option value={168}>Last 7 days</option>
        </select>
      </div>

      {isLoading ? (
        <div className="card flex items-center justify-center py-12">
          <div className="spinner w-8 h-8 text-accent-primary" />
        </div>
      ) : !results ? (
        <div className="card text-center py-12">
          <BarChart3 className="w-12 h-12 text-dark-600 mx-auto mb-4" />
          <p className="text-dark-400">No benchmark data available</p>
          <p className="text-dark-500 text-sm mt-1">
            Execute queries to generate performance data
          </p>
        </div>
      ) : (
        <>
          {results.summary && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="card">
                <div className="flex items-center gap-2 text-dark-400">
                  <Zap className="w-4 h-4" />
                  <span className="text-sm">Total Queries</span>
                </div>
                <p className="text-3xl font-bold text-dark-100 mt-1">
                  {results.summary.totalQueries.toLocaleString()}
                </p>
              </div>
              <div className="card">
                <div className="flex items-center gap-2 text-dark-400">
                  <Clock className="w-4 h-4" />
                  <span className="text-sm">Avg Latency</span>
                </div>
                <p className="text-3xl font-bold text-dark-100 mt-1">
                  {results.summary.overallAvgTimeMs.toFixed(1)} ms
                </p>
              </div>
              <div className="card">
                <div className="flex items-center gap-2 text-dark-400">
                  <TrendingUp className="w-4 h-4" />
                  <span className="text-sm">Query Types</span>
                </div>
                <p className="text-3xl font-bold text-dark-100 mt-1">
                  {results.statsByQueryType.length}
                </p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="font-semibold text-dark-100 mb-4">Latency by Query Type</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={results.statsByQueryType} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#4a4a5a" />
                    <XAxis type="number" stroke="#6e6e80" />
                    <YAxis
                      dataKey="queryType"
                      type="category"
                      stroke="#6e6e80"
                      width={60}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#2d2d3a',
                        border: '1px solid #4a4a5a',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="avgTimeMs" fill="#6366f1" name="Avg (ms)" />
                    <Bar dataKey="p95TimeMs" fill="#22d3ee" name="P95 (ms)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <h3 className="font-semibold text-dark-100 mb-4">Query Volume Over Time</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={results.hourlyStats}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4a4a5a" />
                    <XAxis
                      dataKey="hour"
                      stroke="#6e6e80"
                      tick={{ fontSize: 10 }}
                      tickFormatter={(val) => new Date(val).getHours() + ':00'}
                    />
                    <YAxis stroke="#6e6e80" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#2d2d3a',
                        border: '1px solid #4a4a5a',
                        borderRadius: '8px',
                      }}
                      labelFormatter={(val) => new Date(val).toLocaleString()}
                    />
                    <Line
                      type="monotone"
                      dataKey="queryCount"
                      stroke="#6366f1"
                      strokeWidth={2}
                      dot={false}
                      name="Queries"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card">
            <h3 className="font-semibold text-dark-100 mb-4">Query Type Statistics</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-dark-700">
                    <th className="text-left py-3 px-4 text-dark-400 font-medium">Query Type</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">Count</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">Avg (ms)</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">Min (ms)</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">Max (ms)</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">P95 (ms)</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">P99 (ms)</th>
                    <th className="text-right py-3 px-4 text-dark-400 font-medium">Avg Results</th>
                  </tr>
                </thead>
                <tbody>
                  {results.statsByQueryType.map((stat) => (
                    <tr key={stat.queryType} className="border-b border-dark-800 hover:bg-dark-800/50">
                      <td className="py-3 px-4 text-dark-200 font-medium">{stat.queryType}</td>
                      <td className="py-3 px-4 text-dark-200 text-right font-mono">
                        {stat.totalQueries.toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-accent-secondary text-right font-mono">
                        {stat.avgTimeMs.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-dark-400 text-right font-mono">
                        {stat.minTimeMs.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-dark-400 text-right font-mono">
                        {stat.maxTimeMs.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-dark-200 text-right font-mono">
                        {stat.p95TimeMs.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-dark-200 text-right font-mono">
                        {stat.p99TimeMs.toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-dark-400 text-right font-mono">
                        {stat.avgResults.toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
