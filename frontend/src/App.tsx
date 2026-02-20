import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import SlideList from './components/slides/SlideList';
import SlideDetail from './components/slides/SlideDetail';
import JobStatusList from './components/jobs/JobStatusList';
import BenchmarkRunner from './components/benchmarks/BenchmarkRunner';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/slides" replace />} />
          <Route path="slides" element={<SlideList />} />
          <Route path="slides/:slideId" element={<SlideDetail />} />
          <Route path="jobs" element={<JobStatusList />} />
          <Route path="benchmarks" element={<BenchmarkRunner />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
