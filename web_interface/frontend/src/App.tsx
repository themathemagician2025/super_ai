import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import NewPrediction from './pages/NewPrediction';
import Results from './pages/Results';
import Visualizations from './pages/Visualizations';
import ModelComparison from './pages/ModelComparison';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';

function App() {
    return (
        <Routes>
            <Route element={<Layout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/predict" element={<NewPrediction />} />
                <Route path="/results/:id" element={<Results />} />
                <Route path="/visualizations" element={<Visualizations />} />
                <Route path="/model-comparison" element={<ModelComparison />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
    );
}

export default App;
