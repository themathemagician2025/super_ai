import React from 'react';
import { Outlet, Link } from 'react-router-dom';

const Layout: React.FC = () => {
    return (
        <div className="flex h-screen bg-gray-100">
            {/* Sidebar Navigation */}
            <aside className="w-64 bg-white shadow-md">
                <div className="p-4 border-b">
                    <h1 className="text-xl font-semibold text-indigo-600">Super AI</h1>
                    <p className="text-sm text-gray-500">Prediction Platform</p>
                </div>
                <nav className="p-4">
                    <ul className="space-y-2">
                        <li>
                            <Link to="/" className="block p-2 rounded hover:bg-indigo-50 hover:text-indigo-600">
                                Dashboard
                            </Link>
                        </li>
                        <li>
                            <Link to="/predict" className="block p-2 rounded hover:bg-indigo-50 hover:text-indigo-600">
                                New Prediction
                            </Link>
                        </li>
                        <li>
                            <Link to="/results/latest" className="block p-2 rounded hover:bg-indigo-50 hover:text-indigo-600">
                                Results
                            </Link>
                        </li>
                        <li>
                            <Link to="/visualizations" className="block p-2 rounded hover:bg-indigo-50 hover:text-indigo-600">
                                Visualizations
                            </Link>
                        </li>
                        <li>
                            <Link to="/model-comparison" className="block p-2 rounded hover:bg-indigo-50 hover:text-indigo-600">
                                Model Comparison
                            </Link>
                        </li>
                        <li>
                            <Link to="/settings" className="block p-2 rounded hover:bg-indigo-50 hover:text-indigo-600">
                                Settings
                            </Link>
                        </li>
                    </ul>
                </nav>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-auto p-8">
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
