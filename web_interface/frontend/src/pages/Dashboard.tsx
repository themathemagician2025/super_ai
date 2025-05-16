import React from 'react';

const Dashboard: React.FC = () => {
    return (
        <div>
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
                <p className="text-gray-600">Overview of your prediction metrics and history</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Summary Cards */}
                <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-lg font-semibold text-gray-700 mb-2">Total Predictions</h2>
                    <p className="text-3xl font-bold text-indigo-600">124</p>
                    <p className="text-sm text-gray-500 mt-2">+12% from last month</p>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-lg font-semibold text-gray-700 mb-2">Average Accuracy</h2>
                    <p className="text-3xl font-bold text-indigo-600">87.2%</p>
                    <p className="text-sm text-gray-500 mt-2">+2.5% from last month</p>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-lg font-semibold text-gray-700 mb-2">Active Models</h2>
                    <p className="text-3xl font-bold text-indigo-600">5</p>
                    <p className="text-sm text-gray-500 mt-2">1 new model added</p>
                </div>
            </div>

            {/* Recent Predictions */}
            <div className="bg-white rounded-lg shadow mb-8">
                <div className="p-6 border-b">
                    <h2 className="text-xl font-semibold text-gray-800">Recent Predictions</h2>
                </div>
                <div className="p-6">
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead>
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Input</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Prediction</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {[1, 2, 3, 4, 5].map((item) => (
                                    <tr key={item} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2023-06-{10 + item}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Financial Model {item}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Dataset #{item}20</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Bullish</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                                                {85 + item}%
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Performance Chart Placeholder */}
            <div className="bg-white rounded-lg shadow">
                <div className="p-6 border-b">
                    <h2 className="text-xl font-semibold text-gray-800">Performance Over Time</h2>
                </div>
                <div className="p-6">
                    <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
                        <p className="text-gray-500">Chart visualization will be displayed here</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
