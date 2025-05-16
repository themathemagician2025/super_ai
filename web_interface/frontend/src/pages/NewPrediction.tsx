import React, { useState } from 'react';

const NewPrediction: React.FC = () => {
    const [formData, setFormData] = useState({
        modelType: 'financial',
        dataSource: 'historical',
        parameters: '',
        description: ''
    });

    const [isLoading, setIsLoading] = useState(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        // Simulate API call
        setTimeout(() => {
            setIsLoading(false);
            alert('Prediction request submitted successfully!');
        }, 1500);
    };

    return (
        <div>
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">New Prediction</h1>
                <p className="text-gray-600">Configure and run a new prediction model</p>
            </header>

            <div className="bg-white rounded-lg shadow p-6">
                <form onSubmit={handleSubmit}>
                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="modelType">
                            Model Type
                        </label>
                        <select
                            id="modelType"
                            name="modelType"
                            value={formData.modelType}
                            onChange={handleChange}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        >
                            <option value="financial">Financial Prediction</option>
                            <option value="market">Market Trend Analysis</option>
                            <option value="risk">Risk Assessment</option>
                            <option value="custom">Custom Model</option>
                        </select>
                    </div>

                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="dataSource">
                            Data Source
                        </label>
                        <select
                            id="dataSource"
                            name="dataSource"
                            value={formData.dataSource}
                            onChange={handleChange}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        >
                            <option value="historical">Historical Data</option>
                            <option value="realtime">Real-time Data Feed</option>
                            <option value="custom">Custom Data Source</option>
                            <option value="upload">Upload Data File</option>
                        </select>
                    </div>

                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="parameters">
                            Parameters (JSON format)
                        </label>
                        <textarea
                            id="parameters"
                            name="parameters"
                            value={formData.parameters}
                            onChange={handleChange}
                            placeholder='{"timeframe": "1M", "indicators": ["MA", "RSI"], "confidence": 0.8}'
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline h-32"
                        />
                    </div>

                    <div className="mb-6">
                        <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="description">
                            Description
                        </label>
                        <input
                            id="description"
                            name="description"
                            type="text"
                            value={formData.description}
                            onChange={handleChange}
                            placeholder="Brief description of this prediction run"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline disabled:opacity-50"
                        >
                            {isLoading ? 'Processing...' : 'Run Prediction'}
                        </button>
                        <button
                            type="button"
                            className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        >
                            Save as Template
                        </button>
                    </div>
                </form>
            </div>

            <div className="mt-8 bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Frequently Used Configurations</h2>
                <div className="space-y-4">
                    {[1, 2, 3].map((item) => (
                        <div key={item} className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer">
                            <h3 className="font-medium text-gray-900">Financial Forecast Template {item}</h3>
                            <p className="text-sm text-gray-500 mt-1">Last used 2 days ago</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default NewPrediction;
