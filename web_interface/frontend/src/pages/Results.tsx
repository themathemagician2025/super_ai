import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';

const Results: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const [activeTab, setActiveTab] = useState<'overview' | 'details' | 'visualization'>('overview');

    // Mock prediction data
    const predictionData = {
        id: id || '12345',
        model: 'Financial Forecast Model',
        createdAt: '2023-06-15T10:30:00Z',
        status: 'completed',
        accuracy: 92.7,
        executionTime: '1m 23s',
        parameters: {
            timeframe: '3M',
            indicators: ['MA', 'RSI', 'MACD'],
            confidence: 0.85
        },
        prediction: 'Bullish trend with 92.7% confidence',
        actualOutcome: 'Bullish',
    };

    // Mock SHAP values for feature importance
    const featureImportanceData = {
        categories: ['Interest Rate', 'Market Volume', 'Previous Close', 'EMA', 'RSI', 'MACD', 'Volatility', 'Sector Performance'],
        values: [0.32, 0.27, 0.18, 0.12, 0.05, 0.03, 0.02, 0.01]
    };

    // Mock data for prediction over time
    const timeSeriesOption = {
        title: {
            text: 'Prediction vs Actual',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ['Predicted', 'Actual'],
            bottom: 0
        },
        xAxis: {
            type: 'category',
            data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        },
        yAxis: {
            type: 'value',
            name: 'Value'
        },
        series: [
            {
                name: 'Predicted',
                type: 'line',
                data: [820, 932, 901, 934, 1290, 1330],
                lineStyle: {
                    color: '#5470c6'
                }
            },
            {
                name: 'Actual',
                type: 'line',
                data: [820, 932, 941, 934, 1230, 1320],
                lineStyle: {
                    color: '#91cc75'
                }
            }
        ]
    };

    // Mock SHAP visualization option
    const shapOption = {
        title: {
            text: 'Feature Importance (SHAP Values)',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '8%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            name: 'SHAP Value (impact on model output)',
            nameLocation: 'middle',
            nameGap: 30
        },
        yAxis: {
            type: 'category',
            data: featureImportanceData.categories
        },
        series: [
            {
                name: 'SHAP Value',
                type: 'bar',
                data: featureImportanceData.values,
                itemStyle: {
                    color: function (params: any) {
                        const value = params.value;
                        return value > 0.2 ? '#5470c6' :
                            value > 0.1 ? '#91cc75' :
                                '#fac858';
                    }
                }
            }
        ]
    };

    // Mock LIME visualization option
    const limeOption = {
        title: {
            text: 'Local Feature Importance (LIME)',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '8%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            name: 'Feature Contribution',
            nameLocation: 'middle',
            nameGap: 30
        },
        yAxis: {
            type: 'category',
            data: featureImportanceData.categories.slice().reverse()
        },
        series: [
            {
                name: 'Positive Impact',
                type: 'bar',
                stack: 'total',
                data: [0.25, 0.18, 0.12, 0.05, 0, 0, 0, 0],
                itemStyle: {
                    color: '#91cc75'
                }
            },
            {
                name: 'Negative Impact',
                type: 'bar',
                stack: 'total',
                data: [0, 0, 0, 0, -0.03, -0.04, -0.08, -0.15],
                itemStyle: {
                    color: '#ee6666'
                }
            }
        ]
    };

    return (
        <div>
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">Prediction Results</h1>
                <p className="text-gray-600">Detailed analysis and visualizations of prediction #{predictionData.id}</p>
            </header>

            {/* Status Card */}
            <div className="bg-white rounded-lg shadow mb-6 p-6">
                <div className="flex justify-between items-center">
                    <div>
                        <h2 className="text-xl font-semibold text-gray-800">{predictionData.model}</h2>
                        <p className="text-gray-500">Run on {new Date(predictionData.createdAt).toLocaleString()}</p>
                    </div>
                    <div className="flex items-center">
                        <span className="px-3 py-1 rounded-full bg-green-100 text-green-800 text-sm font-medium mr-4">
                            {predictionData.status}
                        </span>
                        <span className="text-xl font-bold text-indigo-600">{predictionData.accuracy}%</span>
                    </div>
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="border-b border-gray-200 mb-6">
                <nav className="flex -mb-px">
                    <button
                        onClick={() => setActiveTab('overview')}
                        className={`mr-8 py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'overview'
                                ? 'border-indigo-500 text-indigo-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                    >
                        Overview
                    </button>
                    <button
                        onClick={() => setActiveTab('details')}
                        className={`mr-8 py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'details'
                                ? 'border-indigo-500 text-indigo-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                    >
                        Detailed Results
                    </button>
                    <button
                        onClick={() => setActiveTab('visualization')}
                        className={`mr-8 py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'visualization'
                                ? 'border-indigo-500 text-indigo-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                    >
                        Visualizations
                    </button>
                </nav>
            </div>

            {/* Tab Content */}
            <div className="bg-white rounded-lg shadow">
                {activeTab === 'overview' && (
                    <div className="p-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">Prediction Summary</h3>
                                <div className="space-y-4">
                                    <div>
                                        <p className="text-sm text-gray-500">Prediction</p>
                                        <p className="font-medium">{predictionData.prediction}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-500">Actual Outcome</p>
                                        <p className="font-medium">{predictionData.actualOutcome}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-500">Execution Time</p>
                                        <p className="font-medium">{predictionData.executionTime}</p>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800 mb-4">Model Parameters</h3>
                                <div className="space-y-4">
                                    <div>
                                        <p className="text-sm text-gray-500">Timeframe</p>
                                        <p className="font-medium">{predictionData.parameters.timeframe}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-500">Indicators</p>
                                        <p className="font-medium">{predictionData.parameters.indicators.join(', ')}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-500">Confidence Threshold</p>
                                        <p className="font-medium">{predictionData.parameters.confidence}</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="mt-8">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Feature Importance</h3>
                            <div className="h-80">
                                <ReactECharts option={shapOption} style={{ height: '100%' }} />
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'details' && (
                    <div className="p-6">
                        <div className="mb-8">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Detailed Results</h3>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200">
                                    <thead>
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Benchmark</th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Difference</th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200">
                                        {[
                                            { metric: 'Accuracy', value: '92.7%', benchmark: '90.0%', difference: '+2.7%' },
                                            { metric: 'Precision', value: '91.5%', benchmark: '89.2%', difference: '+2.3%' },
                                            { metric: 'Recall', value: '94.2%', benchmark: '92.1%', difference: '+2.1%' },
                                            { metric: 'F1 Score', value: '92.8%', benchmark: '90.6%', difference: '+2.2%' },
                                            { metric: 'ROC AUC', value: '0.956', benchmark: '0.942', difference: '+0.014' }
                                        ].map((item, index) => (
                                            <tr key={index} className="hover:bg-gray-50">
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.metric}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.value}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.benchmark}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${item.difference.startsWith('+') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                                                        }`}>
                                                        {item.difference}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div className="h-80 mt-8">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Prediction vs Actual</h3>
                            <ReactECharts option={timeSeriesOption} style={{ height: '100%' }} />
                        </div>
                    </div>
                )}

                {activeTab === 'visualization' && (
                    <div className="p-6">
                        <div className="mb-8">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">SHAP Values (Feature Importance)</h3>
                            <p className="text-sm text-gray-600 mb-4">
                                SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the prediction.
                            </p>
                            <div className="h-80">
                                <ReactECharts option={shapOption} style={{ height: '100%' }} />
                            </div>
                        </div>

                        <div className="mt-12">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">LIME Explanation</h3>
                            <p className="text-sm text-gray-600 mb-4">
                                LIME (Local Interpretable Model-agnostic Explanations) shows how features contribute to this specific prediction.
                            </p>
                            <div className="h-80">
                                <ReactECharts option={limeOption} style={{ height: '100%' }} />
                            </div>
                        </div>

                        <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                            <h4 className="font-medium text-gray-800 mb-2">Interpretation</h4>
                            <p className="text-sm text-gray-600">
                                This prediction was primarily influenced by <strong>Interest Rate</strong> and <strong>Market Volume</strong>, which
                                together account for nearly 60% of the model's decision. The <strong>Previous Close</strong> and <strong>EMA</strong>
                                were secondary factors. This suggests that macroeconomic factors are currently more significant than technical indicators
                                for this particular prediction.
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex space-x-4">
                <button className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded focus:outline-none focus:shadow-outline">
                    Download Report
                </button>
                <button className="bg-white hover:bg-gray-100 text-gray-800 font-medium py-2 px-4 rounded border focus:outline-none focus:shadow-outline">
                    Share Results
                </button>
                <button className="bg-white hover:bg-gray-100 text-gray-800 font-medium py-2 px-4 rounded border focus:outline-none focus:shadow-outline">
                    Run New Prediction
                </button>
            </div>
        </div>
    );
};

export default Results;
