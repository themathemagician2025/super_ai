/**
 * Visualization.js - Advanced visualization utilities for the Super AI Prediction System
 * Handles chart creation and data visualization using Chart.js
 */

// Create a namespace for advanced visualization functions
window.AdvancedVisualization = (function () {
    // Private variables for chart instances
    const charts = {
        dashboardSportsChart: null,
        dashboardBettingChart: null,
        sportsChart: null,
        bettingChart: null
    };

    // Chart configuration defaults
    const chartDefaults = {
        backgroundColor: [
            'rgba(54, 162, 235, 0.5)',
            'rgba(255, 99, 132, 0.5)',
            'rgba(75, 192, 192, 0.5)',
            'rgba(255, 206, 86, 0.5)'
        ],
        borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 99, 132, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(255, 206, 86, 1)'
        ],
        borderWidth: 1,
        fontColor: '#444'
    };

    /**
     * Initialize all charts on the dashboard
     */
    function initDashboardCharts() {
        initSportsDashboardChart();
        initBettingDashboardChart();
    }

    /**
     * Initialize the sports prediction chart on the dashboard
     */
    function initSportsDashboardChart() {
        const ctx = document.getElementById('dashboard-sports-chart');
        if (!ctx) return;

        charts.dashboardSportsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Home Win', 'Away Win'],
                datasets: [{
                    data: [50, 50], // Default data
                    backgroundColor: [
                        chartDefaults.backgroundColor[0],
                        chartDefaults.backgroundColor[1]
                    ],
                    borderColor: [
                        chartDefaults.borderColor[0],
                        chartDefaults.borderColor[1]
                    ],
                    borderWidth: chartDefaults.borderWidth
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: chartDefaults.fontColor
                        }
                    },
                    title: {
                        display: true,
                        text: 'Win Probability',
                        color: chartDefaults.fontColor
                    }
                }
            }
        });
    }

    /**
     * Initialize the betting prediction chart on the dashboard
     */
    function initBettingDashboardChart() {
        const ctx = document.getElementById('dashboard-betting-chart');
        if (!ctx) return;

        charts.dashboardBettingChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Expected Return', 'Stake', 'Net Profit'],
                datasets: [{
                    label: 'Betting Analysis',
                    data: [0, 0, 0], // Default data
                    backgroundColor: [
                        chartDefaults.backgroundColor[2],
                        chartDefaults.backgroundColor[3],
                        chartDefaults.backgroundColor[0]
                    ],
                    borderColor: [
                        chartDefaults.borderColor[2],
                        chartDefaults.borderColor[3],
                        chartDefaults.borderColor[0]
                    ],
                    borderWidth: chartDefaults.borderWidth
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Amount ($)',
                            color: chartDefaults.fontColor
                        },
                        ticks: {
                            color: chartDefaults.fontColor
                        }
                    },
                    x: {
                        ticks: {
                            color: chartDefaults.fontColor
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Betting Prediction',
                        color: chartDefaults.fontColor
                    }
                }
            }
        });
    }

    /**
     * Initialize sports prediction detail chart
     */
    function initSportsChart() {
        const ctx = document.getElementById('sports-prediction-chart');
        if (!ctx) return;

        charts.sportsChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Speed', 'Strength', 'Technique', 'Recent Form', 'Historical Performance'],
                datasets: [
                    {
                        label: 'Home Team',
                        data: [0, 0, 0, 0, 0],
                        fill: true,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgb(54, 162, 235)',
                        pointBackgroundColor: 'rgb(54, 162, 235)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgb(54, 162, 235)'
                    },
                    {
                        label: 'Away Team',
                        data: [0, 0, 0, 0, 0],
                        fill: true,
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgb(255, 99, 132)',
                        pointBackgroundColor: 'rgb(255, 99, 132)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgb(255, 99, 132)'
                    }
                ]
            },
            options: {
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: 'Team Comparison',
                        color: chartDefaults.fontColor
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 10
                    }
                }
            }
        });
    }

    /**
     * Initialize betting prediction detail chart
     */
    function initBettingChart() {
        const ctx = document.getElementById('betting-prediction-chart');
        if (!ctx) return;

        charts.bettingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    label: 'Expected Return',
                    data: [0, 0, 0],
                    fill: false,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    label: 'Recommended Stake',
                    data: [0, 0, 0],
                    fill: false,
                    borderColor: 'rgb(255, 205, 86)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Amount ($)',
                            color: chartDefaults.fontColor
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: 'Risk Analysis',
                        color: chartDefaults.fontColor
                    }
                }
            }
        });
    }

    /**
     * Initialize all charts on the page
     */
    function initAllCharts() {
        // Initialize dashboard charts
        initDashboardCharts();

        // Initialize detail charts
        initSportsChart();
        initBettingChart();
    }

    /**
     * Visualize sports prediction data
     * @param {Object} data - Sports prediction data
     */
    function visualizeSportsPrediction(data) {
        if (!data) return;

        // Update dashboard chart if it exists
        if (charts.dashboardSportsChart) {
            charts.dashboardSportsChart.data.datasets[0].data = [
                data.home_win_probability * 100,
                data.away_win_probability * 100
            ];
            charts.dashboardSportsChart.update();
        }

        // Update detailed sports chart
        if (charts.sportsChart) {
            // We'll simulate team stats data based on the prediction
            // In a real app, this would come from actual team statistics
            const homeValue = data.home_win_probability * 10;
            const awayValue = data.away_win_probability * 10;

            // Generate slightly randomized values around the base prediction
            const homeStats = [
                homeValue * (0.9 + Math.random() * 0.2),  // Speed
                homeValue * (0.8 + Math.random() * 0.4),  // Strength
                homeValue * (0.85 + Math.random() * 0.3), // Technique
                homeValue * (0.9 + Math.random() * 0.2),  // Recent Form
                homeValue * (0.9 + Math.random() * 0.2)   // Historical
            ];

            const awayStats = [
                awayValue * (0.9 + Math.random() * 0.2),  // Speed
                awayValue * (0.8 + Math.random() * 0.4),  // Strength
                awayValue * (0.85 + Math.random() * 0.3), // Technique
                awayValue * (0.9 + Math.random() * 0.2),  // Recent Form
                awayValue * (0.9 + Math.random() * 0.2)   // Historical
            ];

            charts.sportsChart.data.datasets[0].data = homeStats;
            charts.sportsChart.data.datasets[1].data = awayStats;
            charts.sportsChart.update();
        }
    }

    /**
     * Visualize betting prediction data
     * @param {Object} data - Betting prediction data
     */
    function visualizeBettingPrediction(data) {
        if (!data) return;

        // Update dashboard chart if it exists
        if (charts.dashboardBettingChart) {
            const stake = data.stake || 0;
            const expectedReturn = data.expected_return || 0;
            const netProfit = expectedReturn - stake;

            charts.dashboardBettingChart.data.datasets[0].data = [
                expectedReturn,
                stake,
                netProfit
            ];
            charts.dashboardBettingChart.update();
        }

        // Update detailed betting chart
        if (charts.bettingChart) {
            const baseStake = data.stake || 100;
            const baseReturn = data.expected_return || 0;

            // Generate risk scenarios
            const lowRiskStake = baseStake * 0.5;
            const mediumRiskStake = baseStake;
            const highRiskStake = baseStake * 1.5;

            const lowRiskReturn = baseReturn * 0.7;
            const mediumRiskReturn = baseReturn;
            const highRiskReturn = baseReturn * 1.4;

            charts.bettingChart.data.datasets[0].data = [
                lowRiskReturn,
                mediumRiskReturn,
                highRiskReturn
            ];

            charts.bettingChart.data.datasets[1].data = [
                lowRiskStake,
                mediumRiskStake,
                highRiskStake
            ];

            charts.bettingChart.update();
        }
    }

    /**
     * Update all charts with system status data
     * @param {Object} data - System status data
     */
    function updateStatusCharts(data) {
        // Example implementation - would update status-related charts
        // This would be connected to any system status dashboard visualizations
    }

    /**
     * Update models information charts
     * @param {Object} data - Models data
     */
    function updateModelsCharts(data) {
        // Example implementation - would update model performance charts
        // This would visualize model accuracy, training history, etc.
    }

    // Initialize charts when the DOM is loaded
    document.addEventListener('DOMContentLoaded', function () {
        // Check if Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded. Visualizations will not be available.');
            return;
        }

        // Initialize all charts
        initAllCharts();
    });

    // Return public API
    return {
        initAllCharts,
        visualizeSportsPrediction,
        visualizeBettingPrediction,
        updateStatusCharts,
        updateModelsCharts
    };
})();
