/**
 * Super AI - Advanced Visualization
 * Handles chart creation and data visualization for prediction results
 */

// Chart configuration
const chartConfig = {
    sports: {
        type: 'doughnut',
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Sports Prediction Probability'
                }
            }
        }
    },
    betting: {
        type: 'bar',
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Betting Prediction Analysis'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    },
    evolution: {
        type: 'line',
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Model Evolution Over Time'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Accuracy'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    }
};

// Color schemes
const colorSchemes = {
    primary: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
    light: ['#85c1e9', '#82e0aa', '#f1948a', '#f8c471', '#d7bde2'],
    dark: ['#1a5276', '#186a3b', '#7b241c', '#9c640c', '#5b2c6f']
};

// Initialize charts
function initCharts() {
    console.log('Initializing charts...');

    // Only proceed if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded. Skipping chart initialization.');
        return;
    }

    // Clear any existing chart instances
    Object.values(window.SuperAI.chartInstances || {}).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });

    // Initialize empty chart instances
    window.SuperAI = window.SuperAI || {};
    window.SuperAI.chartInstances = {};

    // Create sports prediction chart
    const sportsChartCanvas = document.getElementById('sports-chart');
    if (sportsChartCanvas) {
        window.SuperAI.chartInstances.sports = new Chart(sportsChartCanvas, {
            type: chartConfig.sports.type,
            data: {
                labels: ['Home Win', 'Away Win'],
                datasets: [{
                    label: 'Probability',
                    data: [0, 0],
                    backgroundColor: [colorSchemes.primary[0], colorSchemes.primary[2]],
                    hoverOffset: 4
                }]
            },
            options: chartConfig.sports.options
        });
    }

    // Create betting prediction chart
    const bettingChartCanvas = document.getElementById('betting-chart');
    if (bettingChartCanvas) {
        window.SuperAI.chartInstances.betting = new Chart(bettingChartCanvas, {
            type: chartConfig.betting.type,
            data: {
                labels: ['Prediction', 'Expected Return', 'Expected Value'],
                datasets: [{
                    label: 'Values',
                    data: [0, 0, 0],
                    backgroundColor: [
                        colorSchemes.primary[1],
                        colorSchemes.primary[3],
                        colorSchemes.primary[4]
                    ],
                    borderColor: [
                        colorSchemes.dark[1],
                        colorSchemes.dark[3],
                        colorSchemes.dark[4]
                    ],
                    borderWidth: 1
                }]
            },
            options: chartConfig.betting.options
        });
    }

    // Create evolution chart
    const evolutionChartCanvas = document.getElementById('evolution-chart');
    if (evolutionChartCanvas) {
        window.SuperAI.chartInstances.evolution = new Chart(evolutionChartCanvas, {
            type: chartConfig.evolution.type,
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Sports Model',
                        data: [],
                        borderColor: colorSchemes.primary[0],
                        backgroundColor: colorSchemes.light[0],
                        tension: 0.1
                    },
                    {
                        label: 'Betting Model',
                        data: [],
                        borderColor: colorSchemes.primary[1],
                        backgroundColor: colorSchemes.light[1],
                        tension: 0.1
                    }
                ]
            },
            options: chartConfig.evolution.options
        });
    }
}

// Visualize prediction result
function visualizePrediction(result, type) {
    if (typeof Chart === 'undefined' || !window.SuperAI.chartInstances) {
        console.warn('Charts not initialized. Skipping visualization.');
        return;
    }

    if (type === 'sports' && window.SuperAI.chartInstances.sports) {
        updateSportsChart(result);
    } else if (type === 'betting' && window.SuperAI.chartInstances.betting) {
        updateBettingChart(result);
    }

    // Show the chart container
    const chartContainer = document.getElementById(`${type}-chart-container`);
    if (chartContainer) {
        chartContainer.style.display = 'block';
    }
}

// Update sports prediction chart
function updateSportsChart(result) {
    const chart = window.SuperAI.chartInstances.sports;

    // Update chart data
    chart.data.datasets[0].data = [
        result.home_win_probability,
        result.away_win_probability
    ];

    chart.update();
}

// Update betting prediction chart
function updateBettingChart(result) {
    const chart = window.SuperAI.chartInstances.betting;

    // Update chart data
    chart.data.datasets[0].data = [
        result.prediction,
        result.expected_return / 100, // Scale for better visualization
        result.expected_value
    ];

    chart.update();
}

// Update prediction charts with historical data
function updatePredictionCharts(predictions) {
    if (typeof Chart === 'undefined' || !window.SuperAI.chartInstances) {
        return;
    }

    // Find the evolution chart
    const evolutionChart = window.SuperAI.chartInstances.evolution;
    if (!evolutionChart) return;

    // Process the predictions data to extract time series
    const timestamps = [];
    const sportsData = [];
    const bettingData = [];

    // Extract data from predictions (this is mockup - adjust based on actual data format)
    Object.entries(predictions).forEach(([key, pred]) => {
        if (pred.start_time) {
            const time = new Date(pred.start_time * 1000).toLocaleTimeString();
            timestamps.push(time);

            if (key.includes('sports')) {
                const accuracy = (pred.evaluation_metrics && pred.evaluation_metrics.accuracy) ||
                    (Math.random() * 0.3 + 0.7); // Mockup data
                sportsData.push(accuracy);

                // Add null for betting to maintain alignment
                if (!key.includes('betting')) {
                    bettingData.push(null);
                }
            }

            if (key.includes('betting')) {
                const accuracy = (pred.evaluation_metrics && pred.evaluation_metrics.r2) ||
                    (Math.random() * 0.4 + 0.5); // Mockup data
                bettingData.push(accuracy);

                // Add null for sports to maintain alignment
                if (!key.includes('sports')) {
                    sportsData.push(null);
                }
            }
        }
    });

    // If we have no real data, add some mockup data for demonstration
    if (timestamps.length === 0) {
        const now = new Date();
        for (let i = 0; i < 5; i++) {
            const time = new Date(now - i * 60000).toLocaleTimeString();
            timestamps.unshift(time);
            sportsData.unshift(Math.random() * 0.3 + 0.7);
            bettingData.unshift(Math.random() * 0.4 + 0.5);
        }
    }

    // Update chart data
    evolutionChart.data.labels = timestamps;

    if (evolutionChart.data.datasets.length >= 2) {
        evolutionChart.data.datasets[0].data = sportsData;
        evolutionChart.data.datasets[1].data = bettingData;
    }

    evolutionChart.update();
}

// Load and display evolution stats image
function loadEvolutionImage() {
    const imgContainer = document.getElementById('evolution-image-container');
    if (!imgContainer) return;

    // Try to load the evolution stats image
    const img = new Image();
    img.onload = function () {
        imgContainer.innerHTML = '';
        imgContainer.appendChild(img);
    };

    img.onerror = function () {
        imgContainer.innerHTML = '<p>Evolution stats image not available</p>';
    };

    img.src = '/static/images/evolution_stats.png';
    img.alt = 'Model Evolution Statistics';
    img.className = 'evolution-image';
}

// Set up visualization after the page loads
document.addEventListener('DOMContentLoaded', () => {
    // If Chart.js is loaded, initialize charts
    if (typeof Chart !== 'undefined') {
        initCharts();
    } else {
        console.warn('Chart.js not loaded. Visualizations will be limited.');
    }

    // Try to load evolution stats image
    loadEvolutionImage();
});

// Export functions for use in main.js
window.initCharts = initCharts;
window.visualizePrediction = visualizePrediction;
window.updatePredictionCharts = updatePredictionCharts;
