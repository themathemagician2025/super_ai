/**
 * Super AI Prediction System - Main JavaScript
 * Handles interactive elements and API connections
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function () {
    // Initialize tab functionality
    initializeTabs();

    // Initialize form submissions
    initializeFormSubmissions();

    // Load system status
    loadSystemStatus();

    // Create sample charts
    createSampleCharts();
});

/**
 * Initialize tab switching functionality
 */
function initializeTabs() {
    const tabs = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', function (e) {
            e.preventDefault();

            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('show', 'active'));

            // Add active class to current tab and content
            this.classList.add('active');
            const targetId = this.getAttribute('href').substring(1);
            document.getElementById(targetId).classList.add('show', 'active');
        });
    });
}

/**
 * Initialize form submission handling
 */
function initializeFormSubmissions() {
    // Sports prediction form
    const sportsForm = document.getElementById('sports-prediction-form');
    if (sportsForm) {
        sportsForm.addEventListener('submit', function (e) {
            e.preventDefault();
            const formData = new FormData(this);

            // Show loading spinner
            document.getElementById('sports-result').innerHTML = '<div class="spinner"></div> Processing...';

            // Get form data
            const homeTeam = document.getElementById('home-team').value;
            const awayTeam = document.getElementById('away-team').value;
            const sportType = document.getElementById('sport-type').value;
            const matchDate = document.getElementById('match-date').value;

            // Construct features array - this is what the backend expects
            // Note: In a real implementation, we would need to convert team names to numeric features
            // Here we're using placeholder values that match the expected input size
            const features = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

            // Make API request to the correct endpoint
            fetch('/api/predictions/sports', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    features: features,
                    teams: [homeTeam, awayTeam],
                    match_date: matchDate,
                    sport_type: sportType
                })
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.status === 'success') {
                        // Create prediction result object with the API response
                        const prediction = {
                            win_probability: data.home_win_probability,
                            confidence: 0.85, // This could come from the API
                            factors: {
                                player_performance: 0.65,
                                team_history: 0.72,
                                opponent_analysis: 0.58
                            }
                        };

                        // Display the prediction
                        displaySportsPrediction(prediction);

                        // Add to recent activity
                        addRecentActivity(`Sports prediction for ${homeTeam} vs ${awayTeam}`);
                    } else {
                        document.getElementById('sports-result').innerHTML =
                            `<div class="alert alert-danger">Error: ${data.message}</div>`;
                    }
                })
                .catch(error => {
                    console.error('Prediction error:', error);
                    document.getElementById('sports-result').innerHTML =
                        `<div class="alert alert-danger">Error: ${error.message}</div>`;
                });
        });
    }

    // Betting prediction form
    const bettingForm = document.getElementById('betting-prediction-form');
    if (bettingForm) {
        bettingForm.addEventListener('submit', function (e) {
            e.preventDefault();

            // Show loading spinner
            document.getElementById('betting-result').innerHTML = '<div class="spinner"></div> Processing...';

            // Get form data
            const eventName = document.getElementById('event-name').value;
            const bettingOdds = parseFloat(document.getElementById('betting-odds').value);
            const stakeAmount = parseFloat(document.getElementById('stake-amount').value);
            const riskLevel = document.getElementById('risk-level').value;

            // Construct features array
            const features = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

            // Make API request to the correct endpoint
            fetch('/api/predictions/betting', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    features: features,
                    event: eventName,
                    odds: bettingOdds,
                    stake: stakeAmount,
                    risk_level: riskLevel
                })
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.status === 'success') {
                        // Create betting prediction result object
                        const prediction = {
                            expected_value: data.expected_return,
                            risk_level: riskLevel,
                            confidence: data.prediction,
                            odds_assessment: {
                                fair_odds: bettingOdds * 0.95, // Example calculation
                                market_odds: bettingOdds,
                                edge: bettingOdds * 0.05 // Example calculation
                            }
                        };

                        // Display the prediction
                        displayBettingPrediction(prediction);

                        // Add to recent activity
                        addRecentActivity(`Betting prediction for ${eventName}`);
                    } else {
                        document.getElementById('betting-result').innerHTML =
                            `<div class="alert alert-danger">Error: ${data.message}</div>`;
                    }
                })
                .catch(error => {
                    console.error('Prediction error:', error);
                    document.getElementById('betting-result').innerHTML =
                        `<div class="alert alert-danger">Error: ${error.message}</div>`;
                });
        });
    }
}

/**
 * Add item to recent activity list
 * @param {string} text - Activity text
 */
function addRecentActivity(text) {
    const activityList = document.getElementById('recent-activity');
    if (activityList) {
        // Remove "No recent activity" message if present
        if (activityList.innerHTML.includes('No recent activity')) {
            activityList.innerHTML = '';
        }

        // Add new activity
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${text}</span>
                <small class="text-muted">${timeStr}</small>
            </div>
        `;
        activityList.prepend(li);
    }
}

/**
 * Display sports prediction results
 * @param {Object} data - The prediction data
 */
function displaySportsPrediction(data) {
    const resultContainer = document.getElementById('sports-result');

    let confidenceClass = 'confidence-medium';
    if (data.confidence > 0.8) confidenceClass = 'confidence-high';
    if (data.confidence < 0.6) confidenceClass = 'confidence-low';

    resultContainer.innerHTML = `
        <div class="prediction-result">
            <div class="row">
                <div class="col-md-6">
                    <h3>Win Probability</h3>
                    <div class="prediction-value">${Math.round(data.win_probability * 100)}%</div>
                    <p>Confidence: <span class="${confidenceClass}">${Math.round(data.confidence * 100)}%</span></p>
                </div>
                <div class="col-md-6">
                    <h3>Key Factors</h3>
                    <ul class="list-group">
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Player Performance
                            <span class="badge bg-primary rounded-pill">${Math.round(data.factors.player_performance * 100)}%</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Team History
                            <span class="badge bg-primary rounded-pill">${Math.round(data.factors.team_history * 100)}%</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Opponent Analysis
                            <span class="badge bg-primary rounded-pill">${Math.round(data.factors.opponent_analysis * 100)}%</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    `;

    // Create visualization
    createSportsPredictionChart(data);
}

/**
 * Display betting prediction results
 * @param {Object} data - The prediction data
 */
function displayBettingPrediction(data) {
    const resultContainer = document.getElementById('betting-result');

    let confidenceClass = 'confidence-medium';
    if (data.confidence > 0.8) confidenceClass = 'confidence-high';
    if (data.confidence < 0.6) confidenceClass = 'confidence-low';

    resultContainer.innerHTML = `
        <div class="prediction-result">
            <div class="row">
                <div class="col-md-6">
                    <h3>Expected Value</h3>
                    <div class="prediction-value">$${data.expected_value.toFixed(2)}</div>
                    <p>Risk Level: ${data.risk_level}</p>
                    <p>Confidence: <span class="${confidenceClass}">${Math.round(data.confidence * 100)}%</span></p>
                </div>
                <div class="col-md-6">
                    <h3>Odds Assessment</h3>
                    <ul class="list-group">
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Fair Odds
                            <span class="badge bg-info rounded-pill">${data.odds_assessment.fair_odds.toFixed(2)}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Market Odds
                            <span class="badge bg-info rounded-pill">${data.odds_assessment.market_odds.toFixed(2)}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Edge
                            <span class="badge bg-success rounded-pill">${data.odds_assessment.edge.toFixed(2)}</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    `;

    // Create visualization
    createBettingPredictionChart(data);
}

/**
 * Load system status from API
 */
function loadSystemStatus() {
    // Elements to update
    const systemStatusBadge = document.getElementById('system-status-badge');
    const systemStatusIndicators = document.querySelectorAll('.system-status-indicator');

    // Simulate API call (replace with actual fetch in production)
    setTimeout(() => {
        const status = {
            system: 'online',
            api: 'online',
            sports_model: 'online',
            betting_model: 'warning'
        };

        // Update main badge
        if (systemStatusBadge) {
            if (status.system === 'online') {
                systemStatusBadge.textContent = 'Online';
                systemStatusBadge.className = 'badge bg-success';
            } else if (status.system === 'warning') {
                systemStatusBadge.textContent = 'Warning';
                systemStatusBadge.className = 'badge bg-warning text-dark';
            } else {
                systemStatusBadge.textContent = 'Offline';
                systemStatusBadge.className = 'badge bg-danger';
            }
        }

        // Update system info tab indicators
        document.querySelectorAll('[data-system-component]').forEach(el => {
            const component = el.getAttribute('data-system-component');
            const statusIndicator = el.querySelector('.status-indicator');

            if (statusIndicator && status[component]) {
                statusIndicator.className = 'status-indicator';
                if (status[component] === 'online') {
                    statusIndicator.classList.add('status-online');
                } else if (status[component] === 'warning') {
                    statusIndicator.classList.add('status-warning');
                } else {
                    statusIndicator.classList.add('status-offline');
                }
            }
        });
    }, 1000);
}

/**
 * Create sample charts for the dashboard
 */
function createSampleCharts() {
    // Sample data for sports predictions
    const sportsData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        datasets: [{
            label: 'Accuracy',
            data: [65, 72, 68, 75, 82, 80, 85],
            borderColor: '#4361ee',
            backgroundColor: 'rgba(67, 97, 238, 0.1)',
            fill: true
        }]
    };

    // Sample data for betting predictions
    const bettingData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        datasets: [{
            label: 'ROI (%)',
            data: [5.2, 7.8, 6.5, 10.2, 8.7, 12.1, 14.5],
            borderColor: '#f72585',
            backgroundColor: 'rgba(247, 37, 133, 0.1)',
            fill: true
        }]
    };

    // Create sports chart if canvas exists
    const sportsChartCanvas = document.getElementById('sports-chart');
    if (sportsChartCanvas) {
        new Chart(sportsChartCanvas, {
            type: 'line',
            data: sportsData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 50,
                        max: 100
                    }
                }
            }
        });
    }

    // Create betting chart if canvas exists
    const bettingChartCanvas = document.getElementById('betting-chart');
    if (bettingChartCanvas) {
        new Chart(bettingChartCanvas, {
            type: 'line',
            data: bettingData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

/**
 * Create sports prediction visualization chart
 * @param {Object} data - The prediction data
 */
function createSportsPredictionChart(data) {
    const canvas = document.getElementById('sports-prediction-chart');
    if (!canvas) return;

    // Clear any existing chart
    if (canvas._chart) {
        canvas._chart.destroy();
    }

    // Create new chart
    const chart = new Chart(canvas, {
        type: 'pie',
        data: {
            labels: ['Win', 'Loss'],
            datasets: [{
                data: [data.win_probability * 100, (1 - data.win_probability) * 100],
                backgroundColor: ['#4361ee', '#f72585']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });

    // Store reference to chart
    canvas._chart = chart;
}

/**
 * Create betting prediction visualization chart
 * @param {Object} data - The prediction data
 */
function createBettingPredictionChart(data) {
    const canvas = document.getElementById('betting-prediction-chart');
    if (!canvas) return;

    // Clear any existing chart
    if (canvas._chart) {
        canvas._chart.destroy();
    }

    // Create new chart
    const chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: ['Fair Odds', 'Market Odds'],
            datasets: [{
                label: 'Odds Comparison',
                data: [data.odds_assessment.fair_odds, data.odds_assessment.market_odds],
                backgroundColor: ['#4895ef', '#f8961e']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Store reference to chart
    canvas._chart = chart;
}
