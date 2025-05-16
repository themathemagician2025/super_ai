/**
 * Frontend Tests for Super AI Prediction System
 *
 * This module contains Jest tests for the UI components
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import axios from 'axios';

// Import components
import Dashboard from '../components/Dashboard';
import ForexPrediction from '../components/ForexPrediction';
import SportsPrediction from '../components/SportsPrediction';
import NavBar from '../components/NavBar';

// Mock axios
jest.mock('axios');

// Mock API response data
const mockDashboardData = {
    predictions_today: 1250,
    accuracy_7d: 0.91,
    active_users: 328,
    avg_response_time_ms: 125
};

const mockForexPrediction = {
    predictions: [
        { date: '2023-11-01', predicted_price: 1.1234 },
        { date: '2023-11-02', predicted_price: 1.1250 }
    ],
    currency_pair: 'EUR/USD',
    model_used: 'forex_rf'
};

const mockSportsPrediction = {
    match: 'Manchester United vs Liverpool',
    league: 'Premier League',
    predicted_outcome: 'home_win',
    probabilities: {
        home_win: 0.55,
        draw: 0.25,
        away_win: 0.20
    },
    model_used: 'sports_rf'
};

// Dashboard Component Tests
describe('Dashboard Component', () => {
    beforeEach(() => {
        axios.get.mockResolvedValue({ data: mockDashboardData });
    });

    test('renders dashboard correctly', async () => {
        render(<Dashboard />);

        // Check if loading state is displayed initially
        expect(screen.getByText(/loading dashboard/i)).toBeInTheDocument();

        // Wait for data to load
        await waitFor(() => {
            expect(screen.getByText(/predictions today/i)).toBeInTheDocument();
        });

        // Check if dashboard metrics are displayed
        expect(screen.getByText('1,250')).toBeInTheDocument(); // predictions_today
        expect(screen.getByText('91%')).toBeInTheDocument(); // accuracy_7d
        expect(screen.getByText('328')).toBeInTheDocument(); // active_users

        // Check if refresh button is available
        expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
    });

    test('refreshes data when refresh button is clicked', async () => {
        render(<Dashboard />);

        // Wait for initial data to load
        await waitFor(() => {
            expect(screen.getByText(/predictions today/i)).toBeInTheDocument();
        });

        // Click refresh button
        fireEvent.click(screen.getByRole('button', { name: /refresh/i }));

        // Check if API was called again
        expect(axios.get).toHaveBeenCalledTimes(2);
        expect(axios.get).toHaveBeenCalledWith('/api/dashboard_metrics');
    });

    test('handles error state', async () => {
        // Mock API error
        axios.get.mockRejectedValueOnce(new Error('Failed to fetch data'));

        render(<Dashboard />);

        // Wait for error state
        await waitFor(() => {
            expect(screen.getByText(/error loading dashboard/i)).toBeInTheDocument();
        });

        // Check if retry button is displayed
        expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });
});

// ForexPrediction Component Tests
describe('ForexPrediction Component', () => {
    beforeEach(() => {
        axios.post.mockResolvedValue({ data: mockForexPrediction });
    });

    test('renders form correctly', () => {
        render(<ForexPrediction />);

        // Check form elements
        expect(screen.getByLabelText(/currency pair/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/days ahead/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/current price/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /predict/i })).toBeInTheDocument();
    });

    test('submits form and displays prediction', async () => {
        render(<ForexPrediction />);

        // Fill form
        await userEvent.selectOptions(screen.getByLabelText(/currency pair/i), 'EUR/USD');
        await userEvent.type(screen.getByLabelText(/days ahead/i), '2');
        await userEvent.type(screen.getByLabelText(/current price/i), '1.12');

        // Submit form
        fireEvent.click(screen.getByRole('button', { name: /predict/i }));

        // Check if API was called with correct data
        expect(axios.post).toHaveBeenCalledWith('/api/forex/predict', expect.objectContaining({
            currency_pair: 'EUR/USD',
            days_ahead: '2',
            current_data: expect.objectContaining({
                open: 1.12
            })
        }));

        // Wait for prediction to be displayed
        await waitFor(() => {
            expect(screen.getByText(/prediction results/i)).toBeInTheDocument();
        });

        // Check prediction details
        expect(screen.getByText('EUR/USD')).toBeInTheDocument();
        expect(screen.getByText('1.1234')).toBeInTheDocument(); // first prediction
        expect(screen.getByText('1.1250')).toBeInTheDocument(); // second prediction
    });

    test('validates form input', async () => {
        render(<ForexPrediction />);

        // Submit empty form
        fireEvent.click(screen.getByRole('button', { name: /predict/i }));

        // Check for validation messages
        await waitFor(() => {
            expect(screen.getByText(/please select a currency pair/i)).toBeInTheDocument();
            expect(screen.getByText(/please enter current price/i)).toBeInTheDocument();
        });

        // API should not be called
        expect(axios.post).not.toHaveBeenCalled();
    });
});

// SportsPrediction Component Tests
describe('SportsPrediction Component', () => {
    beforeEach(() => {
        axios.post.mockResolvedValue({ data: mockSportsPrediction });
    });

    test('renders form correctly', () => {
        render(<SportsPrediction />);

        // Check form elements
        expect(screen.getByLabelText(/home team/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/away team/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/league/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /predict/i })).toBeInTheDocument();
    });

    test('submits form and displays prediction', async () => {
        render(<SportsPrediction />);

        // Fill form
        await userEvent.type(screen.getByLabelText(/home team/i), 'Manchester United');
        await userEvent.type(screen.getByLabelText(/away team/i), 'Liverpool');
        await userEvent.selectOptions(screen.getByLabelText(/league/i), 'Premier League');

        // Submit form
        fireEvent.click(screen.getByRole('button', { name: /predict/i }));

        // Check if API was called with correct data
        expect(axios.post).toHaveBeenCalledWith('/api/sports/predict', expect.objectContaining({
            home_team: 'Manchester United',
            away_team: 'Liverpool',
            league: 'Premier League'
        }));

        // Wait for prediction to be displayed
        await waitFor(() => {
            expect(screen.getByText(/prediction results/i)).toBeInTheDocument();
        });

        // Check prediction details
        expect(screen.getByText(/manchester united vs liverpool/i)).toBeInTheDocument();
        expect(screen.getByText(/home win/i)).toBeInTheDocument(); // predicted outcome
        expect(screen.getByText('55%')).toBeInTheDocument(); // home win probability
    });

    test('displays visual representation of prediction', async () => {
        render(<SportsPrediction />);

        // Fill and submit form
        await userEvent.type(screen.getByLabelText(/home team/i), 'Manchester United');
        await userEvent.type(screen.getByLabelText(/away team/i), 'Liverpool');
        await userEvent.selectOptions(screen.getByLabelText(/league/i), 'Premier League');
        fireEvent.click(screen.getByRole('button', { name: /predict/i }));

        // Wait for prediction to be displayed
        await waitFor(() => {
            expect(screen.getByText(/prediction results/i)).toBeInTheDocument();
        });

        // Check if probability bars are displayed with correct widths
        const homeWinBar = screen.getByTestId('probability-bar-home_win');
        const drawBar = screen.getByTestId('probability-bar-draw');
        const awayWinBar = screen.getByTestId('probability-bar-away_win');

        expect(homeWinBar).toHaveStyle('width: 55%');
        expect(drawBar).toHaveStyle('width: 25%');
        expect(awayWinBar).toHaveStyle('width: 20%');
    });
});

// NavBar Component Tests
describe('NavBar Component', () => {
    test('renders navigation links correctly', () => {
        render(<NavBar />);

        // Check navigation links
        expect(screen.getByRole('link', { name: /home/i })).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /forex/i })).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /sports/i })).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /dashboard/i })).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /api docs/i })).toBeInTheDocument();
    });

    test('highlights active link', () => {
        // Mock current path as /forex
        const mockUseLocation = jest.fn().mockReturnValue({
            pathname: '/forex'
        });

        jest.mock('react-router-dom', () => ({
            ...jest.requireActual('react-router-dom'),
            useLocation: mockUseLocation
        }));

        render(<NavBar />);

        // Check if forex link has active class
        const forexLink = screen.getByRole('link', { name: /forex/i });
        expect(forexLink).toHaveClass('active');

        // Other links should not have active class
        const homeLink = screen.getByRole('link', { name: /home/i });
        expect(homeLink).not.toHaveClass('active');
    });

    test('mobile menu toggle works', () => {
        render(<NavBar />);

        // Menu should be initially closed (on mobile)
        const menuLinks = screen.getByTestId('nav-links');
        expect(menuLinks).not.toHaveClass('open');

        // Click hamburger icon
        const menuToggle = screen.getByLabelText(/toggle menu/i);
        fireEvent.click(menuToggle);

        // Menu should be open
        expect(menuLinks).toHaveClass('open');

        // Click again to close
        fireEvent.click(menuToggle);
        expect(menuLinks).not.toHaveClass('open');
    });
});

// Accessibility Tests
describe('Accessibility', () => {
    test('dashboard has proper heading structure', async () => {
        axios.get.mockResolvedValue({ data: mockDashboardData });
        render(<Dashboard />);

        await waitFor(() => {
            expect(screen.getByRole('heading', { name: /dashboard/i, level: 1 })).toBeInTheDocument();
            expect(screen.getByRole('heading', { name: /metrics/i, level: 2 })).toBeInTheDocument();
        });
    });

    test('form inputs have associated labels', () => {
        render(<ForexPrediction />);

        const currencyInput = screen.getByLabelText(/currency pair/i);
        const daysInput = screen.getByLabelText(/days ahead/i);

        expect(currencyInput).toHaveAttribute('id');
        expect(daysInput).toHaveAttribute('id');

        // Check if labels are properly associated with inputs
        const currencyLabel = screen.getByText(/currency pair/i).closest('label');
        expect(currencyLabel).toHaveAttribute('for', currencyInput.id);
    });

    test('interactive elements are keyboard navigable', async () => {
        render(<ForexPrediction />);

        // Set focus on the first form element
        const currencyInput = screen.getByLabelText(/currency pair/i);
        currencyInput.focus();
        expect(document.activeElement).toBe(currencyInput);

        // Tab to next field
        userEvent.tab();
        expect(document.activeElement).toBe(screen.getByLabelText(/days ahead/i));

        // Tab to next field
        userEvent.tab();
        expect(document.activeElement).toBe(screen.getByLabelText(/current price/i));

        // Tab to submit button
        userEvent.tab();
        expect(document.activeElement).toBe(screen.getByRole('button', { name: /predict/i }));

        // Press Enter to submit
        fireEvent.keyDown(document.activeElement, { key: 'Enter', code: 'Enter' });
        expect(axios.post).toHaveBeenCalled();
    });
});
