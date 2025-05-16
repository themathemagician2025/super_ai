# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import glob
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model classes
from prediction.sports_predictor import SportsPredictor
from betting.betting_prediction import BettingPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_all_csv_files(data_dir):
    """
    Load all CSV files from the data directory
    Returns a dictionary of DataFrames with filenames as keys
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('*.csv'))

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    dataframes = {}
    for csv_file in csv_files:
        try:
            logger.info(f"Loading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            # Only keep files with enough data
            if len(df) >= 10:
                dataframes[csv_file.name] = df
                logger.info(f"  Loaded {csv_file.name}: {df.shape}, columns: {df.columns.tolist()[:5]}...")
            else:
                logger.warning(f"  Skipping {csv_file.name}: Insufficient data ({len(df)} rows)")
        except Exception as e:
            logger.error(f"  Error loading {csv_file.name}: {str(e)}")

    logger.info(f"Successfully loaded {len(dataframes)} CSV files")
    return dataframes

def load_json_files(data_dir):
    """
    Load all JSON files from the data directory
    Returns a dictionary of data with filenames as keys
    """
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))

    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

    json_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            json_data[json_file.name] = data
            logger.info(f"Loaded {json_file.name}")
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {str(e)}")

    return json_data

def identify_dataset_type(df):
    """
    Identify if a dataset is suitable for sports prediction or betting prediction
    Returns 'sports', 'betting', or 'unknown'
    """
    columns = df.columns.tolist()

    # Check for sports prediction relevant columns
    sports_indicators = ['home_score', 'away_score', 'result', 'win', 'loss', 'team', 'match']
    sports_matches = sum(1 for indicator in sports_indicators if any(indicator in col.lower() for col in columns))

    # Check for betting prediction relevant columns
    betting_indicators = ['odds', 'bet', 'stake', 'return', 'profit', 'payout']
    betting_matches = sum(1 for indicator in betting_indicators if any(indicator in col.lower() for col in columns))

    # Decide based on matches
    if sports_matches > betting_matches:
        return 'sports'
    elif betting_matches > 0:
        return 'betting'
    elif sports_matches > 0:
        return 'sports'
    else:
        return 'unknown'

def prepare_sports_data(df):
    """Prepare data for sports prediction model"""
    logger.info("Preparing sports prediction data")

    # Basic cleaning
    df = df.dropna()

    # Make a copy to avoid warnings
    df = df.copy()

    # Handle categorical variables if present
    for col in df.select_dtypes(include=['object']).columns:
        # Skip date, team names and match_id columns
        if any(skip in col.lower() for skip in ['date', 'team', 'name', 'id']):
            continue

        # For other categorical columns, use one-hot encoding
        try:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, one_hot], axis=1)
            df.drop(col, axis=1, inplace=True)
        except:
            df.drop(col, axis=1, inplace=True)

    # Select relevant features and target
    if 'result' in df.columns:
        # Assume 'result' is the target (1 for win, 0 for loss)
        y = df['result'].values

        # Drop non-feature columns
        X = df.drop(['result'] + [col for col in df.columns if any(s in col.lower() for s in ['date', 'team', 'name', 'id'])], errors='ignore', axis=1)
    else:
        # Use a simple target model (e.g., home_score > away_score)
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['result'] = (df['home_score'] > df['away_score']).astype(int)
            y = df['result'].values

            # Drop non-feature columns
            X = df.drop(['result', 'home_score', 'away_score'] +
                       [col for col in df.columns if any(s in col.lower() for s in ['date', 'team', 'name', 'id'])],
                       errors='ignore', axis=1)
        else:
            # Try to find any win/loss column
            for col in df.columns:
                if any(indicator in col.lower() for indicator in ['win', 'outcome', 'result']):
                    try:
                        # Try to convert to binary
                        df['result'] = df[col].map(lambda x: 1 if str(x).lower() in ['win', 'w', '1', 'true', 'yes'] else 0)
                        y = df['result'].values

                        # Drop non-feature columns
                        X = df.drop(['result', col] +
                                    [c for c in df.columns if any(s in c.lower() for s in ['date', 'team', 'name', 'id'])],
                                    errors='ignore', axis=1)
                        break
                    except:
                        continue
            else:
                logger.error("Cannot determine target variable - required columns missing")
                return None, None

    # Ensure all data is numeric
    X = X.select_dtypes(include=[np.number])

    # Check if we have any features left
    if X.shape[1] == 0:
        logger.error("No numeric features available after preprocessing")
        return None, None

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
    logger.info(f"Feature columns: {X.columns.tolist()}")

    return X_scaled, y

def prepare_betting_data(df):
    """Prepare data for betting prediction model"""
    logger.info("Preparing betting prediction data")

    # Basic cleaning
    df = df.dropna()

    # Make a copy to avoid warnings
    df = df.copy()

    # Handle categorical variables if present
    for col in df.select_dtypes(include=['object']).columns:
        # Skip date, team names and match_id columns
        if any(skip in col.lower() for skip in ['date', 'team', 'name', 'id']):
            continue

        # For other categorical columns, use one-hot encoding
        try:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, one_hot], axis=1)
            df.drop(col, axis=1, inplace=True)
        except:
            df.drop(col, axis=1, inplace=True)

    # Select relevant features and target
    if 'odds' in df.columns:
        # Assume 'odds' is the target
        y = df['odds'].values

        # Drop non-feature columns
        X = df.drop(['odds'] + [col for col in df.columns if any(s in col.lower() for s in ['date', 'team', 'name', 'id'])], errors='ignore', axis=1)
    else:
        # Look for other typical odds column names
        odds_columns = [col for col in df.columns if any(s in col.lower() for s in ['odds', 'price', 'payout', 'return'])]

        if odds_columns and len(odds_columns) > 0:
            odds_col = odds_columns[0]
            y = df[odds_col].values

            # Drop non-feature columns
            X = df.drop([odds_col] + [col for col in df.columns if any(s in col.lower() for s in ['date', 'team', 'name', 'id'])], errors='ignore', axis=1)
        else:
            # Generate synthetic odds based on scores
            if 'home_score' in df.columns and 'away_score' in df.columns:
                # Simple model: odds inversely related to score difference
                df['odds'] = 1 + abs(df['home_score'] - df['away_score']) / 5
                y = df['odds'].values

                # Drop non-feature columns
                X = df.drop(['odds', 'home_score', 'away_score'] +
                           [col for col in df.columns if any(s in col.lower() for s in ['date', 'team', 'name', 'id'])],
                           errors='ignore', axis=1)
            else:
                logger.error("Cannot determine target variable - required columns missing")
                return None, None

    # Ensure all data is numeric
    X = X.select_dtypes(include=[np.number])

    # Check if we have any features left
    if X.shape[1] == 0:
        logger.error("No numeric features available after preprocessing")
        return None, None

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
    logger.info(f"Feature columns: {X.columns.tolist()}")

    return X_scaled, y

def train_sports_model(X, y, input_size, hidden_size=64, batch_size=32, epochs=50, learning_rate=0.001):
    """Train the sports prediction model"""
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    # Initialize model
    model = SportsPredictor(input_size=input_size, hidden_size=hidden_size, output_size=1)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            logger.info(f'Sports model - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

        # Calculate accuracy
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / len(y_test)

    logger.info(f'Sports model - Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}')

    return model

def train_betting_model(X, y, input_size, hidden_size=64, batch_size=32, epochs=50, learning_rate=0.001):
    """Train the betting prediction model"""
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    # Initialize model
    model = BettingPredictor(input_size=input_size, hidden_size=hidden_size, output_size=1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            logger.info(f'Betting model - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

        # Calculate mean absolute error
        mae = torch.abs(test_outputs - y_test).mean().item()

    logger.info(f'Betting model - Test Loss: {test_loss.item():.4f}, MAE: {mae:.4f}')

    return model

def save_models(sports_model, betting_model, model_dir):
    """Save the trained models"""
    model_dir = Path(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Save sports model
    torch.save(sports_model.state_dict(), model_dir / 'sports_predictor.pth')
    logger.info(f"Sports model saved to {model_dir / 'sports_predictor.pth'}")

    # Save betting model
    torch.save(betting_model.state_dict(), model_dir / 'betting_predictor.pth')
    logger.info(f"Betting model saved to {model_dir / 'betting_predictor.pth'}")

def main():
    logger.info("Starting enhanced model training using all CSV and JSON files")

    # Get paths
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = root_dir / 'data'
    model_dir = root_dir / 'models'

    # Load all available CSV files
    dataframes = load_all_csv_files(data_dir)

    # Load all available JSON files
    json_data = load_json_files(data_dir)

    # Categorize datasets
    sports_dataframes = []
    betting_dataframes = []

    # Process each dataset
    for name, df in dataframes.items():
        dataset_type = identify_dataset_type(df)

        if dataset_type == 'sports':
            sports_dataframes.append(df)
            logger.info(f"Using {name} for sports prediction (shape: {df.shape})")
        elif dataset_type == 'betting':
            betting_dataframes.append(df)
            logger.info(f"Using {name} for betting prediction (shape: {df.shape})")

    # Combine sports dataframes if possible
    if len(sports_dataframes) > 0:
        logger.info(f"Combining {len(sports_dataframes)} sports datasets")
        # Process each sports dataframe individually due to potential column differences
        sports_models = []
        for i, df in enumerate(sports_dataframes):
            X_sports, y_sports = prepare_sports_data(df)

            if X_sports is not None and y_sports is not None and X_sports.shape[1] > 0:
                # Train sports model
                input_size = X_sports.shape[1]
                logger.info(f"Training sports model {i+1}/{len(sports_dataframes)} (features: {input_size})")
                model = train_sports_model(X_sports, y_sports, input_size=input_size, epochs=30)
                sports_models.append(model)

        # Use the best sports model
        if len(sports_models) > 0:
            sports_model = sports_models[0]  # For simplicity, use the first one
            logger.info(f"Selected sports model from {len(sports_models)} trained models")
        else:
            logger.error("Failed to train any sports prediction models")
            sports_model = None
    else:
        logger.error("No suitable sports data found")
        sports_model = None

    # Combine betting dataframes if possible
    if len(betting_dataframes) > 0:
        logger.info(f"Combining {len(betting_dataframes)} betting datasets")
        # Process each betting dataframe individually due to potential column differences
        betting_models = []
        for i, df in enumerate(betting_dataframes):
            X_betting, y_betting = prepare_betting_data(df)

            if X_betting is not None and y_betting is not None and X_betting.shape[1] > 0:
                # Train betting model
                input_size = X_betting.shape[1]
                logger.info(f"Training betting model {i+1}/{len(betting_dataframes)} (features: {input_size})")
                model = train_betting_model(X_betting, y_betting, input_size=input_size, epochs=30)
                betting_models.append(model)

        # Use the best betting model
        if len(betting_models) > 0:
            betting_model = betting_models[0]  # For simplicity, use the first one
            logger.info(f"Selected betting model from {len(betting_models)} trained models")
        else:
            logger.error("Failed to train any betting prediction models")
            betting_model = None

    # If we don't have betting models, try to use sports data for betting
    elif sports_model is not None:
        logger.info("No suitable betting data found, attempting to repurpose sports data for betting")
        for df in sports_dataframes:
            X_betting, y_betting = prepare_betting_data(df)

            if X_betting is not None and y_betting is not None and X_betting.shape[1] > 0:
                # Train betting model
                input_size = X_betting.shape[1]
                betting_model = train_betting_model(X_betting, y_betting, input_size=input_size)
                logger.info("Successfully trained betting model using sports data")
                break
        else:
            logger.error("Failed to repurpose sports data for betting prediction")
            betting_model = None
    else:
        logger.error("No suitable betting data found")
        betting_model = None

    # Save models if training was successful
    if sports_model is not None and betting_model is not None:
        save_models(sports_model, betting_model, model_dir)
        logger.info("Enhanced training completed successfully")
    elif sports_model is not None:
        # Save just the sports model
        model_dir = Path(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(sports_model.state_dict(), model_dir / 'sports_predictor.pth')
        logger.info(f"Only sports model saved to {model_dir / 'sports_predictor.pth'}")
        logger.info("Enhanced training partially completed")
    elif betting_model is not None:
        # Save just the betting model
        model_dir = Path(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(betting_model.state_dict(), model_dir / 'betting_predictor.pth')
        logger.info(f"Only betting model saved to {model_dir / 'betting_predictor.pth'}")
        logger.info("Enhanced training partially completed")
    else:
        logger.error("Enhanced training completed with errors - no models saved")

if __name__ == "__main__":
    main()
