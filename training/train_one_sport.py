# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Train Single Sport Model Script

A simplified version of train_all_sports.py that trains a model for just one sport.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_all_sports import train_sports_model

if __name__ == "__main__":
    # Train a model for American Football as a demonstration
    sport = 'american_football'
    print(f"Training model for {sport}...")
    model_path = train_sports_model(sport)

    if model_path:
        print(f"Successfully trained and saved model to {model_path}")
    else:
        print(f"Failed to train model for {sport}")
