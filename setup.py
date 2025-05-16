# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
from pathlib import Path

def setup_directories():
    """Create required project directories"""
    base_dirs = [
        "logs",
        "data",
        "models",
        "checkpoints",
        "config",
        "backups"
    ]

    for dir_name in base_dirs:
        Path(dir_name).mkdir(exist_ok=True)
