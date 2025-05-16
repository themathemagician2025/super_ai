# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
import sys
import logging
import random
import json
import pickle
import asyncio
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import neat
import gym
from deap import base, creator, gp, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Project imports
from config import (
    PROJECT_CONFIG,
    NEAT_CONFIG,
    DEAP_CONFIG,
    BASE_DIR,
    SRC_DIR,
    LOG_DIR,
    DATA_DIR,
    RAW_DIR,
    MODELS_DIR
)