import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.15,
    'val_size': 0.15,
    'random_state': 42,
    'max_features': 5000,
    'ngram_range': (1, 3),
    'min_df': 2,
    'max_df': 0.95
}

# Training parameters
TRAINING_CONFIG = {
    'models': ['logistic_regression', 'random_forest', 'gradient_boosting'],
    'cv_folds': 5,
    'n_iter': 20,  # for random search
}

# Non-country patterns (will be used for negative examples)
NON_COUNTRY_PATTERNS = [
    "bank of", "university of", "ministry of", "department of",
    "embassy of", "consulate of", "airport", "airlines",
    "hotel", "restaurant", "museum", "library",
    "company", "corporation", "inc", "ltd", "llc",
    "foundation", "institute", "association", "society",
    "church", "mosque", "temple", "cathedral"
]