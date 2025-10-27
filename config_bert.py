import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORPORATE_DATA_DIR = DATA_DIR / "corporate"  # NEW: For your internal data
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
BERT_MODELS_DIR = MODELS_DIR / "bert_models"  # NEW: For BERT checkpoints

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CORPORATE_DATA_DIR, 
                  SAVED_MODELS_DIR, BERT_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# BERT Model Configuration
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',  # Or 'distilbert-base-uncased' for faster training
    'max_length': 128,  # Maximum sequence length
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    'fp16': True,  # Mixed precision training (faster on GPU)
    'device': 'cuda' if os.environ.get('USE_GPU') else 'cpu',
}

# Alternative models to try
ALTERNATIVE_MODELS = {
    'distilbert': 'distilbert-base-uncased',  # Faster, 40% smaller
    'roberta': 'roberta-base',  # Better performance
    'finbert': 'ProsusAI/finbert',  # Financial domain-specific
    'xlm-roberta': 'xlm-roberta-base',  # Multilingual
}

# Traditional ML fallback config (if BERT too slow)
ML_CONFIG = {
    'test_size': 0.15,
    'val_size': 0.15,
    'random_state': 42,
    'max_features': 10000,
    'ngram_range': (1, 4),  # Increased for better context
    'min_df': 2,
    'max_df': 0.95
}

# Financial instrument patterns (for enhanced negative examples)
FINANCIAL_INSTRUMENTS = [
    # Bonds
    "bond", "bonds", "treasury", "treasuries", "sovereign bond",
    "government bond", "municipal bond", "corporate bond",
    # Deposits
    "deposit", "deposits", "certificate of deposit", "cd", "time deposit",
    # Securities
    "security", "securities", "equity", "equities", "stock", "stocks",
    "note", "notes", "bill", "bills",
    # Derivatives
    "futures", "future", "option", "options", "swap", "swaps",
    "forward", "forwards", "derivative", "derivatives",
    # Currencies
    "currency", "forex", "fx", "exchange rate",
    # Debt instruments
    "debt", "loan", "credit", "debenture", "obligation",
    # Investment products
    "fund", "etf", "mutual fund", "index fund", "reit",
    # Others
    "gilt", "gilts", "bund", "bunds", "tbill", "tbills",
]

# Country-related financial patterns (SHOULD BE NEGATIVE)
COUNTRY_FINANCIAL_PATTERNS = [
    "{country} bond",
    "{country} bonds",
    "{country} treasury",
    "{country} treasuries",
    "{country} government bond",
    "{country} sovereign bond",
    "{country} deposit",
    "{country} deposits",
    "{country} note",
    "{country} notes",
    "{country} bill",
    "{country} bills",
    "{country} debt",
    "{country} securities",
    "{country} equity",
    "{country} stock",
    "government of {country} bond",
    "{country} sovereign debt",
    "{country} forex",
    "{country} currency",
]

# Evaluation metrics focus
EVALUATION_CONFIG = {
    'focus_on_financial_false_positives': True,
    'min_confidence_threshold': 0.7,
    'require_high_precision': True,  # Critical for financial applications
}