"""
Main execution script for Country Name Classifier

Usage:
    python main.py --mode prepare    # Prepare dataset
    python main.py --mode train      # Train models
    python main.py --mode predict    # Run predictions
    python main.py --mode all        # Run complete pipeline
"""

import argparse
import sys
from pathlib import Path

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR
from src.data_preparation import CountryDataPreparation
from src.model_training import CountryClassifierTrainer
from src.inference import CountryClassifier


def prepare_data(csv_path, corporate_data_path=None):
    """Prepare and split dataset"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    prep = CountryDataPreparation(csv_path, corporate_data_path)
    df = prep.create_dataset()
    train, val, test = prep.split_and_save_data(df)
    
    print("\n✓ Data preparation completed successfully!")
    return True


def train_model():
    """Train the classification model"""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    # Check if processed data exists
    train_file = PROCESSED_DATA_DIR / 'train.csv'
    if not train_file.exists():
        print("Error: Training data not found. Please run data preparation first.")
        return False
    
    trainer = CountryClassifierTrainer()
    model = trainer.train_all()
    
    print("\n✓ Model training completed successfully!")
    return True


def run_predictions():
    """Run predictions using trained model"""
    print("\n" + "="*70)
    print("STEP 3: PREDICTION")
    print("="*70)
    
    # Check if model exists
    model_file = SAVED_MODELS_DIR / 'country_classifier.pkl'
    if not model_file.exists():
        print("Error: Trained model not found. Please run training first.")
        return False
    
    classifier = CountryClassifier()
    classifier.evaluate_examples()
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter names to classify (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = classifier.predict(user_input)
            confidence_str = f" ({result['confidence']:.2%})" if result['confidence'] else ""
            
            if result['is_country']:
                print(f"✓ Country{confidence_str}")
            else:
                print(f"✗ Not a Country{confidence_str}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    return True


def run_complete_pipeline(csv_path, corporate_data_path=None):
    """Run the complete pipeline"""
    print("\n" + "="*70)
    print("RUNNING COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Prepare data
    if not prepare_data(csv_path, corporate_data_path):
        print("\n✗ Pipeline failed at data preparation step")
        return False
    
    # Step 2: Train model
    if not train_model():
        print("\n✗ Pipeline failed at training step")
        return False
    
    # Step 3: Run predictions
    if not run_predictions():
        print("\n✗ Pipeline failed at prediction step")
        return False
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Country Name Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode all --csv data/raw/country_aliases.csv
  python main.py --mode prepare --csv data/raw/country_aliases.csv
  python main.py --mode train
  python main.py --mode predict
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['prepare', 'train', 'predict', 'all'],
        required=True,
        help='Execution mode'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to country aliases CSV file (required for prepare/all modes)'
    )
    
    parser.add_argument(
        '--corporate-data',
        type=str,
        default=None,
        help='Path to corporate data CSV/Excel (entities with country names that are NOT countries)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['prepare', 'all'] and args.csv is None:
        parser.error("--csv is required for 'prepare' and 'all' modes")
    
    # Set default CSV path if not provided
    if args.csv is None:
        args.csv = RAW_DATA_DIR / "country_aliases.csv"
    
    csv_path = Path(args.csv)
    corporate_path = Path(args.corporate_data) if args.corporate_data else None
    
    # Check if CSV exists for prepare/all modes
    if args.mode in ['prepare', 'all'] and not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Please download the dataset from:")
        print(f"https://www.kaggle.com/datasets/wbdill/country-aliaseslist-of-alternative-country-names")
        sys.exit(1)
    
    # Execute based on mode
    try:
        if args.mode == 'prepare':
            success = prepare_data(csv_path, corporate_path)
        elif args.mode == 'train':
            success = train_model()
        elif args.mode == 'predict':
            success = run_predictions()
        elif args.mode == 'all':
            success = run_complete_pipeline(csv_path, corporate_path)
        
        if success:
            print("\n✓ Execution completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Execution failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()