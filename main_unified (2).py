"""
Unified Main Script for Country Name Classifier
Supports both Traditional ML and BERT approaches
"""

import argparse
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data_ml(csv_path):
    """Prepare data using traditional approach"""
    from data_preparation import CountryDataPreparation
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    logger.info("="*70)
    logger.info("DATA PREPARATION - TRADITIONAL ML APPROACH")
    logger.info("="*70)
    
    prep = CountryDataPreparation(csv_path)
    df = prep.create_dataset()
    train, val, test = prep.split_and_save_data(df)
    
    logger.info("✓ Data preparation completed!")
    return True


def prepare_data_bert(csv_path, corporate_data_path=None):
    """Prepare data with enhanced financial awareness"""
    from bert_data_preparation import EnhancedDataPreparation
    from config_bert import RAW_DATA_DIR, PROCESSED_DATA_DIR, CORPORATE_DATA_DIR
    
    logger.info("="*70)
    logger.info("DATA PREPARATION - BERT APPROACH (Enhanced)")
    logger.info("="*70)
    
    prep = EnhancedDataPreparation(csv_path, corporate_data_path)
    df = prep.create_enhanced_dataset()
    train, val, test = prep.split_and_save_data(df, include_financial_test=True)
    
    logger.info("✓ Enhanced data preparation completed!")
    logger.info("✓ Financial instruments test set created!")
    return True


def train_ml():
    """Train traditional ML models"""
    from model_training import CountryClassifierTrainer
    
    logger.info("="*70)
    logger.info("TRAINING - TRADITIONAL ML")
    logger.info("="*70)
    
    trainer = CountryClassifierTrainer()
    model = trainer.train_all()
    
    logger.info("✓ Traditional ML training completed!")
    return True


def train_bert(model_name='bert-base-uncased'):
    """Train BERT model (auto-detects PyTorch or TensorFlow)"""
    logger.info("="*70)
    logger.info(f"TRAINING - BERT ({model_name})")
    logger.info("="*70)
    
    # Update config with chosen model
    from config_bert import BERT_CONFIG
    BERT_CONFIG['model_name'] = model_name
    
    # Auto-detection happens in bert_model_training.py
    from bert_model_training import main as train_bert_main
    train_bert_main()
    
    logger.info("✓ BERT training completed!")
    return True


def predict_ml():
    """Run predictions with traditional ML"""
    from inference import CountryClassifier
    
    logger.info("="*70)
    logger.info("PREDICTION - TRADITIONAL ML")
    logger.info("="*70)
    
    classifier = CountryClassifier()
    classifier.evaluate_examples()
    
    # Interactive mode
    classifier.interactive_mode()
    return True


def predict_bert(use_tensorflow=False):
    """Run predictions with BERT"""
    logger.info("="*70)
    logger.info("PREDICTION - BERT")
    if use_tensorflow:
        logger.info("Using TensorFlow backend")
    logger.info("="*70)
    
    if use_tensorflow:
        from bert_inference_tf import TFBERTCountryClassifierInference
        classifier = TFBERTCountryClassifierInference()
    else:
        from bert_inference import BERTCountryClassifierInference
        classifier = BERTCountryClassifierInference()
    
    # Run financial examples test
    classifier.evaluate_financial_examples()
    
    # Interactive mode
    classifier.interactive_mode()
    return True


def evaluate_financial_comparison():
    """Compare both approaches on financial instruments"""
    logger.info("="*70)
    logger.info("FINANCIAL INSTRUMENTS - MODEL COMPARISON")
    logger.info("="*70)
    
    from config_bert import PROCESSED_DATA_DIR
    import pandas as pd
    
    financial_test_file = PROCESSED_DATA_DIR / 'test_financial.csv'
    
    if not financial_test_file.exists():
        logger.error("Financial test set not found. Run data preparation first.")
        return False
    
    df_financial = pd.read_csv(financial_test_file)
    
    # Test Traditional ML
    logger.info("\n1. Testing Traditional ML Approach...")
    try:
        from inference import CountryClassifier
        ml_classifier = CountryClassifier()
        ml_results = ml_classifier.predict_batch(df_financial['text'].tolist())
        
        ml_correct = sum(1 for r, true_label in zip(ml_results, df_financial['label']) 
                        if r['is_country'] == (true_label == 1))
        ml_accuracy = ml_correct / len(df_financial)
        
        ml_false_positives = sum(1 for r, true_label in zip(ml_results, df_financial['label'])
                                if r['is_country'] and true_label == 0)
        
        logger.info(f"   Accuracy: {ml_accuracy:.2%}")
        logger.info(f"   False Positives: {ml_false_positives}")
        
    except Exception as e:
        logger.warning(f"   Traditional ML not available: {e}")
        ml_accuracy = 0
        ml_false_positives = 999
    
    # Test BERT
    logger.info("\n2. Testing BERT Approach...")
    try:
        from bert_inference import BERTCountryClassifierInference
        bert_classifier = BERTCountryClassifierInference()
        bert_results = bert_classifier.predict_batch(df_financial['text'].tolist())
        
        bert_correct = sum(1 for r, true_label in zip(bert_results, df_financial['label'])
                          if r['is_country'] == (true_label == 1))
        bert_accuracy = bert_correct / len(df_financial)
        
        bert_false_positives = sum(1 for r, true_label in zip(bert_results, df_financial['label'])
                                  if r['is_country'] and true_label == 0)
        
        logger.info(f"   Accuracy: {bert_accuracy:.2%}")
        logger.info(f"   False Positives: {bert_false_positives}")
        
    except Exception as e:
        logger.warning(f"   BERT not available: {e}")
        bert_accuracy = 0
        bert_false_positives = 999
    
    # Comparison
    logger.info("\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)
    
    print(f"\n{'Metric':<25} {'Traditional ML':<20} {'BERT':<20} {'Winner':<10}")
    print("-"*75)
    print(f"{'Accuracy':<25} {ml_accuracy:<20.2%} {bert_accuracy:<20.2%} "
          f"{'BERT' if bert_accuracy > ml_accuracy else 'ML':<10}")
    print(f"{'False Positives':<25} {ml_false_positives:<20} {bert_false_positives:<20} "
          f"{'BERT' if bert_false_positives < ml_false_positives else 'ML':<10}")
    
    if bert_false_positives == 0:
        logger.info("\n✅ BERT achieved ZERO false positives on financial instruments!")
    
    return True


def run_complete_pipeline(approach='bert', csv_path=None, corporate_data_path=None, 
                         model_name='bert-base-uncased'):
    """Run complete pipeline"""
    logger.info("="*70)
    logger.info(f"COMPLETE PIPELINE - {approach.upper()} APPROACH")
    logger.info("="*70)
    
    if approach == 'ml':
        # Traditional ML pipeline
        if not prepare_data_ml(csv_path):
            return False
        if not train_ml():
            return False
        if not predict_ml():
            return False
    
    elif approach == 'bert':
        # BERT pipeline
        if not prepare_data_bert(csv_path, corporate_data_path):
            return False
        if not train_bert(model_name):
            return False
        if not predict_bert():
            return False
    
    elif approach == 'both':
        # Train both and compare
        logger.info("\nTraining BOTH approaches for comparison...")
        
        # Prepare enhanced data
        if not prepare_data_bert(csv_path, corporate_data_path):
            return False
        
        # Train ML
        logger.info("\n--- Training Traditional ML ---")
        if not train_ml():
            logger.warning("Traditional ML training failed")
        
        # Train BERT
        logger.info("\n--- Training BERT ---")
        if not train_bert(model_name):
            logger.warning("BERT training failed")
        
        # Compare
        evaluate_financial_comparison()
    
    else:
        logger.error(f"Unknown approach: {approach}")
        return False
    
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Country Name Classifier - Unified Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick baseline with traditional ML
  python main_unified.py --mode all --approach ml --csv data/raw/country_aliases.csv
  
  # Production-ready BERT model
  python main_unified.py --mode all --approach bert --csv data/raw/country_aliases.csv
  
  # BERT with your corporate data
  python main_unified.py --mode all --approach bert \\
      --csv data/raw/country_aliases.csv \\
      --corporate-data data/corporate/corporate_data.csv
  
  # Use FinBERT (best for financial data)
  python main_unified.py --mode all --approach bert \\
      --model finbert \\
      --csv data/raw/country_aliases.csv \\
      --corporate-data data/corporate/corporate_data.csv
  
  # Train both and compare
  python main_unified.py --mode all --approach both \\
      --csv data/raw/country_aliases.csv \\
      --corporate-data data/corporate/corporate_data.csv
  
  # Just evaluate on financial instruments
  python main_unified.py --mode compare
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['prepare', 'train', 'predict', 'all', 'compare'],
        required=True,
        help='Execution mode'
    )
    
    parser.add_argument(
        '--approach',
        type=str,
        choices=['ml', 'bert', 'both'],
        default='bert',
        help='Model approach (ml=Traditional ML, bert=BERT, both=Compare both)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        choices=['bert-base-uncased', 'distilbert', 'roberta', 'finbert'],
        help='BERT model variant'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to country aliases CSV'
    )
    
    parser.add_argument(
        '--corporate-data',
        type=str,
        default=None,
        help='Path to your corporate_data CSV (for enhanced training)'
    )
    
    parser.add_argument(
        '--use-tensorflow',
        action='store_true',
        help='Use TensorFlow instead of PyTorch (firewall-friendly, standard pip)'
    )
    
    args = parser.parse_args()
    
    # Map model names
    model_mapping = {
        'bert-base-uncased': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'finbert': 'ProsusAI/finbert'
    }
    model_name = model_mapping.get(args.model, args.model)
    
    # Validate paths
    if args.mode in ['prepare', 'all'] and args.csv is None:
        from config_bert import RAW_DATA_DIR
        args.csv = RAW_DATA_DIR / "country_aliases.csv"
        
        if not Path(args.csv).exists():
            logger.error(f"Country aliases CSV not found at {args.csv}")
            logger.error("Please download from: https://www.kaggle.com/datasets/wbdill/country-aliaseslist-of-alternative-country-names")
            sys.exit(1)
    
    # Execute
    try:
        if args.mode == 'compare':
            success = evaluate_financial_comparison()
        
        elif args.mode == 'all':
            success = run_complete_pipeline(
                approach=args.approach,
                csv_path=args.csv,
                corporate_data_path=args.corporate_data,
                model_name=model_name
            )
        
        elif args.mode == 'prepare':
            if args.approach == 'bert':
                success = prepare_data_bert(args.csv, args.corporate_data)
            else:
                success = prepare_data_ml(args.csv)
        
        elif args.mode == 'train':
            if args.approach == 'bert':
                success = train_bert(model_name, use_tensorflow=args.use_tensorflow)
            else:
                success = train_ml()
        
        elif args.mode == 'predict':
            if args.approach == 'bert':
                success = predict_bert(use_tensorflow=args.use_tensorflow)
            else:
                success = predict_ml()
        
        if success:
            logger.info("\n✅ Execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Execution failed!")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()