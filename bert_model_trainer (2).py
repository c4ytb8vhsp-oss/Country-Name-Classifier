import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import logging
from config_bert import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CountryClassificationDataset(Dataset):
    """PyTorch Dataset for country classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTCountryClassifier:
    """BERT-based country name classifier"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
    def create_data_loader(self, texts, labels, batch_size=16, shuffle=True):
        """Create PyTorch DataLoader"""
        dataset = CountryClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=BERT_CONFIG['max_length']
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    
    def train_epoch(self, data_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(data_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        
        predictions = []
        true_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def train(self, train_texts, train_labels, val_texts, val_labels,
              epochs=4, batch_size=16, learning_rate=2e-5):
        """Complete training pipeline"""
        
        logger.info("Creating data loaders...")
        train_loader = self.create_data_loader(
            train_texts, train_labels, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = self.create_data_loader(
            val_texts, val_labels,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=BERT_CONFIG['weight_decay']
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=BERT_CONFIG['warmup_steps'],
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Total steps: {total_steps}")
        
        best_val_f1 = 0
        history = []
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model(BERT_MODELS_DIR / 'best_model')
                logger.info(f"✓ Saved best model (F1: {best_val_f1:.4f})")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall']
            })
        
        logger.info(f"\nTraining completed! Best F1: {best_val_f1:.4f}")
        
        return history
    
    def predict(self, texts, batch_size=16):
        """Make predictions on new texts"""
        self.model.eval()
        
        # Create dummy labels for DataLoader
        dummy_labels = [0] * len(texts)
        data_loader = self.create_data_loader(
            texts, dummy_labels,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return all_predictions, all_probabilities
    
    def save_model(self, path):
        """Save model and tokenizer"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load saved model"""
        path = Path(path)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {path}")


def evaluate_on_financial_test_set(classifier, test_file):
    """
    Evaluate specifically on financial instruments
    This is critical to ensure low false positive rate
    """
    logger.info("\n" + "="*70)
    logger.info("FINANCIAL INSTRUMENTS TEST SET EVALUATION")
    logger.info("="*70)
    
    # Load financial test set
    df_financial = pd.read_csv(test_file)
    
    # Make predictions
    predictions, probabilities = classifier.predict(df_financial['text'].tolist())
    
    # Add predictions to dataframe
    df_financial['predicted'] = predictions
    df_financial['confidence'] = [probs[pred] for pred, probs in zip(predictions, probabilities)]
    
    # Calculate metrics
    accuracy = accuracy_score(df_financial['label'], predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_financial['label'], predictions, average='binary'
    )
    
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f} (Critical for financial data!)")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    
    # Analyze false positives (financial instruments classified as countries)
    false_positives = df_financial[(df_financial['label'] == 0) & (df_financial['predicted'] == 1)]
    
    logger.info(f"\n🚨 FALSE POSITIVES: {len(false_positives)} (Should be 0!)")
    if len(false_positives) > 0:
        logger.warning("\nFinancial instruments incorrectly classified as countries:")
        for idx, row in false_positives.iterrows():
            logger.warning(f"  ❌ '{row['text']}' (confidence: {row['confidence']:.2%})")
    
    # Analyze false negatives (countries classified as non-countries)
    false_negatives = df_financial[(df_financial['label'] == 1) & (df_financial['predicted'] == 0)]
    
    logger.info(f"\n⚠️  FALSE NEGATIVES: {len(false_negatives)}")
    if len(false_negatives) > 0:
        logger.warning("\nCountries incorrectly classified as non-countries:")
        for idx, row in false_negatives.iterrows():
            logger.warning(f"  ❌ '{row['text']}' (confidence: {row['confidence']:.2%})")
    
    # Show correct classifications
    logger.info("\n✅ Sample CORRECT Classifications:")
    
    # Countries correctly identified
    correct_countries = df_financial[(df_financial['label'] == 1) & (df_financial['predicted'] == 1)]
    logger.info(f"\nCountries correctly identified ({len(correct_countries)}):")
    for idx, row in correct_countries.head(5).iterrows():
        logger.info(f"  ✓ '{row['text']}' → Country (confidence: {row['confidence']:.2%})")
    
    # Financial instruments correctly rejected
    correct_non_countries = df_financial[(df_financial['label'] == 0) & (df_financial['predicted'] == 0)]
    logger.info(f"\nFinancial instruments correctly rejected ({len(correct_non_countries)}):")
    for idx, row in correct_non_countries.head(10).iterrows():
        logger.info(f"  ✓ '{row['text']}' → NOT Country (confidence: {row['confidence']:.2%})")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'df_results': df_financial
    }


def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("BERT COUNTRY CLASSIFIER - TRAINING PIPELINE")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading training data...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    val_df = pd.read_csv(PROCESSED_DATA_DIR / 'val.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Show class distribution
    logger.info("\nClass distribution:")
    logger.info(f"  Train - Country: {(train_df['label']==1).sum()}, "
               f"Non-Country: {(train_df['label']==0).sum()}")
    
    # Initialize classifier
    model_name = BERT_CONFIG['model_name']
    logger.info(f"\nInitializing {model_name}...")
    
    classifier = BERTCountryClassifier(model_name=model_name)
    
    # Train
    history = classifier.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        val_texts=val_df['text'].tolist(),
        val_labels=val_df['label'].tolist(),
        epochs=BERT_CONFIG['num_epochs'],
        batch_size=BERT_CONFIG['batch_size'],
        learning_rate=BERT_CONFIG['learning_rate']
    )
    
    # Load best model
    logger.info("\nLoading best model for evaluation...")
    classifier.load_model(BERT_MODELS_DIR / 'best_model')
    
    # Evaluate on standard test set
    logger.info("\n" + "="*70)
    logger.info("STANDARD TEST SET EVALUATION")
    logger.info("="*70)
    
    test_loader = classifier.create_data_loader(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        batch_size=BERT_CONFIG['batch_size'],
        shuffle=False
    )
    
    test_metrics = classifier.evaluate(test_loader)
    
    logger.info(f"\nTest Set Results:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    logger.info("\nClassification Report:")
    print(classification_report(
        test_metrics['true_labels'],
        test_metrics['predictions'],
        target_names=['Not Country', 'Country']
    ))
    
    # Evaluate on financial test set
    financial_test_file = PROCESSED_DATA_DIR / 'test_financial.csv'
    if financial_test_file.exists():
        financial_metrics = evaluate_on_financial_test_set(
            classifier, 
            financial_test_file
        )
        
        # Save results
        financial_metrics['df_results'].to_csv(
            PROCESSED_DATA_DIR / 'financial_test_results.csv',
            index=False
        )
    else:
        logger.warning(f"Financial test set not found at {financial_test_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*70)
    
    return classifier, history


if __name__ == "__main__":
    classifier, history = main()