"""
BERT Inference - Auto-detects TensorFlow or PyTorch
Works with either backend automatically!
"""

# Try to import PyTorch, fall back to TensorFlow if not available
try:
    import torch
    BACKEND = 'pytorch'
    print("✓ Using PyTorch backend")
except ImportError:
    try:
        import tensorflow as tf
        BACKEND = 'tensorflow'
        print("✓ Using TensorFlow backend (PyTorch not available)")
    except ImportError:
        raise ImportError("Neither PyTorch nor TensorFlow is installed. Please install one:\n"
                         "  pip install tensorflow==2.15.0  (recommended for firewalls)\n"
                         "  OR\n"
                         "  pip install torch")

# Import appropriate classes based on backend
if BACKEND == 'pytorch':
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
else:  # tensorflow
    import tensorflow as tf
    from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

import numpy as np
from pathlib import Path
from config_bert import BERT_MODELS_DIR, BERT_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTCountryClassifierInference:
    """BERT classifier for inference - works with PyTorch OR TensorFlow"""
    
    def __init__(self, model_path=None):
        """
        Initialize classifier for inference
        
        Args:
            model_path: Path to saved BERT model (default: best_model)
        """
        if model_path is None:
            model_path = BERT_MODELS_DIR / 'best_model'
        
        self.model_path = Path(model_path)
        self.backend = BACKEND
        
        if BACKEND == 'pytorch':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading model from {self.model_path} ({BACKEND} backend)")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if BACKEND == 'pytorch':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                map_location=self.device
            )
            self.model.to(self.device)
            self.model.eval()
        else:  # tensorflow
            self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        logger.info("Model loaded successfully!")
    
    def predict(self, text, return_confidence=True):
        """
        Predict whether text is a country name
        
        Args:
            text: Input text to classify
            return_confidence: Whether to return confidence scores
        
        Returns:
            dict with prediction results
        """
        if BACKEND == 'pytorch':
            return self._predict_pytorch(text, return_confidence)
        else:
            return self._predict_tensorflow(text, return_confidence)
    
    def _predict_pytorch(self, text, return_confidence):
        """PyTorch prediction"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=BERT_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probs[0][prediction].item()
        
        result = {
            'text': text,
            'is_country': bool(prediction == 1),
            'label': 'Country' if prediction == 1 else 'Not Country',
            'confidence': float(confidence) if return_confidence else None,
            'probabilities': {
                'not_country': float(probs[0][0].item()),
                'country': float(probs[0][1].item())
            } if return_confidence else None
        }
        
        return result
    
    def _predict_tensorflow(self, text, return_confidence):
        """TensorFlow prediction"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=BERT_CONFIG['max_length'],
            return_tensors='tf'
        )
        
        outputs = self.model(encoding, training=False)
        logits = outputs.logits
        
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        prediction = int(tf.argmax(logits, axis=1).numpy()[0])
        confidence = float(probs[prediction])
        
        result = {
            'text': text,
            'is_country': bool(prediction == 1),
            'label': 'Country' if prediction == 1 else 'Not Country',
            'confidence': confidence if return_confidence else None,
            'probabilities': {
                'not_country': float(probs[0]),
                'country': float(probs[1])
            } if return_confidence else None
        }
        
        return result
    
    def predict_batch(self, texts, batch_size=16):
        """
        Predict for multiple texts efficiently
        
        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        if BACKEND == 'pytorch':
            return self._predict_batch_pytorch(texts, batch_size)
        else:
            return self._predict_batch_tensorflow(texts, batch_size)
    
    def _predict_batch_pytorch(self, texts, batch_size):
        """PyTorch batch prediction"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=BERT_CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                for j, text in enumerate(batch_texts):
                    pred = predictions[j].item()
                    confidence = probs[j][pred].item()
                    
                    results.append({
                        'text': text,
                        'is_country': bool(pred == 1),
                        'label': 'Country' if pred == 1 else 'Not Country',
                        'confidence': float(confidence),
                        'probabilities': {
                            'not_country': float(probs[j][0].item()),
                            'country': float(probs[j][1].item())
                        }
                    })
        
        return results
    
    def _predict_batch_tensorflow(self, texts, batch_size):
        """TensorFlow batch prediction"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=BERT_CONFIG['max_length'],
                return_tensors='tf'
            )
            
            outputs = self.model(encodings, training=False)
            logits = outputs.logits
            
            probs = tf.nn.softmax(logits, axis=1).numpy()
            predictions = tf.argmax(logits, axis=1).numpy()
            
            for j, text in enumerate(batch_texts):
                pred = int(predictions[j])
                confidence = float(probs[j][pred])
                
                results.append({
                    'text': text,
                    'is_country': bool(pred == 1),
                    'label': 'Country' if pred == 1 else 'Not Country',
                    'confidence': confidence,
                    'probabilities': {
                        'not_country': float(probs[j][0]),
                        'country': float(probs[j][1])
                    }
                })
        
        return results
    
    def evaluate_financial_examples(self):
        """Test the classifier with example inputs"""
        test_cases = [
            # Financial instruments (should be NOT Country)
            ("US Treasury Bond", False),
            ("US Treasury 10Y", False),
            ("Korea Government Bond", False),
            ("Korean Treasury", False),
            ("South Korea Sovereign Bond", False),
            ("Japan Government Bond", False),
            ("German Bund", False),
            ("UK Gilt", False),
            ("Korea Deposit", False),
            ("Korean Won Deposit", False),
            ("US Dollar Deposit", False),
            ("Brazil Sovereign Debt", False),
            ("Mexico Sovereign Bond", False),
            ("Bank of Korea", False),
            ("Bank of America", False),
            ("University of Germany", False),
            ("Japan Airlines", False),
            
            # Actual countries (should be Country)
            ("United States", True),
            ("United States of America", True),
            ("Korea", True),
            ("South Korea", True),
            ("Republic of Korea", True),
            ("Japan", True),
            ("Germany", True),
            ("Federal Republic of Germany", True),
            ("United Kingdom", True),
            ("Brazil", True),
            ("Mexico", True),
        ]
        
        print("\n" + "="*70)
        print(f"BERT CLASSIFIER ({BACKEND.upper()}) - FINANCIAL INSTRUMENTS TEST")
        print("="*70)
        
        results = self.predict_batch([tc[0] for tc in test_cases])
        
        correct = 0
        false_positives = []
        false_negatives = []
        
        for (text, expected), result in zip(test_cases, results):
            is_correct = result['is_country'] == expected
            correct += int(is_correct)
            
            status = "✓" if is_correct else "✗"
            expected_label = "Country" if expected else "NOT Country"
            
            print(f"\n{status} '{text}'")
            print(f"   Expected: {expected_label} | Got: {result['label']} "
                  f"(confidence: {result['confidence']:.2%})")
            
            if not is_correct:
                if result['is_country'] and not expected:
                    false_positives.append(text)
                elif not result['is_country'] and expected:
                    false_negatives.append(text)
        
        accuracy = correct / len(test_cases)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
        
        if false_positives:
            print(f"\n🚨 FALSE POSITIVES ({len(false_positives)}):")
            for fp in false_positives:
                print(f"   - {fp}")
        else:
            print("\n✅ NO FALSE POSITIVES!")
        
        if false_negatives:
            print(f"\n⚠️  FALSE NEGATIVES ({len(false_negatives)}):")
            for fn in false_negatives:
                print(f"   - {fn}")
        
        return {
            'accuracy': accuracy,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n" + "="*70)
        print(f"INTERACTIVE MODE - {BACKEND.upper()} BACKEND")
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
                
                result = self.predict(user_input)
                
                if result['is_country']:
                    print(f"✓ COUNTRY (confidence: {result['confidence']:.2%})")
                else:
                    print(f"✗ NOT A COUNTRY (confidence: {result['confidence']:.2%})")
                
                print(f"   Probabilities: Country={result['probabilities']['country']:.2%}, "
                      f"Not Country={result['probabilities']['not_country']:.2%}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function for testing"""
    # Initialize classifier
    classifier = BERTCountryClassifierInference()
    
    # Run financial examples test
    results = classifier.evaluate_financial_examples()
    
    # Interactive mode
    classifier.interactive_mode()


if __name__ == "__main__":
    main()import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path
from config_bert import BERT_MODELS_DIR, BERT_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTCountryClassifierInference:
    """Production-ready BERT classifier for country name detection"""
    
    def __init__(self, model_path=None):
        """
        Initialize classifier for inference
        
        Args:
            model_path: Path to saved BERT model (default: best_model)
        """
        if model_path is None:
            model_path = BERT_MODELS_DIR / 'best_model'
        
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def predict(self, text, return_confidence=True):
        """
        Predict whether text is a country name
        
        Args:
            text: Input text to classify
            return_confidence: Whether to return confidence scores
        
        Returns:
            dict with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=BERT_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probs[0][prediction].item()
        
        result = {
            'text': text,
            'is_country': bool(prediction == 1),
            'label': 'Country' if prediction == 1 else 'Not Country',
            'confidence': confidence if return_confidence else None,
            'probabilities': {
                'not_country': probs[0][0].item(),
                'country': probs[0][1].item()
            } if return_confidence else None
        }
        
        return result
    
    def predict_batch(self, texts, batch_size=16):
        """
        Predict for multiple texts efficiently
        
        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=BERT_CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Process results
                for j, text in enumerate(batch_texts):
                    pred = predictions[j].item()
                    confidence = probs[j][pred].item()
                    
                    results.append({
                        'text': text,
                        'is_country': bool(pred == 1),
                        'label': 'Country' if pred == 1 else 'Not Country',
                        'confidence': confidence,
                        'probabilities': {
                            'not_country': probs[j][0].item(),
                            'country': probs[j][1].item()
                        }
                    })
        
        return results
    
    def evaluate_financial_examples(self):
        """Test with financial instrument examples"""
        
        print("\n" + "="*70)
        print("BERT COUNTRY CLASSIFIER - FINANCIAL INSTRUMENTS TEST")
        print("="*70)
        
        test_cases = [
            # Should be NOT Country (0)
            ("US Treasury Bond", False),
            ("US Treasury 10Y", False),
            ("Korea Government Bond", False),
            ("Korean Treasury", False),
            ("South Korea Sovereign Bond", False),
            ("Japan Government Bond", False),
            ("German Bund", False),
            ("UK Gilt", False),
            ("Korea Deposit", False),
            ("Korean Won Deposit", False),
            ("US Dollar Deposit", False),
            ("Brazil Sovereign Debt", False),
            ("Mexico Sovereign Bond", False),
            ("Bank of Korea", False),
            ("Bank of America", False),
            ("University of Germany", False),
            ("Japan Airlines", False),
            
            # Should be Country (1)
            ("United States", True),
            ("United States of America", True),
            ("Korea", True),
            ("South Korea", True),
            ("Republic of Korea", True),
            ("Japan", True),
            ("Germany", True),
            ("Federal Republic of Germany", True),
            ("United Kingdom", True),
            ("Brazil", True),
            ("Mexico", True),
        ]
        
        print("\n" + "-"*70)
        print("Test Results:")
        print("-"*70)
        
        correct = 0
        total = len(test_cases)
        false_positives = []
        false_negatives = []
        
        for text, expected_is_country in test_cases:
            result = self.predict(text)
            is_correct = result['is_country'] == expected_is_country
            correct += int(is_correct)
            
            status = "✓" if is_correct else "✗"
            expected = "Country" if expected_is_country else "NOT Country"
            
            print(f"{status} '{text}'")
            print(f"   Expected: {expected} | Got: {result['label']} "
                  f"(confidence: {result['confidence']:.2%})")
            
            if not is_correct:
                if result['is_country'] and not expected_is_country:
                    false_positives.append(text)
                elif not result['is_country'] and expected_is_country:
                    false_negatives.append(text)
        
        accuracy = correct / total
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        if false_positives:
            print(f"\n🚨 FALSE POSITIVES ({len(false_positives)}):")
            print("   (Financial instruments classified as countries)")
            for fp in false_positives:
                print(f"   - {fp}")
        else:
            print("\n✅ NO FALSE POSITIVES! (Perfect for financial data)")
        
        if false_negatives:
            print(f"\n⚠️  FALSE NEGATIVES ({len(false_negatives)}):")
            print("   (Countries classified as non-countries)")
            for fn in false_negatives:
                print(f"   - {fn}")
        
        return {
            'accuracy': accuracy,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n" + "="*70)
        print("INTERACTIVE MODE - Enter text to classify")
        print("Type 'quit' to exit")
        print("="*70)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                result = self.predict(user_input)
                
                if result['is_country']:
                    print(f"✓ COUNTRY (confidence: {result['confidence']:.2%})")
                else:
                    print(f"✗ NOT A COUNTRY (confidence: {result['confidence']:.2%})")
                
                print(f"   Probabilities: Country={result['probabilities']['country']:.2%}, "
                      f"Not Country={result['probabilities']['not_country']:.2%}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function for testing"""
    # Initialize classifier
    classifier = BERTCountryClassifierInference()
    
    # Run financial examples test
    results = classifier.evaluate_financial_examples()
    
    # Interactive mode
    classifier.interactive_mode()


if __name__ == "__main__":
    main()