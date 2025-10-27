import joblib
import numpy as np
from pathlib import Path
from config import SAVED_MODELS_DIR
from src.feature_engineering import FeatureEngineering

class CountryClassifier:
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize the classifier with saved model and vectorizer"""
        if model_path is None:
            model_path = SAVED_MODELS_DIR / 'country_classifier.pkl'
        if vectorizer_path is None:
            vectorizer_path = SAVED_MODELS_DIR / 'vectorizer.pkl'
        
        self.model = joblib.load(model_path)
        self.feature_engineering = FeatureEngineering()
        self.feature_engineering.load_vectorizer(vectorizer_path)
        
        print("Country classifier loaded successfully!")
    
    def predict(self, text):
        """
        Predict whether a text is a country name or not
        
        Args:
            text (str): Input text to classify
        
        Returns:
            dict: Dictionary containing prediction and probability
        """
        # Prepare features
        features = self.feature_engineering.combine_features([text], fit=False)
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        
        # Get probability scores
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        else:
            confidence = None
        
        result = {
            'text': text,
            'is_country': bool(prediction),
            'label': 'Country' if prediction == 1 else 'Not Country',
            'confidence': float(confidence) if confidence is not None else None
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict for multiple texts at once
        
        Args:
            texts (list): List of texts to classify
        
        Returns:
            list: List of prediction dictionaries
        """
        # Prepare features
        features = self.feature_engineering.combine_features(texts, fit=False)
        
        # Get predictions
        predictions = self.model.predict(features)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        else:
            probabilities = None
        
        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            confidence = probabilities[i][pred] if probabilities is not None else None
            
            result = {
                'text': text,
                'is_country': bool(pred),
                'label': 'Country' if pred == 1 else 'Not Country',
                'confidence': float(confidence) if confidence is not None else None
            }
            results.append(result)
        
        return results
    
    def evaluate_examples(self):
        """Test the classifier with example inputs"""
        test_cases = [
            "South Korea",
            "Republic of South Korea",
            "State of South Korea",
            "Bank of South Korea",
            "South Korea Bank",
            "University of South Korea",
            "United States",
            "United States of America",
            "USA",
            "Bank of America",
            "United Airlines",
            "Germany",
            "Federal Republic of Germany",
            "Deutsche Bank",
            "France",
            "Republic of France",
            "Air France",
            "China",
            "People's Republic of China",
            "Bank of China",
            "India",
            "Republic of India",
            "State Bank of India",
            "Japan",
            "Embassy of Japan",
            "Brazil",
            "Federative Republic of Brazil",
            "Central Bank of Brazil",
            "United Kingdom",
            "The United Kingdom",
            "British Airways",
            "Microsoft",
            "Apple Inc",
            "Harvard University",
        ]
        
        print("\n" + "="*70)
        print("COUNTRY CLASSIFIER - TEST EXAMPLES")
        print("="*70)
        
        results = self.predict_batch(test_cases)
        
        for result in results:
            confidence_str = f" (confidence: {result['confidence']:.2%})" if result['confidence'] else ""
            print(f"\nInput: '{result['text']}'")
            print(f"Prediction: {result['label']}{confidence_str}")
        
        return results


def main():
    """Main function to demonstrate usage"""
    # Initialize classifier
    classifier = CountryClassifier()
    
    # Run test examples
    classifier.evaluate_examples()
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE - Enter 'quit' to exit")
    print("="*70)
    
    while True:
        user_input = input("\nEnter a name to classify: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = classifier.predict(user_input)
        confidence_str = f" (confidence: {result['confidence']:.2%})" if result['confidence'] else ""
        print(f"Prediction: {result['label']}{confidence_str}")


if __name__ == "__main__":
    main()