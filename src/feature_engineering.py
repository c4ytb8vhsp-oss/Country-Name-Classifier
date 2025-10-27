import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from config import *

class FeatureEngineering:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
    def extract_text_features(self, text):
        """Extract hand-crafted features from text"""
        features = {}
        
        text_lower = text.lower()
        
        # Length features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Pattern matching features
        features['has_bank'] = int('bank' in text_lower)
        features['has_university'] = int('university' in text_lower or 'college' in text_lower)
        features['has_hotel'] = int('hotel' in text_lower)
        features['has_airport'] = int('airport' in text_lower)
        features['has_airlines'] = int('airline' in text_lower)
        features['has_ministry'] = int('ministry' in text_lower)
        features['has_department'] = int('department' in text_lower)
        features['has_embassy'] = int('embassy' in text_lower or 'consulate' in text_lower)
        features['has_company'] = int(any(word in text_lower for word in ['inc', 'llc', 'ltd', 'corp', 'company']))
        features['has_foundation'] = int('foundation' in text_lower or 'institute' in text_lower)
        
        # Country-related keywords
        features['has_republic'] = int('republic' in text_lower)
        features['has_kingdom'] = int('kingdom' in text_lower)
        features['has_state'] = int('state' in text_lower)
        features['has_federation'] = int('federation' in text_lower)
        features['has_democratic'] = int('democratic' in text_lower)
        features['has_peoples'] = int("people's" in text_lower or "peoples" in text_lower)
        features['has_commonwealth'] = int('commonwealth' in text_lower)
        features['has_government'] = int('government' in text_lower)
        
        # Capitalization features
        features['starts_with_capital'] = int(text[0].isupper() if text else 0)
        features['capital_word_ratio'] = sum(1 for word in text.split() if word and word[0].isupper()) / max(len(text.split()), 1)
        
        # Special character features
        features['has_of'] = int(' of ' in text_lower)
        features['has_the'] = int(text_lower.startswith('the '))
        
        # Preposition patterns (common in organization names)
        features['preposition_count'] = len(re.findall(r'\b(of|in|at|by|for|with)\b', text_lower))
        
        return features
    
    def create_feature_matrix(self, texts):
        """Create feature matrix from list of texts"""
        feature_list = [self.extract_text_features(text) for text in texts]
        df_features = pd.DataFrame(feature_list)
        return df_features
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer on training texts"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=MODEL_CONFIG['max_features'],
            ngram_range=MODEL_CONFIG['ngram_range'],
            min_df=MODEL_CONFIG['min_df'],
            max_df=MODEL_CONFIG['max_df'],
            lowercase=True,
            analyzer='char_wb'  # Character n-grams can capture name patterns better
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_features
    
    def transform_tfidf(self, texts):
        """Transform texts using fitted TF-IDF vectorizer"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted yet!")
        
        return self.tfidf_vectorizer.transform(texts)
    
    def combine_features(self, texts, fit=False):
        """Combine TF-IDF and hand-crafted features"""
        # Get TF-IDF features
        if fit:
            tfidf_features = self.fit_tfidf(texts)
        else:
            tfidf_features = self.transform_tfidf(texts)
        
        # Get hand-crafted features
        handcrafted_features = self.create_feature_matrix(texts)
        
        # Scale hand-crafted features
        if fit:
            handcrafted_scaled = self.scaler.fit_transform(handcrafted_features)
        else:
            handcrafted_scaled = self.scaler.transform(handcrafted_features)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([tfidf_features, csr_matrix(handcrafted_scaled)])
        
        return combined_features
    
    def save_vectorizer(self, path):
        """Save the fitted vectorizer and scaler"""
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler
        }, path)
        print(f"Vectorizer saved to {path}")
    
    def load_vectorizer(self, path):
        """Load the fitted vectorizer and scaler"""
        data = joblib.load(path)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.scaler = data['scaler']
        print(f"Vectorizer loaded from {path}")


if __name__ == "__main__":
    # Example usage
    from data_preparation import CountryDataPreparation
    
    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Extract features
    X_train = fe.combine_features(train_df['text'].tolist(), fit=True)
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Save vectorizer
    fe.save_vectorizer(SAVED_MODELS_DIR / 'vectorizer.pkl')