import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from config import *
from src.feature_engineering import FeatureEngineering

class CountryClassifierTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_engineering = FeatureEngineering()
        
    def load_data(self):
        """Load train, validation, and test datasets"""
        train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
        val_df = pd.read_csv(PROCESSED_DATA_DIR / 'val.csv')
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """Prepare feature matrices"""
        # Extract features
        X_train = self.feature_engineering.combine_features(train_df['text'].tolist(), fit=True)
        X_val = self.feature_engineering.combine_features(val_df['text'].tolist(), fit=False)
        X_test = self.feature_engineering.combine_features(test_df['text'].tolist(), fit=False)
        
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train logistic regression with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Logistic Regression")
        print("="*50)
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'max_iter': [1000],
            'class_weight': ['balanced', None]
        }
        
        lr = LogisticRegression(random_state=MODEL_CONFIG['random_state'])
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        val_score = f1_score(y_val, val_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Validation F1 Score: {val_score:.4f}")
        
        self.models['logistic_regression'] = {
            'model': best_model,
            'val_score': val_score
        }
        
        return best_model, val_score
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train random forest with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Random Forest")
        print("="*50)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=MODEL_CONFIG['random_state'])
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        val_score = f1_score(y_val, val_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Validation F1 Score: {val_score:.4f}")
        
        self.models['random_forest'] = {
            'model': best_model,
            'val_score': val_score
        }
        
        return best_model, val_score
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train gradient boosting with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Gradient Boosting")
        print("="*50)
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=MODEL_CONFIG['random_state'])
        
        grid_search = GridSearchCV(
            gb, param_grid, cv=3, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        val_score = f1_score(y_val, val_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Validation F1 Score: {val_score:.4f}")
        
        self.models['gradient_boosting'] = {
            'model': best_model,
            'val_score': val_score
        }
        
        return best_model, val_score
    
    def select_best_model(self):
        """Select the best model based on validation performance"""
        best_score = -1
        best_name = None
        
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        for name, data in self.models.items():
            score = data['val_score']
            print(f"{name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\nBest model: {best_name} with F1 score: {best_score:.4f}")
        
        return self.best_model, best_name
    
    def evaluate_on_test(self, X_test, y_test):
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No best model selected!")
        
        y_pred = self.best_model.predict(X_test)
        
        print("\n" + "="*50)
        print(f"Test Set Evaluation - {self.best_model_name}")
        print("="*50)
        
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Not Country', 'Country']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    
    def save_model(self):
        """Save the best model and feature engineering pipeline"""
        if self.best_model is None:
            raise ValueError("No best model to save!")
        
        # Save model
        model_path = SAVED_MODELS_DIR / 'country_classifier.pkl'
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save feature engineering pipeline
        vectorizer_path = SAVED_MODELS_DIR / 'vectorizer.pkl'
        self.feature_engineering.save_vectorizer(vectorizer_path)
        
    def train_all(self):
        """Complete training pipeline"""
        # Load data
        print("Loading data...")
        train_df, val_df, test_df = self.load_data()
        
        # Prepare features
        print("Preparing features...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(
            train_df, val_df, test_df
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Validation set size: {X_val.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Train models
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_gradient_boosting(X_train, y_train, X_val, y_val)
        
        # Select best model
        self.select_best_model()
        
        # Evaluate on test set
        self.evaluate_on_test(X_test, y_test)
        
        # Save model
        self.save_model()
        
        return self.best_model


if __name__ == "__main__":
    trainer = CountryClassifierTrainer()
    model = trainer.train_all()