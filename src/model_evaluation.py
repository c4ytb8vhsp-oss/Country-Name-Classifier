"""
Model Evaluation Module
Provides comprehensive evaluation metrics and visualizations for the country classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
import joblib
from pathlib import Path
from config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR
from src.feature_engineering import FeatureEngineering


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model=None, feature_engineering=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model (if None, will load from saved models)
            feature_engineering: FeatureEngineering instance (if None, will create new)
        """
        if model is None:
            self.model = self.load_model()
        else:
            self.model = model
            
        if feature_engineering is None:
            self.feature_engineering = FeatureEngineering()
            self.feature_engineering.load_vectorizer(
                SAVED_MODELS_DIR / 'vectorizer.pkl'
            )
        else:
            self.feature_engineering = feature_engineering
    
    def load_model(self):
        """Load saved model"""
        model_path = SAVED_MODELS_DIR / 'country_classifier.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        return joblib.load(model_path)
    
    def load_test_data(self):
        """Load test dataset"""
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')
        
        X_test = self.feature_engineering.combine_features(
            test_df['text'].tolist(), 
            fit=False
        )
        y_test = test_df['label'].values
        
        return X_test, y_test, test_df['text'].tolist()
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            # ROC-AUC
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            metrics['roc_auc'] = auc(fpr, tpr)
            
            # Average Precision
            metrics['avg_precision'] = average_precision_score(y_true, y_proba[:, 1])
        
        return metrics
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Not Country', 'Country'],
            digits=4
        )
        print(report)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Country', 'Country'],
            yticklabels=['Not Country', 'Country'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save figure (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_proba, save_path=None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save figure (optional)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_confidence(self, y_true, y_proba, save_path=None):
        """
        Plot prediction confidence distribution
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save figure (optional)
        """
        # Get confidence for predicted class
        y_pred = np.argmax(y_proba, axis=1)
        confidence = np.max(y_proba, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (y_pred == y_true)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(confidence[correct_mask], bins=20, alpha=0.7, 
                    label='Correct', color='green', edgecolor='black')
        axes[0].hist(confidence[~correct_mask], bins=20, alpha=0.7, 
                    label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Confidence Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prediction Confidence Distribution', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot by correctness
        df_conf = pd.DataFrame({
            'confidence': confidence,
            'correct': ['Correct' if c else 'Incorrect' for c in correct_mask]
        })
        df_conf.boxplot(column='confidence', by='correct', ax=axes[1])
        axes[1].set_xlabel('Prediction Correctness', fontsize=12)
        axes[1].set_ylabel('Confidence Score', fontsize=12)
        axes[1].set_title('Confidence by Prediction Correctness', 
                         fontsize=13, fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence plot saved to {save_path}")
        
        plt.show()
    
    def analyze_errors(self, texts, y_true, y_pred, y_proba, n_examples=10):
        """
        Analyze and display misclassified examples
        
        Args:
            texts: Original text inputs
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            n_examples: Number of examples to show
        """
        # Find misclassified examples
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            print("\n✓ No errors found! Perfect classification!")
            return
        
        print("\n" + "="*70)
        print(f"ERROR ANALYSIS - Showing up to {n_examples} misclassified examples")
        print("="*70)
        
        # Sort by confidence (show high-confidence errors first)
        confidence = np.max(y_proba, axis=1)
        sorted_indices = error_indices[np.argsort(-confidence[error_indices])]
        
        for i, idx in enumerate(sorted_indices[:n_examples]):
            true_label = 'Country' if y_true[idx] == 1 else 'Not Country'
            pred_label = 'Country' if y_pred[idx] == 1 else 'Not Country'
            conf = confidence[idx]
            
            print(f"\n{i+1}. Text: '{texts[idx]}'")
            print(f"   True: {true_label} | Predicted: {pred_label} | Confidence: {conf:.2%}")
            
            # Add reasoning hints
            text_lower = texts[idx].lower()
            if 'bank' in text_lower or 'university' in text_lower:
                print(f"   💡 Hint: Contains organization keyword")
            if 'republic' in text_lower or 'kingdom' in text_lower:
                print(f"   💡 Hint: Contains country-related keyword")
    
    def analyze_by_confidence_threshold(self, y_true, y_pred, y_proba, 
                                       thresholds=[0.5, 0.7, 0.8, 0.9, 0.95]):
        """
        Analyze performance at different confidence thresholds
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            thresholds: List of confidence thresholds to analyze
        """
        print("\n" + "="*70)
        print("PERFORMANCE BY CONFIDENCE THRESHOLD")
        print("="*70)
        
        results = []
        confidence = np.max(y_proba, axis=1)
        
        for threshold in thresholds:
            mask = confidence >= threshold
            
            if mask.sum() == 0:
                continue
            
            acc = accuracy_score(y_true[mask], y_pred[mask])
            prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)
            rec = recall_score(y_true[mask], y_pred[mask], zero_division=0)
            f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
            coverage = mask.sum() / len(y_true)
            
            results.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })
        
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        return df_results
    
    def generate_evaluation_report(self, save_plots=True):
        """
        Generate comprehensive evaluation report
        
        Args:
            save_plots: Whether to save plots to disk
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*70)
        
        # Load test data
        X_test, y_test, texts = self.load_test_data()
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        print("\n" + "-"*70)
        print("OVERALL METRICS")
        print("-"*70)
        for metric, value in metrics.items():
            print(f"{metric.upper():20s}: {value:.4f}")
        
        # Classification report
        self.print_classification_report(y_test, y_pred)
        
        # Confusion matrix
        plot_path = 'confusion_matrix.png' if save_plots else None
        self.plot_confusion_matrix(y_test, y_pred, save_path=plot_path)
        
        if y_proba is not None:
            # ROC curve
            plot_path = 'roc_curve.png' if save_plots else None
            self.plot_roc_curve(y_test, y_proba, save_path=plot_path)
            
            # Precision-Recall curve
            plot_path = 'precision_recall_curve.png' if save_plots else None
            self.plot_precision_recall_curve(y_test, y_proba, save_path=plot_path)
            
            # Confidence distribution
            plot_path = 'confidence_distribution.png' if save_plots else None
            self.plot_prediction_confidence(y_test, y_proba, save_path=plot_path)
            
            # Threshold analysis
            self.analyze_by_confidence_threshold(y_test, y_pred, y_proba)
            
            # Error analysis
            self.analyze_errors(texts, y_test, y_pred, y_proba, n_examples=10)
        
        print("\n" + "="*70)
        print("✓ EVALUATION REPORT COMPLETE")
        print("="*70)
        
        return metrics


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    metrics = evaluator.generate_evaluation_report(save_plots=True)
    
    print("\nFinal Metrics Summary:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")