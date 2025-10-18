"""
CTR Model Training Script
Trains a LightGBM model for click-through rate prediction.
"""

import os
import pickle
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, log_loss, classification_report, 
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import joblib

from models.ctr_simulator import generate_training_data
from api.features import FeatureExtractor
from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class CTRModelTrainer:
    """Trains and evaluates CTR prediction models."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.model_path
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.feature_importance = None
        self.training_metrics = {}
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def prepare_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with features and labels.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, labels)
        """
        logger.info(f"Generating {n_samples} training samples...")
        
        # Generate synthetic data
        dataset = generate_training_data(n_samples=n_samples)
        
        # Extract features
        logger.info("Extracting features...")
        headlines = dataset['headline'].tolist()
        features_df = self.feature_extractor.extract_features_batch(headlines)
        
        # Prepare labels (use observed CTR as target)
        labels = dataset['observed_ctr'].values
        
        # Remove non-feature columns
        feature_columns = [col for col in features_df.columns 
                          if col not in ['headline', 'full_embedding']]
        features_df = features_df[feature_columns]
        
        logger.info(f"Prepared data: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        return features_df, labels
    
    def split_data(self, features_df: pd.DataFrame, labels: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features_df: Feature matrix
            labels: Target labels
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_df, labels, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=None
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
        """
        Train a baseline LightGBM model with default parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained LightGBM model
        """
        logger.info("Training baseline LightGBM model...")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Default parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Convert to sklearn interface for consistency
        sklearn_model = lgb.LGBMRegressor(**params)
        sklearn_model.fit(X_train, y_train)
        
        logger.info("Baseline model training completed")
        return sklearn_model
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'verbose': -1,
                'random_state': 42
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    LightGBMPruningCallback(trial, 'valid_0-rmse')
                ]
            )
            
            # Return validation RMSE
            return model.best_score['valid_0']['rmse']
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        })
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation RMSE: {study.best_value:.6f}")
        
        return best_params
    
    def train_optimized_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            best_params: Dict[str, Any]) -> lgb.LGBMRegressor:
        """
        Train model with optimized hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            best_params: Optimized hyperparameters
            
        Returns:
            Trained LightGBM model
        """
        logger.info("Training optimized LightGBM model...")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Convert to sklearn interface
        sklearn_model = lgb.LGBMRegressor(**best_params)
        sklearn_model.fit(X_train, y_train)
        
        logger.info("Optimized model training completed")
        return sklearn_model
    
    def evaluate_model(self, model: lgb.LGBMRegressor, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2)),
            'mae': np.mean(np.abs(y_test - y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            'r2': model.score(X_test, y_test)
        }
        
        # For binary classification metrics (treating CTR as binary)
        y_binary = (y_test > y_test.median()).astype(int)
        y_pred_binary = (y_pred > y_test.median()).astype(int)
        
        if len(np.unique(y_binary)) > 1:
            metrics['auc'] = roc_auc_score(y_binary, y_pred)
            metrics['logloss'] = log_loss(y_binary, y_pred)
        
        logger.info("Model evaluation completed")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, model: lgb.LGBMRegressor, 
                              feature_names: list, top_n: int = 20):
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to show
        """
        logger.info("Generating feature importance plot...")
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Store for later use
        self.feature_importance = feature_importance_df
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(self.model_path), 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
    
    def plot_calibration_curve(self, model: lgb.LGBMRegressor, X_test: pd.DataFrame, 
                             y_test: pd.Series):
        """
        Plot calibration curve for model predictions.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        logger.info("Generating calibration curve...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create binary labels for calibration
        y_binary = (y_test > y_test.median()).astype(int)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_pred, n_bins=10
        )
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"Model (Brier Score: {self._calculate_brier_score(y_binary, y_pred):.3f})")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(self.model_path), 'calibration_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curve saved to {plot_path}")
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Brier score for calibration assessment."""
        return np.mean((y_true - y_pred) ** 2)
    
    def save_model(self, model: lgb.LGBMRegressor, metrics: Dict[str, float]):
        """
        Save trained model and metadata.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
        """
        logger.info(f"Saving model to {self.model_path}")
        
        # Save model
        joblib.dump(model, self.model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'LightGBM',
            'training_metrics': metrics,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'feature_count': model.n_features_in_,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info("Model and metadata saved successfully")
    
    def train_full_pipeline(self, n_samples: int = 10000, optimize: bool = True, 
                          n_trials: int = 20) -> lgb.LGBMRegressor:
        """
        Run the complete training pipeline.
        
        Args:
            n_samples: Number of training samples
            optimize: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            Trained model
        """
        logger.info("Starting full training pipeline...")
        
        # Prepare data
        features_df, labels = self.prepare_data(n_samples)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(features_df, labels)
        
        # Train model
        if optimize:
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
            model = self.train_optimized_model(X_train, y_train, X_val, y_val, best_params)
        else:
            # Train baseline model
            model = self.train_baseline_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        self.training_metrics = metrics
        
        # Generate plots
        self.plot_feature_importance(model, features_df.columns.tolist())
        self.plot_calibration_curve(model, X_test, y_test)
        
        # Save model
        self.save_model(model, metrics)
        
        # Store model
        self.model = model
        
        logger.info("Training pipeline completed successfully!")
        return model


def main():
    """Main training function."""
    trainer = CTRModelTrainer()
    
    # Train model with optimization
    model = trainer.train_full_pipeline(
        n_samples=10000,
        optimize=True,
        n_trials=20
    )
    
    print("Training completed successfully!")
    print(f"Model saved to: {trainer.model_path}")
    print(f"Training metrics: {trainer.training_metrics}")


if __name__ == "__main__":
    main()
