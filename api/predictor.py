"""
CTR Prediction Service
Provides inference capabilities for the trained CTR model.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from api.features import FeatureExtractor
from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class CTRPredictor:
    """CTR prediction service using trained LightGBM model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the CTR predictor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path or settings.model_path
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.feature_names = None
        self.model_metadata = None
        
        # Load model if it exists
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the trained model and metadata.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = joblib.load(self.model_path)
                
                # Load metadata if available
                metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    self.model_metadata = joblib.load(metadata_path)
                    logger.info("Model metadata loaded successfully")
                
                # Get feature names from model
                if hasattr(self.model, 'feature_name_'):
                    self.feature_names = self.model.feature_name_
                elif hasattr(self.model, 'feature_importances_'):
                    # Fallback: create generic feature names
                    n_features = len(self.model.feature_importances_)
                    self.feature_names = [f"feature_{i}" for i in range(n_features)]
                
                logger.info(f"Model loaded successfully with {len(self.feature_names)} features")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and ready for inference.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def predict_ctr(self, headlines: List[str]) -> List[float]:
        """
        Predict CTR for a list of headlines.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            List of predicted CTR values
        """
        if not self.is_model_loaded():
            logger.error("Model not loaded. Cannot make predictions.")
            return [0.0] * len(headlines)
        
        try:
            # Extract features
            logger.info(f"Extracting features for {len(headlines)} headlines")
            features_df = self.feature_extractor.extract_features_batch(headlines)
            
            # Prepare features for prediction
            feature_columns = [col for col in features_df.columns 
                             if col not in ['headline', 'full_embedding']]
            X = features_df[feature_columns]
            
            # Ensure feature order matches training
            if self.feature_names:
                # Reorder columns to match training
                missing_features = set(self.feature_names) - set(X.columns)
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with zero values
                    for feature in missing_features:
                        X[feature] = 0.0
                
                # Reorder columns
                X = X[self.feature_names]
            
            # Make predictions
            logger.info("Making CTR predictions")
            predictions = self.model.predict(X)
            
            # Ensure predictions are non-negative and reasonable
            predictions = np.clip(predictions, 0.0, 1.0)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return [0.0] * len(headlines)
    
    def predict_ctr_single(self, headline: str) -> float:
        """
        Predict CTR for a single headline.
        
        Args:
            headline: Headline string
            
        Returns:
            Predicted CTR value
        """
        predictions = self.predict_ctr([headline])
        return predictions[0] if predictions else 0.0
    
    def rank_headlines(self, headlines: List[str], 
                      return_scores: bool = False) -> List[Dict[str, Any]]:
        """
        Rank headlines by predicted CTR.
        
        Args:
            headlines: List of headline strings
            return_scores: Whether to include prediction scores
            
        Returns:
            List of dictionaries with headline and ranking information
        """
        if not headlines:
            return []
        
        # Get predictions
        predictions = self.predict_ctr(headlines)
        
        # Create ranking data
        ranking_data = []
        for i, (headline, score) in enumerate(zip(headlines, predictions)):
            item = {
                'headline': headline,
                'rank': i + 1,
                'predicted_ctr': score
            }
            if return_scores:
                item['raw_score'] = score
            
            ranking_data.append(item)
        
        # Sort by predicted CTR (descending)
        ranking_data.sort(key=lambda x: x['predicted_ctr'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(ranking_data):
            item['rank'] = i + 1
        
        logger.info(f"Ranked {len(headlines)} headlines")
        return ranking_data
    
    def get_top_headlines(self, headlines: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top K headlines by predicted CTR.
        
        Args:
            headlines: List of headline strings
            top_k: Number of top headlines to return
            
        Returns:
            List of top headlines with ranking information
        """
        ranked_headlines = self.rank_headlines(headlines, return_scores=True)
        return ranked_headlines[:top_k]
    
    def compare_headlines(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Compare multiple headlines and provide detailed analysis.
        
        Args:
            headlines: List of headline strings to compare
            
        Returns:
            Dictionary with comparison results
        """
        if not headlines:
            return {'error': 'No headlines provided'}
        
        # Get predictions
        predictions = self.predict_ctr(headlines)
        
        # Calculate statistics
        mean_ctr = np.mean(predictions)
        std_ctr = np.std(predictions)
        max_ctr = np.max(predictions)
        min_ctr = np.min(predictions)
        
        # Find best and worst
        best_idx = np.argmax(predictions)
        worst_idx = np.argmin(predictions)
        
        # Create comparison results
        comparison = {
            'headlines': [
                {
                    'headline': headline,
                    'predicted_ctr': float(score),
                    'is_best': i == best_idx,
                    'is_worst': i == worst_idx
                }
                for i, (headline, score) in enumerate(zip(headlines, predictions))
            ],
            'statistics': {
                'mean_ctr': float(mean_ctr),
                'std_ctr': float(std_ctr),
                'max_ctr': float(max_ctr),
                'min_ctr': float(min_ctr),
                'range_ctr': float(max_ctr - min_ctr)
            },
            'best_headline': {
                'headline': headlines[best_idx],
                'predicted_ctr': float(predictions[best_idx])
            },
            'worst_headline': {
                'headline': headlines[worst_idx],
                'predicted_ctr': float(predictions[worst_idx])
            }
        }
        
        logger.info(f"Compared {len(headlines)} headlines")
        return comparison
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_model_loaded():
            return {'error': 'Model not loaded'}
        
        info = {
            'model_type': 'LightGBM',
            'model_path': self.model_path,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'is_loaded': True
        }
        
        # Add metadata if available
        if self.model_metadata:
            info.update({
                'training_date': self.model_metadata.get('training_date'),
                'training_metrics': self.model_metadata.get('training_metrics', {}),
                'feature_importance': self.model_metadata.get('feature_importance')
            })
        
        return info
    
    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of feature importance information
        """
        if not self.is_model_loaded():
            return []
        
        # Get feature importance from model
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
            
            # Create importance data
            importance_data = [
                {'feature': name, 'importance': float(imp)}
                for name, imp in zip(feature_names, importance)
            ]
            
            # Sort by importance
            importance_data.sort(key=lambda x: x['importance'], reverse=True)
            
            return importance_data[:top_n]
        
        return []
    
    def validate_headlines(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Validate headlines for prediction.
        
        Args:
            headlines: List of headlines to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': [],
            'invalid': [],
            'warnings': []
        }
        
        for i, headline in enumerate(headlines):
            issues = []
            
            # Check length
            if len(headline.strip()) == 0:
                issues.append("Empty headline")
            elif len(headline) > 200:
                issues.append("Headline too long (>200 chars)")
            elif len(headline) < 10:
                issues.append("Headline too short (<10 chars)")
            
            # Check for special characters
            if any(char in headline for char in ['<', '>', '&']):
                issues.append("Contains HTML-like characters")
            
            # Check for excessive repetition
            words = headline.lower().split()
            if len(words) > 0:
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                max_repetition = max(word_counts.values())
                if max_repetition > len(words) * 0.4:
                    issues.append("Excessive word repetition")
            
            if issues:
                validation_results['invalid'].append({
                    'index': i,
                    'headline': headline,
                    'issues': issues
                })
            else:
                validation_results['valid'].append({
                    'index': i,
                    'headline': headline
                })
        
        # Add warnings
        if len(validation_results['invalid']) > 0:
            validation_results['warnings'].append(
                f"{len(validation_results['invalid'])} headlines have validation issues"
            )
        
        return validation_results


# Global predictor instance
_predictor_instance = None


def get_predictor() -> CTRPredictor:
    """
    Get the global predictor instance (singleton pattern).
    
    Returns:
        CTRPredictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CTRPredictor()
    return _predictor_instance


def predict_ctr_batch(headlines: List[str]) -> List[float]:
    """
    Convenience function for batch CTR prediction.
    
    Args:
        headlines: List of headlines
        
    Returns:
        List of predicted CTR values
    """
    predictor = get_predictor()
    return predictor.predict_ctr(headlines)


def predict_ctr_single(headline: str) -> float:
    """
    Convenience function for single CTR prediction.
    
    Args:
        headline: Headline string
        
    Returns:
        Predicted CTR value
    """
    predictor = get_predictor()
    return predictor.predict_ctr_single(headline)


def main():
    """Test the predictor functionality."""
    predictor = CTRPredictor()
    
    if not predictor.is_model_loaded():
        print("Model not loaded. Please train a model first.")
        return
    
    # Test headlines
    test_headlines = [
        "Get 50% Off Today Only!",
        "Transform Your Life in 30 Days",
        "Free Shipping on All Orders",
        "Limited Time Offer - Don't Miss Out!",
        "Revolutionary New Product Launch"
    ]
    
    print("Testing CTR Predictor...")
    print(f"Model Info: {predictor.get_model_info()}")
    
    # Test predictions
    predictions = predictor.predict_ctr(test_headlines)
    print(f"\nPredictions: {predictions}")
    
    # Test ranking
    ranked = predictor.rank_headlines(test_headlines)
    print(f"\nRanked Headlines:")
    for item in ranked:
        print(f"  {item['rank']}. {item['headline']} (CTR: {item['predicted_ctr']:.4f})")
    
    # Test comparison
    comparison = predictor.compare_headlines(test_headlines)
    print(f"\nBest Headline: {comparison['best_headline']['headline']} (CTR: {comparison['best_headline']['predicted_ctr']:.4f})")
    print(f"Worst Headline: {comparison['worst_headline']['headline']} (CTR: {comparison['worst_headline']['predicted_ctr']:.4f})")


if __name__ == "__main__":
    main()
