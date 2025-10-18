"""
Test script for Phase 2 components: Training, Prediction, and Diversity Filtering
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.train_ctr_model import CTRModelTrainer
from api.predictor import CTRPredictor
from utils.diversity_filter import DiversityFilter
from models.ctr_simulator import generate_training_data
from api.features import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_training_pipeline():
    """Test the training pipeline."""
    print("=" * 60)
    print("TESTING TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = CTRModelTrainer()
        print("✓ CTRModelTrainer initialized successfully")
        
        # Test data preparation
        features_df, labels = trainer.prepare_data(n_samples=1000)  # Smaller sample for testing
        print(f"✓ Data prepared: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(features_df, labels)
        print(f"✓ Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        
        # Test baseline model training
        model = trainer.train_baseline_model(X_train, y_train, X_val, y_val)
        print("✓ Baseline model trained successfully")
        
        # Test model evaluation
        metrics = trainer.evaluate_model(model, X_test, y_test)
        print(f"✓ Model evaluated: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Test model saving
        trainer.save_model(model, metrics)
        print("✓ Model saved successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
        return False


def test_prediction_service():
    """Test the prediction service."""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION SERVICE")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = CTRPredictor()
        print("✓ CTRPredictor initialized successfully")
        
        # Check if model is loaded
        if not predictor.is_model_loaded():
            print("⚠ Model not loaded - this is expected if no model has been trained yet")
            return True
        
        # Test headlines
        test_headlines = [
            "Get 50% Off Today Only!",
            "Transform Your Life in 30 Days",
            "Free Shipping on All Orders",
            "Limited Time Offer - Don't Miss Out!",
            "Revolutionary New Product Launch"
        ]
        
        # Test single prediction
        single_pred = predictor.predict_ctr_single(test_headlines[0])
        print(f"✓ Single prediction: {single_pred:.4f}")
        
        # Test batch prediction
        batch_preds = predictor.predict_ctr(test_headlines)
        print(f"✓ Batch predictions: {len(batch_preds)} predictions generated")
        
        # Test ranking
        ranked = predictor.rank_headlines(test_headlines)
        print(f"✓ Headlines ranked: {len(ranked)} headlines")
        
        # Test comparison
        comparison = predictor.compare_headlines(test_headlines)
        print(f"✓ Headlines compared: Best CTR={comparison['best_headline']['predicted_ctr']:.4f}")
        
        # Test model info
        model_info = predictor.get_model_info()
        print(f"✓ Model info retrieved: {model_info.get('feature_count', 0)} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction service failed: {e}")
        return False


def test_diversity_filter():
    """Test the diversity filter."""
    print("\n" + "=" * 60)
    print("TESTING DIVERSITY FILTER")
    print("=" * 60)
    
    try:
        # Initialize filter
        filter_obj = DiversityFilter()
        print("✓ DiversityFilter initialized successfully")
        
        # Test headlines with some duplicates
        test_headlines = [
            "Get 50% Off Today Only!",
            "Get 50% Off Today Only!",  # Duplicate
            "Transform Your Life in 30 Days",
            "Transform Your Life in 30 Days",  # Duplicate
            "Free Shipping on All Orders",
            "Limited Time Offer - Don't Miss Out!",
            "Revolutionary New Product Launch",
            "Amazing Deal - Buy Now!",
            "Special Promotion - Limited Time",
            "Incredible Savings - Act Fast!"
        ]
        
        # Test diversity analysis
        analysis = filter_obj.analyze_diversity(test_headlines)
        print(f"✓ Diversity analysis: Score={analysis['diversity_score']:.3f}")
        
        # Test similarity filtering
        filtered = filter_obj.filter_similar_headlines(test_headlines)
        print(f"✓ Similarity filtering: {len(test_headlines)} → {len(filtered)} headlines")
        
        # Test diversity enforcement
        diverse = filter_obj.ensure_diversity(test_headlines)
        print(f"✓ Diversity enforcement: {len(test_headlines)} → {len(diverse)} headlines")
        
        # Test recommendations
        recommendations = filter_obj.get_diversity_recommendations(test_headlines)
        print(f"✓ Recommendations generated: {len(recommendations)} suggestions")
        
        return True
        
    except Exception as e:
        print(f"✗ Diversity filter failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("\n" + "=" * 60)
    print("TESTING COMPONENT INTEGRATION")
    print("=" * 60)
    
    try:
        # Test feature extraction
        feature_extractor = FeatureExtractor()
        test_headlines = [
            "Get 50% Off Today Only!",
            "Transform Your Life in 30 Days",
            "Free Shipping on All Orders"
        ]
        
        features_df = feature_extractor.extract_features_batch(test_headlines)
        print(f"✓ Feature extraction: {features_df.shape[1]} features extracted")
        
        # Test diversity filter with features
        filter_obj = DiversityFilter()
        diverse_headlines = filter_obj.ensure_diversity(test_headlines)
        print(f"✓ Diversity filtering: {len(test_headlines)} → {len(diverse_headlines)} headlines")
        
        # Test prediction (if model exists)
        predictor = CTRPredictor()
        if predictor.is_model_loaded():
            predictions = predictor.predict_ctr(diverse_headlines)
            print(f"✓ Prediction integration: {len(predictions)} predictions for diverse headlines")
        else:
            print("⚠ Prediction skipped - no model loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def test_performance():
    """Test performance of components."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE")
    print("=" * 60)
    
    try:
        import time
        
        # Test feature extraction performance
        feature_extractor = FeatureExtractor()
        test_headlines = [f"Test headline {i} for performance testing" for i in range(100)]
        
        start_time = time.time()
        features_df = feature_extractor.extract_features_batch(test_headlines)
        feature_time = time.time() - start_time
        print(f"✓ Feature extraction: {len(test_headlines)} headlines in {feature_time:.2f}s ({len(test_headlines)/feature_time:.1f} headlines/s)")
        
        # Test diversity filter performance
        filter_obj = DiversityFilter()
        start_time = time.time()
        diverse = filter_obj.ensure_diversity(test_headlines)
        diversity_time = time.time() - start_time
        print(f"✓ Diversity filtering: {len(test_headlines)} headlines in {diversity_time:.2f}s ({len(test_headlines)/diversity_time:.1f} headlines/s)")
        
        # Test prediction performance (if model exists)
        predictor = CTRPredictor()
        if predictor.is_model_loaded():
            start_time = time.time()
            predictions = predictor.predict_ctr(diverse)
            prediction_time = time.time() - start_time
            print(f"✓ Prediction: {len(diverse)} headlines in {prediction_time:.2f}s ({len(diverse)/prediction_time:.1f} headlines/s)")
        else:
            print("⚠ Prediction performance test skipped - no model loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    print("PHASE 2 COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("Training Pipeline", test_training_pipeline),
        ("Prediction Service", test_prediction_service),
        ("Diversity Filter", test_diversity_filter),
        ("Component Integration", test_integration),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed!")
    else:
        print("⚠ Some tests failed - check the output above for details")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
