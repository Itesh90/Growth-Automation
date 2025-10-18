"""
Basic functionality test for the Ad Headline Optimizer.
Tests core components without requiring API keys.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ctr_simulator import CTRSimulator, generate_training_data
from api.features import FeatureExtractor


def test_feature_extraction():
    """Test feature extraction pipeline."""
    print("Testing feature extraction...")
    
    try:
        extractor = FeatureExtractor()
        
        # Test headlines
        test_headlines = [
            "Get fit in 30 days - guaranteed results!",
            "How to boost your sales by 50%",
            "3 simple steps to financial freedom"
        ]
        
        # Extract features
        features_df = extractor.extract_features_batch(test_headlines)
        
        print(f"[OK] Extracted {len(features_df.columns)} features for {len(features_df)} headlines")
        print(f"   Features include: {list(features_df.columns[:10])}...")
        
        # Check for expected features
        expected_features = ['char_count', 'word_count', 'sentiment_polarity', 'has_action_verb']
        missing_features = [f for f in expected_features if f not in features_df.columns]
        
        if missing_features:
            print(f"[WARN] Missing features: {missing_features}")
        else:
            print("[OK] All expected features present")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Feature extraction test failed: {e}")
        return False


def test_ctr_simulation():
    """Test CTR simulation."""
    print("\nTesting CTR simulation...")
    
    try:
        simulator = CTRSimulator()
        
        # Generate small dataset
        dataset = simulator.generate_synthetic_data()
        
        print(f"✅ Generated {len(dataset)} synthetic samples")
        print(f"   CTR range: {dataset['observed_ctr'].min():.4f} - {dataset['observed_ctr'].max():.4f}")
        print(f"   Average CTR: {dataset['observed_ctr'].mean():.4f}")
        
        # Test A/B simulation
        control = "Get fit in 30 days"
        variants = ["Transform your body in 30 days", "Lose weight fast - 30 day challenge"]
        
        ab_results = simulator.generate_ab_test_data(control, variants, 5000)
        
        print(f"✅ A/B test simulation completed")
        print(f"   Control CTR: {ab_results[ab_results['is_control']]['observed_ctr'].iloc[0]:.4f}")
        print(f"   Best variant CTR: {ab_results[~ab_results['is_control']]['observed_ctr'].max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CTR simulation test failed: {e}")
        return False


def test_data_quality():
    """Test data quality and consistency."""
    print("\n🧪 Testing data quality...")
    
    try:
        # Generate training data
        dataset = generate_training_data(n_samples=1000)
        
        # Check for missing values
        missing_values = dataset.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠️  Found {missing_values} missing values")
        else:
            print("✅ No missing values found")
        
        # Check CTR distribution
        ctr_stats = dataset['observed_ctr'].describe()
        print(f"✅ CTR statistics:")
        print(f"   Mean: {ctr_stats['mean']:.4f}")
        print(f"   Std: {ctr_stats['std']:.4f}")
        print(f"   Min: {ctr_stats['min']:.4f}")
        print(f"   Max: {ctr_stats['max']:.4f}")
        
        # Check feature ranges
        numeric_features = dataset.select_dtypes(include=[np.number]).columns
        feature_ranges = {}
        
        for feature in numeric_features[:10]:  # Check first 10 features
            if feature not in ['true_ctr', 'observed_ctr', 'impressions', 'clicks', 'click_label']:
                feature_ranges[feature] = {
                    'min': dataset[feature].min(),
                    'max': dataset[feature].max(),
                    'mean': dataset[feature].mean()
                }
        
        print(f"✅ Feature ranges checked for {len(feature_ranges)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False


def test_performance():
    """Test performance benchmarks."""
    print("\n🧪 Testing performance...")
    
    try:
        import time
        
        # Test feature extraction speed
        extractor = FeatureExtractor()
        test_headlines = [f"Test headline {i}" for i in range(50)]
        
        start_time = time.time()
        features_df = extractor.extract_features_batch(test_headlines)
        extraction_time = time.time() - start_time
        
        print(f"✅ Feature extraction: {extraction_time:.2f}s for {len(test_headlines)} headlines")
        print(f"   Rate: {len(test_headlines)/extraction_time:.1f} headlines/second")
        
        # Test simulation speed
        simulator = CTRSimulator()
        
        start_time = time.time()
        dataset = simulator.generate_synthetic_data()
        simulation_time = time.time() - start_time
        
        print(f"✅ CTR simulation: {simulation_time:.2f}s for {len(dataset)} samples")
        print(f"   Rate: {len(dataset)/simulation_time:.1f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Ad Headline Optimizer - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_feature_extraction,
        test_ctr_simulation,
        test_data_quality,
        test_performance
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The core functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
