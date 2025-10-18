"""
Simple test for Phase 2 components
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all components can be imported."""
    try:
        from models.train_ctr_model import CTRModelTrainer
        print("✓ CTRModelTrainer imported successfully")
        
        from api.predictor import CTRPredictor
        print("✓ CTRPredictor imported successfully")
        
        from utils.diversity_filter import DiversityFilter
        print("✓ DiversityFilter imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_initialization():
    """Test that components can be initialized."""
    try:
        from models.train_ctr_model import CTRModelTrainer
        from api.predictor import CTRPredictor
        from utils.diversity_filter import DiversityFilter
        
        trainer = CTRModelTrainer()
        print("✓ CTRModelTrainer initialized")
        
        predictor = CTRPredictor()
        print("✓ CTRPredictor initialized")
        
        filter_obj = DiversityFilter()
        print("✓ DiversityFilter initialized")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of components."""
    try:
        from utils.diversity_filter import DiversityFilter
        
        # Test diversity filter
        filter_obj = DiversityFilter()
        test_headlines = [
            "Get 50% Off Today Only!",
            "Get 50% Off Today Only!",  # Duplicate
            "Transform Your Life in 30 Days",
            "Free Shipping on All Orders"
        ]
        
        # Test diversity analysis
        analysis = filter_obj.analyze_diversity(test_headlines)
        print(f"✓ Diversity analysis: Score={analysis['diversity_score']:.3f}")
        
        # Test filtering
        filtered = filter_obj.filter_similar_headlines(test_headlines)
        print(f"✓ Filtering: {len(test_headlines)} → {len(filtered)} headlines")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple Phase 2 Component Tests")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Initialization", test_initialization),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
