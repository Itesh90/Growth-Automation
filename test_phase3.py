"""
Test script for Phase 3 components: A/B Testing, Database Schema, and Dashboard
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.ab_harness import ABTestHarness, ABTestConfig, ABTestResult
from api.predictor import CTRPredictor
from utils.diversity_filter import DiversityFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ab_testing_framework():
    """Test the A/B testing framework."""
    print("=" * 60)
    print("TESTING A/B TESTING FRAMEWORK")
    print("=" * 60)
    
    try:
        # Initialize harness
        harness = ABTestHarness()
        print("✓ ABTestHarness initialized successfully")
        
        # Test sample size calculation
        sample_size = harness.calculate_sample_size(
            baseline_ctr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8
        )
        print(f"✓ Sample size calculation: {sample_size} per variant")
        
        # Test test creation
        test_id = "test_phase3_001"
        config = harness.create_test(
            test_id=test_id,
            variant_a="Get 50% Off Today Only!",
            variant_b="Transform Your Life in 30 Days",
            target_impressions=10000
        )
        print(f"✓ Test created: {test_id}")
        
        # Test data simulation
        simulated_data = harness.simulate_test_data(
            variant_a=config.variant_a,
            variant_b=config.variant_b,
            impressions_a=10000,
            impressions_b=10000,
            true_ctr_a=0.05,
            true_ctr_b=0.06
        )
        print(f"✓ Data simulated: {simulated_data['clicks_a']} vs {simulated_data['clicks_b']} clicks")
        
        # Test statistical analysis
        test_results = harness.run_statistical_test(
            clicks_a=simulated_data['clicks_a'],
            impressions_a=simulated_data['impressions_a'],
            clicks_b=simulated_data['clicks_b'],
            impressions_b=simulated_data['impressions_b']
        )
        print(f"✓ Statistical test: p-value={test_results['p_value_z']:.4f}, significant={test_results['is_significant']}")
        
        # Test full test execution
        result = harness.run_test(
            test_id=test_id,
            impressions_a=simulated_data['impressions_a'],
            impressions_b=simulated_data['impressions_b'],
            clicks_a=simulated_data['clicks_a'],
            clicks_b=simulated_data['clicks_b']
        )
        print(f"✓ Test executed: lift={result.lift:.1%}, p-value={result.p_value:.4f}")
        
        # Test test simulation
        test_id_2 = "test_phase3_002"
        config_2 = harness.create_test(
            test_id=test_id_2,
            variant_a="Free Shipping on All Orders",
            variant_b="Limited Time Offer - Don't Miss Out!",
            target_impressions=5000
        )
        
        result_2 = harness.simulate_test(
            test_id=test_id_2,
            true_ctr_a=0.04,
            true_ctr_b=0.05
        )
        print(f"✓ Test simulated: lift={result_2.lift:.1%}, significant={result_2.is_significant}")
        
        # Test report generation
        report = harness.generate_report(test_id)
        print(f"✓ Report generated: {len(report)} characters")
        
        # Test test listing
        tests = harness.list_tests()
        print(f"✓ Tests listed: {len(tests)} tests found")
        
        return True
        
    except Exception as e:
        print(f"✗ A/B testing framework failed: {e}")
        return False


def test_database_schema():
    """Test database schema functionality."""
    print("\n" + "=" * 60)
    print("TESTING DATABASE SCHEMA")
    print("=" * 60)
    
    try:
        # Check if schema file exists
        schema_path = "database/schema.sql"
        if os.path.exists(schema_path):
            print("✓ Database schema file exists")
            
            # Read and validate schema
            with open(schema_path, 'r') as f:
                schema_content = f.read()
            
            # Check for key tables
            required_tables = [
                'campaigns', 'headlines', 'headline_features', 'ctr_predictions',
                'ab_tests', 'ab_test_results', 'model_performance', 'feature_importance'
            ]
            
            for table in required_tables:
                if f"CREATE TABLE" in schema_content and table in schema_content:
                    print(f"✓ Table '{table}' found in schema")
                else:
                    print(f"⚠ Table '{table}' not found in schema")
            
            # Check for indexes
            if "CREATE INDEX" in schema_content:
                print("✓ Database indexes defined")
            else:
                print("⚠ No database indexes found")
            
            # Check for views
            if "CREATE OR REPLACE VIEW" in schema_content:
                print("✓ Database views defined")
            else:
                print("⚠ No database views found")
            
            # Check for triggers
            if "CREATE TRIGGER" in schema_content:
                print("✓ Database triggers defined")
            else:
                print("⚠ No database triggers found")
            
            print(f"✓ Schema file size: {len(schema_content)} characters")
            
        else:
            print("✗ Database schema file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Database schema test failed: {e}")
        return False


def test_dashboard_components():
    """Test dashboard components."""
    print("\n" + "=" * 60)
    print("TESTING DASHBOARD COMPONENTS")
    print("=" * 60)
    
    try:
        # Test dashboard import
        from app.dashboard import Dashboard
        print("✓ Dashboard class imported successfully")
        
        # Test dashboard initialization
        dashboard = Dashboard()
        print("✓ Dashboard initialized successfully")
        
        # Test component initialization
        print(f"✓ ABTestHarness: {type(dashboard.ab_harness).__name__}")
        print(f"✓ CTRPredictor: {type(dashboard.predictor).__name__}")
        print(f"✓ DiversityFilter: {type(dashboard.diversity_filter).__name__}")
        
        # Test campaign data
        campaigns = dashboard._get_campaigns()
        print(f"✓ Campaigns retrieved: {len(campaigns)} campaigns")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard components test failed: {e}")
        return False


def test_integration():
    """Test integration between Phase 3 components."""
    print("\n" + "=" * 60)
    print("TESTING PHASE 3 INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize components
        harness = ABTestHarness()
        predictor = CTRPredictor()
        diversity_filter = DiversityFilter()
        
        # Test headlines
        test_headlines = [
            "Get 50% Off Today Only!",
            "Transform Your Life in 30 Days",
            "Free Shipping on All Orders",
            "Limited Time Offer - Don't Miss Out!",
            "Revolutionary New Product Launch"
        ]
        
        # Test diversity filtering
        diverse_headlines = diversity_filter.ensure_diversity(test_headlines)
        print(f"✓ Diversity filtering: {len(test_headlines)} → {len(diverse_headlines)} headlines")
        
        # Test A/B test creation with diverse headlines
        if len(diverse_headlines) >= 2:
            test_id = "integration_test_001"
            config = harness.create_test(
                test_id=test_id,
                variant_a=diverse_headlines[0],
                variant_b=diverse_headlines[1],
                target_impressions=5000
            )
            print(f"✓ A/B test created with diverse headlines: {test_id}")
            
            # Test prediction integration (if model is available)
            if predictor.is_model_loaded():
                predictions = predictor.predict_ctr(diverse_headlines)
                print(f"✓ Predictions generated for {len(predictions)} headlines")
                
                # Test ranking
                ranked = predictor.rank_headlines(diverse_headlines)
                print(f"✓ Headlines ranked: {len(ranked)} headlines")
            else:
                print("⚠ Predictions skipped - no model loaded")
            
            # Test simulation
            result = harness.simulate_test(
                test_id=test_id,
                true_ctr_a=0.05,
                true_ctr_b=0.06
            )
            print(f"✓ Test simulated: lift={result.lift:.1%}, significant={result.is_significant}")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 3 integration test failed: {e}")
        return False


def test_performance():
    """Test performance of Phase 3 components."""
    print("\n" + "=" * 60)
    print("TESTING PHASE 3 PERFORMANCE")
    print("=" * 60)
    
    try:
        import time
        
        # Test A/B testing performance
        harness = ABTestHarness()
        
        start_time = time.time()
        for i in range(10):
            test_id = f"perf_test_{i}"
            config = harness.create_test(
                test_id=test_id,
                variant_a=f"Test headline A {i}",
                variant_b=f"Test headline B {i}",
                target_impressions=1000
            )
            
            result = harness.simulate_test(
                test_id=test_id,
                true_ctr_a=0.05,
                true_ctr_b=0.06
            )
        
        ab_test_time = time.time() - start_time
        print(f"✓ A/B testing: 10 tests in {ab_test_time:.2f}s ({10/ab_test_time:.1f} tests/s)")
        
        # Test diversity filtering performance
        diversity_filter = DiversityFilter()
        test_headlines = [f"Test headline {i} for performance testing" for i in range(50)]
        
        start_time = time.time()
        diverse = diversity_filter.ensure_diversity(test_headlines)
        diversity_time = time.time() - start_time
        print(f"✓ Diversity filtering: {len(test_headlines)} headlines in {diversity_time:.2f}s ({len(test_headlines)/diversity_time:.1f} headlines/s)")
        
        # Test statistical calculations
        start_time = time.time()
        for i in range(100):
            harness.run_statistical_test(
                clicks_a=500,
                impressions_a=10000,
                clicks_b=600,
                impressions_b=10000
            )
        stats_time = time.time() - start_time
        print(f"✓ Statistical tests: 100 tests in {stats_time:.2f}s ({100/stats_time:.1f} tests/s)")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 3 performance test failed: {e}")
        return False


def main():
    """Run all Phase 3 tests."""
    print("PHASE 3 COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("A/B Testing Framework", test_ab_testing_framework),
        ("Database Schema", test_database_schema),
        ("Dashboard Components", test_dashboard_components),
        ("Phase 3 Integration", test_integration),
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
        print("🎉 All Phase 3 tests passed!")
    else:
        print("⚠ Some tests failed - check the output above for details")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
