#!/usr/bin/env python3
"""
Ad Headline Optimizer - Demo Without API Keys
Demonstrates core functionality without requiring LLM API keys.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{title}")
    print("-" * len(title))

def demo_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print_section("Feature Extraction Demo")
    
    try:
        from api.features import FeatureExtractor
        
        # Sample headlines
        headlines = [
            "Get 50% Off Today Only!",
            "Transform Your Life in 30 Days",
            "Free Shipping on All Orders",
            "Limited Time Offer - Don't Miss Out!",
            "Revolutionary New Product Launch"
        ]
        
        print(f"Extracting features from {len(headlines)} headlines...")
        
        # Initialize extractor
        extractor = FeatureExtractor()
        
        # Extract features
        start_time = time.time()
        features_df = extractor.extract_features_batch(headlines)
        end_time = time.time()
        
        print(f"✅ Successfully extracted {features_df.shape[1]} features")
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"📊 Feature columns: {list(features_df.columns[:10])}...")
        
        # Show sample features
        print("\n📈 Sample Features:")
        sample_features = features_df.head(3)[['char_count', 'word_count', 'sentiment_polarity', 'readability_score']]
        print(sample_features.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False

def demo_ctr_simulation():
    """Demonstrate CTR simulation capabilities."""
    print_section("CTR Simulation Demo")
    
    try:
        from models.ctr_simulator import CTRSimulator
        
        # Initialize simulator
        simulator = CTRSimulator()
        
        print("Generating synthetic training data...")
        start_time = time.time()
        
        # Generate data
        data = simulator.generate_training_data(n_samples=1000)
        
        end_time = time.time()
        
        print(f"✅ Generated {data.shape[0]} samples with {data.shape[1]} features")
        print(f"⏱️  Generation time: {end_time - start_time:.2f} seconds")
        
        # Show CTR statistics
        ctr_stats = data['ctr'].describe()
        print(f"\n📊 CTR Statistics:")
        print(f"   Mean CTR: {ctr_stats['mean']:.4f}")
        print(f"   Min CTR:  {ctr_stats['min']:.4f}")
        print(f"   Max CTR:  {ctr_stats['max']:.4f}")
        print(f"   Std CTR:  {ctr_stats['std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ CTR simulation failed: {e}")
        return False

def demo_ab_testing():
    """Demonstrate A/B testing capabilities."""
    print_section("A/B Testing Demo")
    
    try:
        from experiments.ab_harness import ABTestHarness
        
        # Initialize harness
        harness = ABTestHarness()
        
        # Create test configuration
        test_config = harness.create_test(
            test_id="demo_test",
            variant_a="Get 50% Off Today Only!",
            variant_b="Transform Your Life in 30 Days",
            target_impressions=10000
        )
        
        print(f"✅ Created A/B test: {test_config.test_id}")
        print(f"   Variant A: {test_config.variant_a}")
        print(f"   Variant B: {test_config.variant_b}")
        print(f"   Target impressions: {test_config.target_impressions:,}")
        
        # Simulate test results
        print("\nSimulating test results...")
        result = harness.simulate_test(
            test_id="demo_test",
            true_ctr_a=0.05,
            true_ctr_b=0.06
        )
        
        print(f"✅ Test simulation completed:")
        print(f"   Variant A CTR: {result.ctr_a:.3f}")
        print(f"   Variant B CTR: {result.ctr_b:.3f}")
        print(f"   Lift: {result.lift:.1%}")
        print(f"   P-value: {result.p_value:.4f}")
        print(f"   Significant: {'Yes' if result.is_significant else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ A/B testing failed: {e}")
        return False

def demo_content_safety():
    """Demonstrate content safety capabilities."""
    print_section("Content Safety Demo")
    
    try:
        from utils.guardrails import ContentGuardrails
        
        # Initialize guardrails
        guardrails = ContentGuardrails()
        
        # Test headlines (some with issues)
        test_headlines = [
            "Get 50% Off Today Only!",
            "This is a fucking amazing deal!",  # Will be flagged
            "CLICK HERE NOW!!! FREE MONEY!!!",  # Will be flagged
            "Transform Your Life in 30 Days",
            "Limited Time Offer - Don't Miss Out!"
        ]
        
        print(f"Testing {len(test_headlines)} headlines for safety...")
        
        # Check safety
        results = guardrails.batch_filter(test_headlines)
        safe_headlines = guardrails.get_safe_content(test_headlines)
        
        print(f"✅ Safety check completed:")
        print(f"   Total headlines: {len(test_headlines)}")
        print(f"   Safe headlines: {len(safe_headlines)}")
        print(f"   Safety rate: {len(safe_headlines)/len(test_headlines):.1%}")
        
        # Show flagged content
        flagged = [h for h in test_headlines if h not in safe_headlines]
        if flagged:
            print(f"\n⚠️  Flagged content:")
            for headline in flagged:
                print(f"   - '{headline}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Content safety check failed: {e}")
        return False

def demo_diversity_filtering():
    """Demonstrate diversity filtering capabilities."""
    print_section("Diversity Filtering Demo")
    
    try:
        from utils.diversity_filter import DiversityFilter
        
        # Initialize filter
        diversity_filter = DiversityFilter()
        
        # Sample headlines (some similar)
        headlines = [
            "Get 50% Off Today Only!",
            "Get 50% Discount Today Only!",  # Very similar
            "Transform Your Life in 30 Days",
            "Change Your Life in 30 Days",  # Very similar
            "Free Shipping on All Orders",
            "Revolutionary New Product Launch"
        ]
        
        print(f"Filtering {len(headlines)} headlines for diversity...")
        
        # Filter for diversity
        start_time = time.time()
        diverse_headlines = diversity_filter.ensure_diversity(headlines)
        end_time = time.time()
        
        print(f"✅ Diversity filtering completed:")
        print(f"   Original headlines: {len(headlines)}")
        print(f"   Diverse headlines: {len(diverse_headlines)}")
        print(f"   Processing time: {end_time - start_time:.2f} seconds")
        
        print(f"\n📝 Diverse headlines:")
        for i, headline in enumerate(diverse_headlines, 1):
            print(f"   {i}. {headline}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diversity filtering failed: {e}")
        return False

def demo_prompt_templates():
    """Demonstrate prompt template system."""
    print_section("Prompt Templates Demo")
    
    try:
        from prompts.templates import PromptManager
        
        # Initialize prompt manager
        pm = PromptManager()
        
        # Get available templates
        templates = pm.get_available_templates()
        
        print(f"✅ Available prompt templates: {len(templates)}")
        for template in templates:
            print(f"   - {template}")
        
        # Load a specific template
        template = pm.load_template("ecommerce_urgent")
        print(f"\n📝 Sample template (ecommerce_urgent):")
        print(f"   Tone: {template.get('tone', 'N/A')}")
        print(f"   Use case: {template.get('use_case', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt templates failed: {e}")
        return False

def main():
    """Run the complete demo."""
    print_header("Ad Headline Optimizer - Core Functionality Demo")
    print("Demonstrating key features without requiring API keys...")
    
    demos = [
        ("Feature Extraction", demo_feature_extraction),
        ("CTR Simulation", demo_ctr_simulation),
        ("A/B Testing", demo_ab_testing),
        ("Content Safety", demo_content_safety),
        ("Diversity Filtering", demo_diversity_filtering),
        ("Prompt Templates", demo_prompt_templates),
    ]
    
    results = []
    
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} demo failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Demo Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} demos")
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
    
    if passed == total:
        print("\n🎉 All demos passed! The core system is working correctly.")
        print("\nNext steps:")
        print("• Add OpenAI/Anthropic API keys to .env file for LLM generation")
        print("• Train the CTR model: python models/train_ctr_model.py")
        print("• Launch Streamlit UI: streamlit run app/streamlit_app.py")
    else:
        print(f"\n⚠️  {total - passed} demos failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

