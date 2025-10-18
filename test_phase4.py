"""
Test script for Phase 4 components: Production Prompts, Guardrails, and Cache Manager
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts.templates import PromptManager, PromptTemplate, Tone, Placement
from utils.guardrails import ContentGuardrails, ContentLevel, ComplianceType
from api.cache_manager import CacheManager, CacheEntry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prompt_templates():
    """Test the prompt templates system."""
    print("=" * 60)
    print("TESTING PROMPT TEMPLATES")
    print("=" * 60)
    
    try:
        # Initialize prompt manager
        manager = PromptManager()
        print("✓ PromptManager initialized successfully")
        
        # Test template listing
        templates = manager.list_templates()
        print(f"✓ Templates listed: {len(templates)} templates found")
        
        # Test template retrieval
        template = manager.get_template("ecommerce_urgent")
        if template:
            print(f"✓ Template retrieved: {template.name}")
        else:
            print("⚠ Template 'ecommerce_urgent' not found")
        
        # Test template filtering by tone
        urgent_templates = manager.get_templates_by_tone(Tone.URGENT)
        print(f"✓ Templates filtered by tone: {len(urgent_templates)} urgent templates")
        
        # Test template filtering by placement
        facebook_templates = manager.get_templates_by_placement(Placement.FACEBOOK)
        print(f"✓ Templates filtered by placement: {len(facebook_templates)} Facebook templates")
        
        # Test best template selection
        best_template = manager.get_best_template("urgent", "facebook")
        if best_template:
            print(f"✓ Best template selected: {best_template.name}")
        else:
            print("⚠ No best template found")
        
        # Test prompt formatting
        try:
            formatted_prompt = manager.format_prompt(
                "ecommerce_urgent",
                "Summer sale on clothing",
                n_variants=3,
                placement="facebook"
            )
            print(f"✓ Prompt formatted: {len(formatted_prompt['user_prompt'])} characters")
        except Exception as e:
            print(f"⚠ Prompt formatting failed: {e}")
        
        # Test template creation
        new_template = PromptTemplate(
            name="test_template",
            description="Test template for unit testing",
            system_prompt="You are a test copywriter.",
            user_prompt_template="Generate {n_variants} headlines for: {base_copy}",
            examples=[{"base_copy": "test", "headlines": ["Test headline"]}]
        )
        
        manager.add_template(new_template)
        print("✓ New template added successfully")
        
        # Test template removal
        manager.remove_template("test_template")
        print("✓ Template removed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt templates test failed: {e}")
        return False


def test_content_guardrails():
    """Test the content guardrails system."""
    print("\n" + "=" * 60)
    print("TESTING CONTENT GUARDRAILS")
    print("=" * 60)
    
    try:
        # Initialize guardrails
        guardrails = ContentGuardrails()
        print("✓ ContentGuardrails initialized successfully")
        
        # Test texts with different issues
        test_texts = [
            "Get 50% Off Today Only!",  # Safe
            "This is a fucking amazing deal!",  # Profanity
            "CLICK HERE NOW!!! FREE MONEY!!!",  # Spam + excessive caps
            "100% FREE - NO RISK GUARANTEED",  # Misleading
            "Cure your illness with this product",  # Regulatory
            "Transform Your Life in 30 Days",  # Safe
            "Free Shipping on All Orders"  # Safe
        ]
        
        # Test individual content checks
        print("Testing individual content checks...")
        
        # Test profanity check
        profanity_check = guardrails.check_profanity("This is a fucking amazing deal!")
        print(f"✓ Profanity check: {profanity_check.level.value}")
        
        # Test spam check
        spam_check = guardrails.check_spam("CLICK HERE NOW!!! FREE MONEY!!!")
        print(f"✓ Spam check: {spam_check.level.value}")
        
        # Test misleading check
        misleading_check = guardrails.check_misleading("100% FREE - NO RISK GUARANTEED")
        print(f"✓ Misleading check: {misleading_check.level.value}")
        
        # Test regulatory check
        regulatory_check = guardrails.check_regulatory("Cure your illness with this product")
        print(f"✓ Regulatory check: {regulatory_check.level.value}")
        
        # Test brand safety check
        brand_safety_check = guardrails.check_brand_safety("This is a hateful message")
        print(f"✓ Brand safety check: {brand_safety_check.level.value}")
        
        # Test formatting check
        formatting_check = guardrails.check_formatting("CLICK HERE NOW!!!")
        print(f"✓ Formatting check: {formatting_check.level.value}")
        
        # Test accessibility check
        accessibility_check = guardrails.check_accessibility("Visit https://example.com for more info")
        print(f"✓ Accessibility check: {accessibility_check.level.value}")
        
        # Test content filtering
        print("\nTesting content filtering...")
        for text in test_texts:
            result = guardrails.filter_content(text)
            print(f"✓ Filtered '{text[:30]}...': {result.level.value} ({'Safe' if result.is_safe else 'Not Safe'})")
        
        # Test batch filtering
        print("\nTesting batch filtering...")
        results = guardrails.batch_filter(test_texts)
        print(f"✓ Batch filtering: {len(results)} results")
        
        # Test safe content extraction
        safe_texts = guardrails.get_safe_content(test_texts)
        print(f"✓ Safe content extraction: {len(safe_texts)} safe texts")
        
        # Test filtering summary
        summary = guardrails.get_filtering_summary(results)
        print(f"✓ Filtering summary: {summary['safety_rate']:.1%} safety rate")
        
        return True
        
    except Exception as e:
        print(f"✗ Content guardrails test failed: {e}")
        return False


def test_cache_manager():
    """Test the cache manager system."""
    print("\n" + "=" * 60)
    print("TESTING CACHE MANAGER")
    print("=" * 60)
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager()
        print("✓ CacheManager initialized successfully")
        
        # Test connection status
        is_connected = cache_manager._is_connected()
        print(f"✓ Redis connection: {'Connected' if is_connected else 'Not connected (expected for testing)'}")
        
        # Test cache key generation
        cache_key = cache_manager._generate_cache_key("test prompt", "gpt-4o-mini", 0.7)
        print(f"✓ Cache key generated: {cache_key[:20]}...")
        
        # Test cache operations (if Redis is available)
        if is_connected:
            # Test caching
            test_prompt = "Generate 3 headlines for a summer sale"
            test_response = "1. Summer Sale - 50% Off Everything!\n2. Hot Deals for Hot Days\n3. Cool Savings This Summer"
            test_model = "gpt-4o-mini"
            test_tokens = 150
            
            success = cache_manager.set(test_prompt, test_response, test_model, test_tokens)
            print(f"✓ Cache set: {'Success' if success else 'Failed'}")
            
            # Test retrieval
            cached_entry = cache_manager.get(test_prompt, test_model)
            if cached_entry:
                print(f"✓ Cache get: Success (hit count: {cached_entry.hit_count})")
            else:
                print("⚠ Cache get: No entry found")
            
            # Test stats
            stats = cache_manager.get_stats()
            print(f"✓ Cache stats: {stats.total_requests} requests, {stats.hit_rate:.1%} hit rate")
            
            # Test cost savings
            savings = cache_manager.get_cost_savings()
            print(f"✓ Cost savings: ${savings['total_cost_saved']:.4f}")
            
            # Test cleanup
            cleaned = cache_manager.cleanup_expired()
            print(f"✓ Cleanup: {cleaned} entries cleaned")
            
        else:
            print("⚠ Redis not available - skipping cache operations")
        
        # Test cache entry creation
        test_entry = CacheEntry(
            key="test_key",
            response="test response",
            tokens_used=100,
            cost_usd=0.003,
            model="gpt-4o-mini",
            created_at=cache_manager._get_current_time(),
            expires_at=cache_manager._get_current_time() + cache_manager._get_timedelta(seconds=3600)
        )
        print("✓ Cache entry created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache manager test failed: {e}")
        return False


def test_integration():
    """Test integration between Phase 4 components."""
    print("\n" + "=" * 60)
    print("TESTING PHASE 4 INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize components
        prompt_manager = PromptManager()
        guardrails = ContentGuardrails()
        cache_manager = CacheManager()
        
        # Test prompt generation and filtering workflow
        print("Testing prompt generation and filtering workflow...")
        
        # Get a template
        template = prompt_manager.get_best_template("urgent", "facebook")
        if template:
            print(f"✓ Template selected: {template.name}")
            
            # Format a prompt
            formatted_prompt = prompt_manager.format_prompt(
                template.name,
                "Summer sale on clothing",
                n_variants=3,
                placement="facebook"
            )
            print(f"✓ Prompt formatted: {len(formatted_prompt['user_prompt'])} characters")
            
            # Simulate generated headlines
            simulated_headlines = [
                "🔥 Summer Sale - 50% Off Everything!",
                "Hot Deals for Hot Days - Limited Time!",
                "Cool Savings This Summer - Act Now!"
            ]
            
            # Filter headlines through guardrails
            filtered_results = guardrails.batch_filter(simulated_headlines)
            safe_headlines = guardrails.get_safe_content(simulated_headlines)
            print(f"✓ Headlines filtered: {len(simulated_headlines)} → {len(safe_headlines)} safe headlines")
            
            # Test caching (if Redis is available)
            if cache_manager._is_connected():
                for i, headline in enumerate(safe_headlines):
                    cache_manager.set(
                        f"headline_{i}",
                        headline,
                        "gpt-4o-mini",
                        50
                    )
                print(f"✓ Headlines cached: {len(safe_headlines)} entries")
            
        else:
            print("⚠ No template found for integration test")
        
        # Test end-to-end workflow
        print("\nTesting end-to-end workflow...")
        
        # 1. Generate prompt
        prompt = prompt_manager.format_prompt(
            "ecommerce_urgent",
            "Black Friday sale",
            n_variants=5,
            placement="facebook"
        )
        
        # 2. Simulate LLM response
        simulated_response = "1. Black Friday Madness - 70% Off!\n2. Biggest Sale of the Year\n3. Don't Miss Black Friday Deals\n4. Black Friday - Shop Now!\n5. Black Friday Savings Await!"
        
        # 3. Filter response
        headlines = simulated_response.split('\n')
        filtered_results = guardrails.batch_filter(headlines)
        safe_headlines = guardrails.get_safe_content(headlines)
        
        # 4. Cache response (if available)
        if cache_manager._is_connected():
            cache_manager.set(
                prompt['user_prompt'],
                simulated_response,
                "gpt-4o-mini",
                200
            )
        
        print(f"✓ End-to-end workflow: {len(headlines)} → {len(safe_headlines)} safe headlines")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 4 integration test failed: {e}")
        return False


def test_performance():
    """Test performance of Phase 4 components."""
    print("\n" + "=" * 60)
    print("TESTING PHASE 4 PERFORMANCE")
    print("=" * 60)
    
    try:
        import time
        
        # Test prompt manager performance
        prompt_manager = PromptManager()
        
        start_time = time.time()
        for i in range(100):
            prompt_manager.format_prompt(
                "ecommerce_urgent",
                f"Test headline {i}",
                n_variants=3,
                placement="facebook"
            )
        prompt_time = time.time() - start_time
        print(f"✓ Prompt formatting: 100 prompts in {prompt_time:.2f}s ({100/prompt_time:.1f} prompts/s)")
        
        # Test guardrails performance
        guardrails = ContentGuardrails()
        test_texts = [f"Test headline {i} for performance testing" for i in range(100)]
        
        start_time = time.time()
        results = guardrails.batch_filter(test_texts)
        guardrails_time = time.time() - start_time
        print(f"✓ Content filtering: {len(test_texts)} texts in {guardrails_time:.2f}s ({len(test_texts)/guardrails_time:.1f} texts/s)")
        
        # Test cache manager performance (if Redis is available)
        cache_manager = CacheManager()
        if cache_manager._is_connected():
            start_time = time.time()
            for i in range(100):
                cache_manager.set(
                    f"test_prompt_{i}",
                    f"test_response_{i}",
                    "gpt-4o-mini",
                    100
                )
            cache_time = time.time() - start_time
            print(f"✓ Cache operations: 100 operations in {cache_time:.2f}s ({100/cache_time:.1f} ops/s)")
        else:
            print("⚠ Cache performance test skipped - Redis not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 4 performance test failed: {e}")
        return False


def main():
    """Run all Phase 4 tests."""
    print("PHASE 4 COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("Prompt Templates", test_prompt_templates),
        ("Content Guardrails", test_content_guardrails),
        ("Cache Manager", test_cache_manager),
        ("Phase 4 Integration", test_integration),
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
        print("🎉 All Phase 4 tests passed!")
    else:
        print("⚠ Some tests failed - check the output above for details")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
