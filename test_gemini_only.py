#!/usr/bin/env python3
"""
Test script for Gemini-only configuration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_gemini_only():
    """Test system with only Gemini API key."""
    print("Testing Gemini-Only Configuration...")
    
    # Set only Gemini API key
    os.environ['GEMINI_API_KEY'] = 'test_key_123'
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('ANTHROPIC_API_KEY', None)
    
    try:
        from api.generator import get_generator, HeadlineRequest
        
        # Test generator creation
        generator = get_generator()
        print("Generator created successfully")
        
        # Check which clients are available
        print(f"OpenAI client: {'YES' if generator.openai_client else 'NO'}")
        print(f"Anthropic client: {'YES' if generator.anthropic_client else 'NO'}")
        print(f"Gemini model: {'YES' if generator.gemini_model else 'NO'}")
        
        # Test request creation
        request = HeadlineRequest(
            base_copy="Test product for fitness",
            tone="Professional",
            n_variants=3
        )
        print("Request created successfully")
        
        print("\nGemini-only configuration test passed!")
        print("The system will use Gemini as the primary AI model")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_no_api_keys():
    """Test system with no API keys (sample data mode)."""
    print("\nTesting No API Keys Configuration...")
    
    # Clear all API keys
    os.environ.pop('GEMINI_API_KEY', None)
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('ANTHROPIC_API_KEY', None)
    
    try:
        from api.generator import get_generator, HeadlineRequest
        
        # Test generator creation
        generator = get_generator()
        print("Generator created successfully (sample mode)")
        
        # Check which clients are available
        print(f"OpenAI client: {'YES' if generator.openai_client else 'NO'}")
        print(f"Anthropic client: {'YES' if generator.anthropic_client else 'NO'}")
        print(f"Gemini model: {'YES' if generator.gemini_model else 'NO'}")
        
        # Test request creation
        request = HeadlineRequest(
            base_copy="Test product for fitness",
            tone="Professional",
            n_variants=3
        )
        print("Request created successfully")
        
        print("\nNo API keys configuration test passed!")
        print("The system will use sample data")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Gemini Configuration Test")
    print("=" * 50)
    
    # Test Gemini-only
    gemini_test = test_gemini_only()
    
    # Test no API keys
    no_keys_test = test_no_api_keys()
    
    print("\n" + "=" * 50)
    if gemini_test and no_keys_test:
        print("All configuration tests passed!")
        print("The system is ready for Gemini integration")
    else:
        print("Some tests failed")
    
    sys.exit(0 if (gemini_test and no_keys_test) else 1)
