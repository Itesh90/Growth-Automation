"""
Simple tests that work with the actual API structure.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestBasicFunctionality:
    """Test basic functionality without complex mocking."""
    
    def test_imports(self):
        """Test that all modules can be imported."""
        from api.generator import HeadlineRequest, HeadlineResult
        from api.features_lightweight import extract_headlines_features_lightweight
        from config import settings
        
        assert HeadlineRequest is not None
        assert HeadlineResult is not None
        assert extract_headlines_features_lightweight is not None
        assert settings is not None
    
    def test_headline_request_validation(self):
        """Test HeadlineRequest validation."""
        from api.generator import HeadlineRequest
        
        # Valid request
        request = HeadlineRequest(
            base_copy="Amazing new product for tech enthusiasts - get 50% off today!",
            tone="Professional",
            placement_type="Email Subject",
            n_variants=5
        )
        assert request.base_copy == "Amazing new product for tech enthusiasts - get 50% off today!"
        assert request.tone == "Professional"
        assert request.placement_type == "Email Subject"
        assert request.n_variants == 5
    
    def test_headline_request_invalid_tone(self):
        """Test HeadlineRequest with invalid tone."""
        from api.generator import HeadlineRequest
        
        with pytest.raises(ValueError):
            HeadlineRequest(
                base_copy="Test product description",
                tone="InvalidTone",
                placement_type="Email Subject"
            )
    
    def test_headline_request_invalid_placement(self):
        """Test HeadlineRequest with invalid placement type."""
        from api.generator import HeadlineRequest
        
        with pytest.raises(ValueError):
            HeadlineRequest(
                base_copy="Test product description",
                tone="Professional",
                placement_type="InvalidPlacement"
            )
    
    def test_headline_request_short_copy(self):
        """Test HeadlineRequest with short base copy."""
        from api.generator import HeadlineRequest
        
        with pytest.raises(ValueError):
            HeadlineRequest(
                base_copy="Short",
                tone="Professional",
                placement_type="Email Subject"
            )
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        from api.features_lightweight import extract_headlines_features_lightweight
        
        headlines = [
            "Amazing new product - 50% off today!",
            "Get the best deal on our premium service",
            "Limited time offer - don't miss out!"
        ]
        
        features = extract_headlines_features_lightweight(headlines)
        
        # Check that we get a DataFrame with the right number of rows
        assert len(features) == len(headlines)
        
        # Check that required columns exist
        required_columns = ['char_count', 'word_count', 'has_emoji', 'has_numbers', 
                          'has_exclamation', 'has_question', 'power_word_count']
        
        for col in required_columns:
            assert col in features.columns
    
    def test_feature_extraction_empty(self):
        """Test feature extraction with empty list."""
        from api.features_lightweight import extract_headlines_features_lightweight
        
        features = extract_headlines_features_lightweight([])
        assert features.empty
    
    def test_config_loading(self):
        """Test that configuration loads properly."""
        from config import settings
        
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'app_version')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'log_level')
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'anthropic_api_key')
        assert hasattr(settings, 'gemini_api_key')
        assert hasattr(settings, 'gemini_model')
    
    def test_headline_result_structure(self):
        """Test HeadlineResult structure."""
        from api.generator import HeadlineResult
        
        result = HeadlineResult(
            headline="Test headline",
            reasoning="Test reasoning",
            char_count=12,
            generation_time=1.0,
            cost_usd=0.01,
            model_used="test-model"
        )
        
        assert result.headline == "Test headline"
        assert result.reasoning == "Test reasoning"
        assert result.char_count == 12
        assert result.generation_time == 1.0
        assert result.cost_usd == 0.01
        assert result.model_used == "test-model"
