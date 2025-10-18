"""
Feature extraction pipeline for headline analysis.
Converts headlines into ML features for CTR prediction.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat
from sentence_transformers import SentenceTransformer
import redis
import json
import hashlib

from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts comprehensive features from headlines for ML models."""
    
    def __init__(self):
        self.sentence_model = None
        self.redis_client = None
        self._init_sentence_model()
        self._init_redis()
        self._init_power_words()
    
    def _init_sentence_model(self):
        """Initialize sentence transformer model."""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
    
    def _init_redis(self):
        """Initialize Redis client for feature caching."""
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established for feature caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Feature caching disabled.")
            self.redis_client = None
    
    def _init_power_words(self):
        """Initialize power word dictionaries."""
        self.action_verbs = {
            'get', 'grab', 'snag', 'score', 'win', 'earn', 'save', 'cut', 'slash',
            'boost', 'increase', 'double', 'triple', 'maximize', 'minimize',
            'discover', 'unlock', 'reveal', 'expose', 'uncover', 'find',
            'build', 'create', 'make', 'design', 'develop', 'launch',
            'start', 'begin', 'stop', 'end', 'finish', 'complete',
            'learn', 'master', 'understand', 'know', 'discover'
        }
        
        self.urgency_words = {
            'now', 'today', 'immediately', 'urgent', 'limited', 'exclusive',
            'only', 'last chance', 'expires', 'deadline', 'rush', 'hurry',
            'quick', 'fast', 'instant', 'immediate', 'asap', 'right now'
        }
        
        self.benefit_words = {
            'free', 'save', 'earn', 'profit', 'money', 'cash', 'bonus',
            'discount', 'deal', 'offer', 'special', 'exclusive', 'premium',
            'guaranteed', 'proven', 'tested', 'results', 'success',
            'easy', 'simple', 'fast', 'quick', 'instant', 'effortless'
        }
        
        self.number_words = {
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            'first', 'second', 'third', 'last', 'final'
        }
    
    def _get_cache_key(self, headline: str) -> str:
        """Generate cache key for headline features."""
        return f"features:{hashlib.md5(headline.encode()).hexdigest()}"
    
    def _get_cached_features(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached features."""
        if not self.redis_client:
            return None
            
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Feature cache hit for: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Feature cache retrieval failed: {e}")
        return None
    
    def _cache_features(self, cache_key: str, features: Dict[str, Any]):
        """Cache extracted features."""
        if not self.redis_client:
            return
            
        try:
            self.redis_client.setex(
                cache_key, 
                settings.feature_cache_ttl, 
                json.dumps(features, default=str)
            )
            logger.debug(f"Cached features for: {cache_key}")
        except Exception as e:
            logger.warning(f"Feature cache storage failed: {e}")
    
    def extract_basic_features(self, headline: str) -> Dict[str, Any]:
        """Extract basic text features."""
        features = {}
        
        # Character and word counts
        features['char_count'] = len(headline)
        features['word_count'] = len(headline.split())
        features['char_count_no_spaces'] = len(headline.replace(' ', ''))
        
        # Punctuation counts
        features['num_digits'] = len(re.findall(r'\d', headline))
        features['num_exclamations'] = headline.count('!')
        features['num_questions'] = headline.count('?')
        features['num_periods'] = headline.count('.')
        features['num_commas'] = headline.count(',')
        features['num_colons'] = headline.count(':')
        features['num_semicolons'] = headline.count(';')
        
        # Case analysis
        features['all_caps_ratio'] = sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0
        features['title_case_ratio'] = sum(1 for c in headline if c.istitle()) / len(headline) if headline else 0
        
        # Special characters
        features['contains_emoji'] = int(bool(re.search(r'[^\w\s]', headline)))
        features['num_emojis'] = len(re.findall(r'[^\w\s]', headline))
        
        return features
    
    def extract_linguistic_features(self, headline: str) -> Dict[str, Any]:
        """Extract linguistic features using TextBlob and textstat."""
        features = {}
        
        try:
            blob = TextBlob(headline)
            
            # Sentiment analysis
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
            
            # Readability scores
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(headline)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(headline)
            features['gunning_fog'] = textstat.gunning_fog(headline)
            features['smog_index'] = textstat.smog_index(headline)
            features['automated_readability_index'] = textstat.automated_readability_index(headline)
            
            # Text complexity
            features['avg_syllables_per_word'] = textstat.avg_syllables_per_word(headline)
            features['avg_sentence_length'] = textstat.avg_sentence_length(headline)
            
        except Exception as e:
            logger.warning(f"Linguistic feature extraction failed: {e}")
            # Set default values
            features.update({
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0,
                'smog_index': 0.0,
                'automated_readability_index': 0.0,
                'avg_syllables_per_word': 0.0,
                'avg_sentence_length': 0.0
            })
        
        return features
    
    def extract_power_word_features(self, headline: str) -> Dict[str, Any]:
        """Extract power word and structure features."""
        features = {}
        headline_lower = headline.lower()
        words = set(headline_lower.split())
        
        # Power word detection
        features['has_action_verb'] = int(bool(words.intersection(self.action_verbs)))
        features['has_urgency_word'] = int(bool(words.intersection(self.urgency_words)))
        features['has_benefit_word'] = int(bool(words.intersection(self.benefit_words)))
        features['has_number_word'] = int(bool(words.intersection(self.number_words)))
        
        # Count power words
        features['num_action_verbs'] = len(words.intersection(self.action_verbs))
        features['num_urgency_words'] = len(words.intersection(self.urgency_words))
        features['num_benefit_words'] = len(words.intersection(self.benefit_words))
        features['num_number_words'] = len(words.intersection(self.number_words))
        
        # Structure features
        features['starts_with_question'] = int(headline.strip().startswith(('What', 'How', 'Why', 'When', 'Where', 'Who')))
        features['starts_with_number'] = int(headline.strip()[0].isdigit() if headline else 0)
        features['ends_with_exclamation'] = int(headline.strip().endswith('!') if headline else 0)
        features['ends_with_question'] = int(headline.strip().endswith('?') if headline else 0)
        
        # Contains specific patterns
        features['contains_question_mark'] = int('?' in headline)
        features['contains_exclamation'] = int('!' in headline)
        features['contains_ellipsis'] = int('...' in headline or '…' in headline)
        features['contains_quotes'] = int('"' in headline or "'" in headline)
        
        return features
    
    def extract_embedding_features(self, headline: str) -> Dict[str, Any]:
        """Extract sentence embedding features."""
        features = {}
        
        try:
            if self.sentence_model:
                embedding = self.sentence_model.encode([headline])[0]
                
                # Store first 10 dimensions as individual features
                for i in range(min(10, len(embedding))):
                    features[f'embedding_dim_{i}'] = float(embedding[i])
                
                # Store embedding statistics
                features['embedding_mean'] = float(np.mean(embedding))
                features['embedding_std'] = float(np.std(embedding))
                features['embedding_max'] = float(np.max(embedding))
                features['embedding_min'] = float(np.min(embedding))
                
                # Store full embedding as a separate field for similarity calculations
                features['full_embedding'] = embedding.tolist()
                
            else:
                # Set default values if model not available
                for i in range(10):
                    features[f'embedding_dim_{i}'] = 0.0
                features.update({
                    'embedding_mean': 0.0,
                    'embedding_std': 0.0,
                    'embedding_max': 0.0,
                    'embedding_min': 0.0,
                    'full_embedding': [0.0] * 384
                })
                
        except Exception as e:
            logger.warning(f"Embedding feature extraction failed: {e}")
            # Set default values
            for i in range(10):
                features[f'embedding_dim_{i}'] = 0.0
            features.update({
                'embedding_mean': 0.0,
                'embedding_std': 0.0,
                'embedding_max': 0.0,
                'embedding_min': 0.0,
                'full_embedding': [0.0] * 384
            })
        
        return features
    
    def extract_contextual_features(self, headline: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract contextual features based on placement and audience."""
        features = {}
        
        if context:
            # Placement type features
            placement = context.get('placement_type', 'Email Subject')
            features['is_email_subject'] = int(placement == 'Email Subject')
            features['is_social_ad'] = int(placement == 'Social Ad')
            features['is_display_ad'] = int(placement == 'Display Ad')
            
            # Audience segment features
            audience = context.get('audience_segment', 'general')
            features['is_b2b_audience'] = int(audience in ['business', 'enterprise', 'professional'])
            features['is_b2c_audience'] = int(audience in ['consumer', 'retail', 'personal'])
            features['is_tech_audience'] = int(audience in ['tech', 'developer', 'engineer'])
        else:
            # Default values
            features.update({
                'is_email_subject': 1,
                'is_social_ad': 0,
                'is_display_ad': 0,
                'is_b2b_audience': 0,
                'is_b2c_audience': 1,
                'is_tech_audience': 0
            })
        
        return features
    
    def extract_features(self, headline: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract all features for a headline.
        
        Args:
            headline: The headline text
            context: Optional context information (placement_type, audience_segment)
            
        Returns:
            Dictionary of extracted features
        """
        # Check cache first
        cache_key = self._get_cache_key(headline)
        cached_features = self._get_cached_features(cache_key)
        
        if cached_features:
            # Add contextual features if not cached
            if context:
                contextual_features = self.extract_contextual_features(headline, context)
                cached_features.update(contextual_features)
            return cached_features
        
        # Extract all features
        features = {}
        
        # Basic features
        features.update(self.extract_basic_features(headline))
        
        # Linguistic features
        features.update(self.extract_linguistic_features(headline))
        
        # Power word features
        features.update(self.extract_power_word_features(headline))
        
        # Embedding features
        features.update(self.extract_embedding_features(headline))
        
        # Contextual features
        features.update(self.extract_contextual_features(headline, context))
        
        # Cache features (without contextual features for reusability)
        features_to_cache = {k: v for k, v in features.items() 
                           if not k.startswith(('is_', 'placement_', 'audience_'))}
        self._cache_features(cache_key, features_to_cache)
        
        return features
    
    def extract_features_batch(self, headlines: List[str], context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Extract features for multiple headlines.
        
        Args:
            headlines: List of headline texts
            context: Optional context information
            
        Returns:
            DataFrame with features for each headline
        """
        features_list = []
        
        for headline in headlines:
            try:
                features = self.extract_features(headline, context)
                features['headline'] = headline
                features_list.append(features)
            except Exception as e:
                logger.error(f"Feature extraction failed for headline '{headline}': {e}")
                # Add default features for failed extraction
                default_features = self._get_default_features()
                default_features['headline'] = headline
                features_list.append(default_features)
        
        df = pd.DataFrame(features_list)
        
        # Ensure consistent column order
        feature_columns = [col for col in df.columns if col != 'headline']
        df = df[['headline'] + sorted(feature_columns)]
        
        return df
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default feature values for error cases."""
        return {
            'char_count': 0,
            'word_count': 0,
            'char_count_no_spaces': 0,
            'num_digits': 0,
            'num_exclamations': 0,
            'num_questions': 0,
            'num_periods': 0,
            'num_commas': 0,
            'num_colons': 0,
            'num_semicolons': 0,
            'all_caps_ratio': 0.0,
            'title_case_ratio': 0.0,
            'contains_emoji': 0,
            'num_emojis': 0,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'gunning_fog': 0.0,
            'smog_index': 0.0,
            'automated_readability_index': 0.0,
            'avg_syllables_per_word': 0.0,
            'avg_sentence_length': 0.0,
            'has_action_verb': 0,
            'has_urgency_word': 0,
            'has_benefit_word': 0,
            'has_number_word': 0,
            'num_action_verbs': 0,
            'num_urgency_words': 0,
            'num_benefit_words': 0,
            'num_number_words': 0,
            'starts_with_question': 0,
            'starts_with_number': 0,
            'ends_with_exclamation': 0,
            'ends_with_question': 0,
            'contains_question_mark': 0,
            'contains_exclamation': 0,
            'contains_ellipsis': 0,
            'contains_quotes': 0,
            'is_email_subject': 1,
            'is_social_ad': 0,
            'is_display_ad': 0,
            'is_b2b_audience': 0,
            'is_b2c_audience': 1,
            'is_tech_audience': 0
        }


# Global feature extractor instance
feature_extractor = FeatureExtractor()


def extract_headline_features(headline: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for single headline feature extraction.
    
    Args:
        headline: The headline text
        context: Optional context information
        
    Returns:
        Dictionary of extracted features
    """
    return feature_extractor.extract_features(headline, context)


def extract_headlines_features(headlines: List[str], context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Convenience function for batch headline feature extraction.
    
    Args:
        headlines: List of headline texts
        context: Optional context information
        
    Returns:
        DataFrame with features for each headline
    """
    return feature_extractor.extract_features_batch(headlines, context)
