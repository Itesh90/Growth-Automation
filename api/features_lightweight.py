"""
Lightweight feature extraction for faster performance.
Uses simple heuristics instead of heavy ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
from textblob import TextBlob


def extract_headlines_features_lightweight(headlines: List[str], context: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Extract features using lightweight methods for faster performance.
    
    Args:
        headlines: List of headline texts
        context: Additional context (placement type, etc.)
        
    Returns:
        DataFrame with extracted features
    """
    if context is None:
        context = {}
    
    features = []
    
    for headline in headlines:
        # Basic text features
        char_count = len(headline)
        word_count = len(headline.split())
        
        # Sentiment analysis (lightweight)
        try:
            blob = TextBlob(headline)
            sentiment_score = blob.sentiment.polarity
            sentiment_label = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        except:
            sentiment_score = 0.0
            sentiment_label = "neutral"
        
        # Readability features
        avg_word_length = np.mean([len(word) for word in headline.split()]) if word_count > 0 else 0
        sentence_count = len(re.split(r'[.!?]+', headline))
        
        # Engagement features
        has_question = '?' in headline
        has_exclamation = '!' in headline
        has_numbers = bool(re.search(r'\d', headline))
        has_emoji = bool(re.search(r'[^\w\s]', headline))
        
        # Power words (common marketing terms)
        power_words = ['free', 'new', 'best', 'top', 'exclusive', 'limited', 'urgent', 'secret', 'proven', 'guaranteed']
        power_word_count = sum(1 for word in power_words if word.lower() in headline.lower())
        
        # Placement-specific features
        placement_type = context.get('placement_type', 'Email Subject')
        is_email = placement_type == 'Email Subject'
        is_social = placement_type == 'Social Ad'
        is_display = placement_type == 'Display Ad'
        
        # Length appropriateness
        if is_email:
            length_score = 1.0 if char_count <= 50 else 0.8 if char_count <= 70 else 0.5
        elif is_social:
            length_score = 1.0 if char_count <= 100 else 0.8 if char_count <= 150 else 0.5
        else:  # display
            length_score = 1.0 if char_count <= 60 else 0.8 if char_count <= 80 else 0.5
        
        features.append({
            'headline': headline,
            'char_count': char_count,
            'word_count': word_count,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'has_question': has_question,
            'has_exclamation': has_exclamation,
            'has_numbers': has_numbers,
            'has_emoji': has_emoji,
            'power_word_count': power_word_count,
            'length_score': length_score,
            'is_email': is_email,
            'is_social': is_social,
            'is_display': is_display
        })
    
    return pd.DataFrame(features)


def calculate_predicted_ctr_lightweight(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate predicted CTR using lightweight heuristics.
    
    Args:
        features_df: DataFrame with extracted features
        
    Returns:
        DataFrame with predicted CTR added
    """
    predicted_ctr = []
    
    for _, row in features_df.iterrows():
        # Base CTR
        ctr = 0.02
        
        # Length factor
        ctr += row['length_score'] * 0.01
        
        # Sentiment factor
        if row['sentiment_label'] == 'positive':
            ctr += 0.005
        elif row['sentiment_label'] == 'negative':
            ctr -= 0.003
        
        # Engagement factors
        if row['has_question']:
            ctr += 0.003
        if row['has_exclamation']:
            ctr += 0.002
        if row['has_numbers']:
            ctr += 0.002
        if row['power_word_count'] > 0:
            ctr += row['power_word_count'] * 0.001
        
        # Word count factor
        if 5 <= row['word_count'] <= 10:
            ctr += 0.002
        elif row['word_count'] > 15:
            ctr -= 0.002
        
        # Ensure reasonable bounds
        ctr = max(0.005, min(0.15, ctr))
        
        predicted_ctr.append(ctr)
    
    features_df['predicted_ctr'] = predicted_ctr
    features_df['predicted_ctr_pct'] = [ctr * 100 for ctr in predicted_ctr]
    
    return features_df
