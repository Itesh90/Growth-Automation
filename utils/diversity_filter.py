"""
Diversity Filter
Ensures headline variants are diverse and not too similar to each other.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re
from collections import Counter

from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class DiversityFilter:
    """Filters headlines to ensure diversity and quality."""
    
    def __init__(self, similarity_threshold: float = 0.8, 
                 min_diversity_score: float = 0.3):
        """
        Initialize the diversity filter.
        
        Args:
            similarity_threshold: Maximum similarity between headlines (0-1)
            min_diversity_score: Minimum diversity score for headline sets (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.min_diversity_score = min_diversity_score
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Common words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
    
    def calculate_semantic_similarity(self, headlines: List[str]) -> np.ndarray:
        """
        Calculate semantic similarity between headlines using embeddings.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Similarity matrix (n x n)
        """
        if len(headlines) < 2:
            return np.array([[1.0]])
        
        try:
            # Get embeddings
            embeddings = self.embedding_model.encode(headlines)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            # Fallback to TF-IDF similarity
            return self.calculate_tfidf_similarity(headlines)
    
    def calculate_tfidf_similarity(self, headlines: List[str]) -> np.ndarray:
        """
        Calculate similarity using TF-IDF vectors.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Similarity matrix (n x n)
        """
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(headlines)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            # Return identity matrix as fallback
            n = len(headlines)
            return np.eye(n)
    
    def calculate_lexical_similarity(self, headlines: List[str]) -> np.ndarray:
        """
        Calculate lexical similarity based on word overlap.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Similarity matrix (n x n)
        """
        n = len(headlines)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Tokenize and normalize
                    words_i = set(self._tokenize(headlines[i]))
                    words_j = set(self._tokenize(headlines[j]))
                    
                    # Calculate Jaccard similarity
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    
                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
                    else:
                        similarity_matrix[i, j] = 0.0
        
        return similarity_matrix
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words, removing stop words and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            List of normalized words
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove stop words and short words
        words = [word for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return words
    
    def calculate_structural_similarity(self, headlines: List[str]) -> np.ndarray:
        """
        Calculate structural similarity based on length, punctuation, etc.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Similarity matrix (n x n)
        """
        n = len(headlines)
        similarity_matrix = np.zeros((n, n))
        
        # Extract structural features
        features = []
        for headline in headlines:
            feature = {
                'length': len(headline),
                'word_count': len(headline.split()),
                'exclamation_count': headline.count('!'),
                'question_count': headline.count('?'),
                'uppercase_ratio': sum(1 for c in headline if c.isupper()) / len(headline) if headline else 0,
                'digit_count': sum(1 for c in headline if c.isdigit()),
                'special_char_count': sum(1 for c in headline if c in '!@#$%^&*()_+-=[]{}|;:,.<>?')
            }
            features.append(feature)
        
        # Calculate similarity based on features
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate normalized difference for each feature
                    similarities = []
                    for key in features[i]:
                        val_i = features[i][key]
                        val_j = features[j][key]
                        
                        # Normalize by range
                        max_val = max(val_i, val_j)
                        if max_val > 0:
                            similarity = 1 - abs(val_i - val_j) / max_val
                        else:
                            similarity = 1.0
                        
                        similarities.append(similarity)
                    
                    # Average similarity
                    similarity_matrix[i, j] = np.mean(similarities)
        
        return similarity_matrix
    
    def calculate_overall_similarity(self, headlines: List[str]) -> np.ndarray:
        """
        Calculate overall similarity combining multiple methods.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Combined similarity matrix (n x n)
        """
        # Calculate different types of similarity
        semantic_sim = self.calculate_semantic_similarity(headlines)
        lexical_sim = self.calculate_lexical_similarity(headlines)
        structural_sim = self.calculate_structural_similarity(headlines)
        
        # Weighted combination
        weights = {
            'semantic': 0.5,
            'lexical': 0.3,
            'structural': 0.2
        }
        
        overall_sim = (weights['semantic'] * semantic_sim + 
                      weights['lexical'] * lexical_sim + 
                      weights['structural'] * structural_sim)
        
        return overall_sim
    
    def filter_similar_headlines(self, headlines: List[str], 
                               similarity_threshold: Optional[float] = None) -> List[str]:
        """
        Filter out headlines that are too similar to each other.
        
        Args:
            headlines: List of headline strings
            similarity_threshold: Maximum allowed similarity (overrides default)
            
        Returns:
            List of filtered headlines
        """
        if len(headlines) <= 1:
            return headlines
        
        threshold = similarity_threshold or self.similarity_threshold
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_overall_similarity(headlines)
        
        # Find headlines to keep
        keep_indices = []
        n = len(headlines)
        
        for i in range(n):
            is_similar = False
            
            # Check if this headline is too similar to any already kept headline
            for j in keep_indices:
                if similarity_matrix[i, j] > threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                keep_indices.append(i)
        
        # Return filtered headlines
        filtered_headlines = [headlines[i] for i in keep_indices]
        
        logger.info(f"Filtered {len(headlines)} headlines to {len(filtered_headlines)} (removed {len(headlines) - len(filtered_headlines)} similar)")
        
        return filtered_headlines
    
    def calculate_diversity_score(self, headlines: List[str]) -> float:
        """
        Calculate overall diversity score for a set of headlines.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(headlines) <= 1:
            return 1.0
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_overall_similarity(headlines)
        
        # Calculate average similarity (excluding diagonal)
        n = len(headlines)
        total_similarity = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_similarity += similarity_matrix[i, j]
                count += 1
        
        if count == 0:
            return 1.0
        
        average_similarity = total_similarity / count
        
        # Convert to diversity score (1 - average_similarity)
        diversity_score = 1 - average_similarity
        
        return diversity_score
    
    def ensure_diversity(self, headlines: List[str], 
                        min_diversity_score: Optional[float] = None) -> List[str]:
        """
        Ensure headlines meet minimum diversity requirements.
        
        Args:
            headlines: List of headline strings
            min_diversity_score: Minimum required diversity score
            
        Returns:
            List of headlines that meet diversity requirements
        """
        if len(headlines) <= 1:
            return headlines
        
        min_score = min_diversity_score or self.min_diversity_score
        
        # Calculate current diversity score
        current_diversity = self.calculate_diversity_score(headlines)
        
        if current_diversity >= min_score:
            logger.info(f"Diversity score {current_diversity:.3f} meets requirement {min_score:.3f}")
            return headlines
        
        logger.info(f"Diversity score {current_diversity:.3f} below requirement {min_score:.3f}, filtering...")
        
        # Filter similar headlines
        filtered_headlines = self.filter_similar_headlines(headlines)
        
        # Check if we still have enough headlines
        if len(filtered_headlines) < len(headlines) * 0.5:
            logger.warning("Filtering removed too many headlines, relaxing threshold")
            # Try with a higher threshold
            filtered_headlines = self.filter_similar_headlines(headlines, similarity_threshold=0.9)
        
        return filtered_headlines
    
    def analyze_diversity(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyze diversity of a set of headlines.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            Dictionary with diversity analysis
        """
        if len(headlines) <= 1:
            return {
                'diversity_score': 1.0,
                'similarity_matrix': np.array([[1.0]]),
                'analysis': 'Single headline - perfect diversity'
            }
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_overall_similarity(headlines)
        
        # Calculate diversity score
        diversity_score = self.calculate_diversity_score(headlines)
        
        # Find most similar pair
        max_similarity = 0
        most_similar_pair = None
        
        for i in range(len(headlines)):
            for j in range(i + 1, len(headlines)):
                if similarity_matrix[i, j] > max_similarity:
                    max_similarity = similarity_matrix[i, j]
                    most_similar_pair = (i, j)
        
        # Calculate statistics
        similarities = []
        for i in range(len(headlines)):
            for j in range(i + 1, len(headlines)):
                similarities.append(similarity_matrix[i, j])
        
        analysis = {
            'diversity_score': diversity_score,
            'similarity_matrix': similarity_matrix,
            'max_similarity': max_similarity,
            'most_similar_pair': most_similar_pair,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'headline_count': len(headlines),
            'meets_diversity_threshold': diversity_score >= self.min_diversity_score
        }
        
        if most_similar_pair:
            analysis['most_similar_headlines'] = [
                headlines[most_similar_pair[0]],
                headlines[most_similar_pair[1]]
            ]
        
        return analysis
    
    def get_diversity_recommendations(self, headlines: List[str]) -> List[str]:
        """
        Get recommendations for improving diversity.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze diversity
        analysis = self.analyze_diversity(headlines)
        
        if analysis['diversity_score'] < self.min_diversity_score:
            recommendations.append(f"Diversity score {analysis['diversity_score']:.3f} is below threshold {self.min_diversity_score:.3f}")
        
        if analysis['max_similarity'] > self.similarity_threshold:
            recommendations.append(f"Most similar headlines have {analysis['max_similarity']:.3f} similarity (threshold: {self.similarity_threshold:.3f})")
        
        # Check for common patterns
        word_counts = Counter()
        for headline in headlines:
            words = self._tokenize(headline)
            word_counts.update(words)
        
        # Find overused words
        overused_words = [word for word, count in word_counts.items() 
                         if count > len(headlines) * 0.5]
        
        if overused_words:
            recommendations.append(f"Overused words: {', '.join(overused_words[:5])}")
        
        # Check length diversity
        lengths = [len(headline) for headline in headlines]
        length_std = np.std(lengths)
        if length_std < 10:
            recommendations.append("Headlines have similar lengths - try varying length")
        
        # Check punctuation diversity
        exclamation_count = sum(1 for headline in headlines if '!' in headline)
        if exclamation_count > len(headlines) * 0.7:
            recommendations.append("Too many headlines use exclamation marks")
        
        return recommendations


def main():
    """Test the diversity filter functionality."""
    filter_obj = DiversityFilter()
    
    # Test headlines
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
    
    print("Testing Diversity Filter...")
    print(f"Original headlines: {len(test_headlines)}")
    
    # Test diversity analysis
    analysis = filter_obj.analyze_diversity(test_headlines)
    print(f"Diversity score: {analysis['diversity_score']:.3f}")
    print(f"Meets threshold: {analysis['meets_diversity_threshold']}")
    
    # Test filtering
    filtered = filter_obj.filter_similar_headlines(test_headlines)
    print(f"Filtered headlines: {len(filtered)}")
    
    # Test diversity enforcement
    diverse = filter_obj.ensure_diversity(test_headlines)
    print(f"Diverse headlines: {len(diverse)}")
    
    # Test recommendations
    recommendations = filter_obj.get_diversity_recommendations(test_headlines)
    print(f"Recommendations: {recommendations}")
    
    print("\nFiltered headlines:")
    for i, headline in enumerate(filtered, 1):
        print(f"  {i}. {headline}")


if __name__ == "__main__":
    main()
