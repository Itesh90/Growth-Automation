"""
Synthetic CTR data generator for training and testing.
Creates realistic click-through rate data with known ground truth.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from api.features import feature_extractor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CTRSimulationConfig:
    """Configuration for CTR simulation."""
    n_samples: int = 10000
    min_ctr: float = 0.005  # 0.5%
    max_ctr: float = 0.05   # 5%
    noise_level: float = 0.1
    seed: int = 42


class CTRSimulator:
    """Generates synthetic CTR data with realistic patterns."""
    
    def __init__(self, config: Optional[CTRSimulationConfig] = None):
        self.config = config or CTRSimulationConfig()
        self.feature_weights = {}
        self.bias = 0.0
        self.scaler = StandardScaler()
        self._init_ground_truth_weights()
        
    def _init_ground_truth_weights(self):
        """Initialize ground truth feature weights based on marketing research."""
        # Positive weights (increase CTR)
        self.feature_weights = {
            # Basic features
            'char_count': -0.02,  # Shorter headlines perform better
            'word_count': -0.01,
            'num_digits': 0.05,   # Numbers increase CTR
            'num_exclamations': 0.03,
            'num_questions': 0.04,
            
            # Linguistic features
            'sentiment_polarity': 0.08,  # Positive sentiment helps
            'flesch_reading_ease': 0.02,  # Easier to read is better
            'avg_syllables_per_word': -0.01,  # Shorter words better
            
            # Power words
            'has_action_verb': 0.06,
            'has_urgency_word': 0.04,
            'has_benefit_word': 0.07,
            'has_number_word': 0.05,
            'num_action_verbs': 0.02,
            'num_benefit_words': 0.03,
            
            # Structure features
            'starts_with_question': 0.05,
            'starts_with_number': 0.04,
            'ends_with_exclamation': 0.03,
            'contains_question_mark': 0.02,
            'contains_exclamation': 0.02,
            
            # Context features
            'is_email_subject': 0.01,
            'is_social_ad': -0.01,
            'is_display_ad': -0.02,
            'is_b2c_audience': 0.02,
            
            # Embedding features (first few dimensions)
            'embedding_dim_0': 0.03,
            'embedding_dim_1': 0.02,
            'embedding_dim_2': -0.01,
            'embedding_dim_3': 0.02,
            'embedding_dim_4': 0.01,
        }
        
        # Set bias to achieve realistic CTR distribution
        self.bias = -2.5  # Logit scale bias
        
    def _generate_sample_headlines(self, n_samples: int) -> List[str]:
        """Generate diverse sample headlines for simulation."""
        base_templates = [
            "Get {benefit} in {time}",
            "{number} ways to {action} your {goal}",
            "How to {action} {goal} {timeframe}",
            "{action} your {goal} with {method}",
            "Why {audience} choose {solution}",
            "{number}% of {audience} {action} this",
            "Stop {problem}. Start {solution}.",
            "{action} {goal} in {time} - {guarantee}",
            "The {adjective} way to {action} {goal}",
            "{number} {benefit} tips for {audience}",
            "What {audience} need to know about {topic}",
            "{action} {goal} like a {role}",
            "{number} {solution} that {benefit}",
            "How {audience} {action} {goal} {timeframe}",
            "{action} {goal} without {obstacle}",
            "The {number} {solution} for {problem}",
            "{action} {goal} in {time} - {proof}",
            "Why {audience} {action} {solution}",
            "{number} {benefit} {solution} for {audience}",
            "{action} {goal} with {method} - {guarantee}"
        ]
        
        # Fill-in words
        fill_words = {
            'benefit': ['results', 'success', 'profit', 'growth', 'savings', 'freedom', 'confidence'],
            'time': ['30 days', '1 week', '24 hours', 'minutes', 'instantly', 'today'],
            'number': ['3', '5', '7', '10', '15', '20', '50', '100'],
            'action': ['boost', 'increase', 'double', 'maximize', 'achieve', 'get', 'build', 'create'],
            'goal': ['sales', 'revenue', 'conversions', 'leads', 'engagement', 'growth', 'success'],
            'timeframe': ['fast', 'quickly', 'easily', 'effortlessly', 'instantly'],
            'method': ['AI', 'automation', 'strategy', 'system', 'formula', 'blueprint'],
            'audience': ['entrepreneurs', 'marketers', 'businesses', 'professionals', 'experts'],
            'solution': ['method', 'system', 'strategy', 'tool', 'technique', 'approach'],
            'problem': ['struggling', 'failing', 'losing', 'wasting time', 'missing out'],
            'guarantee': ['guaranteed', 'proven', 'tested', 'results-driven', 'successful'],
            'adjective': ['secret', 'proven', 'simple', 'powerful', 'effective', 'winning'],
            'role': ['pro', 'expert', 'champion', 'winner', 'leader', 'master'],
            'topic': ['marketing', 'sales', 'growth', 'automation', 'strategy', 'success'],
            'obstacle': ['effort', 'complexity', 'risk', 'cost', 'time', 'struggle'],
            'proof': ['proven', 'tested', 'guaranteed', 'results-driven', 'successful']
        }
        
        headlines = []
        random.seed(self.config.seed)
        
        for i in range(n_samples):
            template = random.choice(base_templates)
            headline = template
            
            # Replace placeholders with random words
            for placeholder, word_list in fill_words.items():
                if f"{{{placeholder}}}" in headline:
                    word = random.choice(word_list)
                    headline = headline.replace(f"{{{placeholder}}}", word)
            
            # Add some variation
            if random.random() < 0.3:
                headline += "!"
            if random.random() < 0.2:
                headline = headline.replace(".", "?")
            
            headlines.append(headline)
        
        return headlines
    
    def _calculate_true_ctr(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate true CTR using logistic function with ground truth weights."""
        # Initialize logits
        logits = np.full(len(features), self.bias)
        
        # Add weighted features
        for feature, weight in self.feature_weights.items():
            if feature in features.columns:
                logits += features[feature] * weight
        
        # Convert to probabilities using sigmoid
        true_ctr = expit(logits)
        
        # Apply bounds
        true_ctr = np.clip(true_ctr, self.config.min_ctr, self.config.max_ctr)
        
        return true_ctr
    
    def _add_noise(self, true_ctr: np.ndarray, impressions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add realistic noise to CTR simulation."""
        # Add noise to true CTR
        noise = np.random.normal(0, self.config.noise_level, len(true_ctr))
        noisy_ctr = np.clip(true_ctr + noise, 0.001, 0.1)  # Keep reasonable bounds
        
        # Generate actual clicks using binomial distribution
        clicks = np.random.binomial(impressions, noisy_ctr)
        
        # Calculate observed CTR
        observed_ctr = clicks / impressions
        
        return observed_ctr, clicks
    
    def generate_synthetic_data(self, headlines: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate synthetic CTR dataset.
        
        Args:
            headlines: Optional list of headlines. If None, generates sample headlines.
            
        Returns:
            DataFrame with features, true CTR, and simulated clicks/impressions
        """
        logger.info(f"Generating {self.config.n_samples} synthetic samples...")
        
        # Set random seed for reproducibility
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        
        # Generate or use provided headlines
        if headlines is None:
            headlines = self._generate_sample_headlines(self.config.n_samples)
        else:
            headlines = headlines[:self.config.n_samples]
        
        # Extract features
        logger.info("Extracting features...")
        features_df = feature_extractor.extract_features_batch(headlines)
        
        # Calculate true CTR
        logger.info("Calculating true CTR...")
        true_ctr = self._calculate_true_ctr(features_df)
        
        # Generate impressions (realistic distribution)
        # Most ads get few impressions, some get many
        impressions = np.random.lognormal(mean=6, sigma=1.5, size=len(headlines)).astype(int)
        impressions = np.clip(impressions, 100, 100000)  # Reasonable bounds
        
        # Add noise and generate clicks
        logger.info("Adding noise and generating clicks...")
        observed_ctr, clicks = self._add_noise(true_ctr, impressions)
        
        # Create final dataset
        dataset = features_df.copy()
        dataset['true_ctr'] = true_ctr
        dataset['observed_ctr'] = observed_ctr
        dataset['impressions'] = impressions
        dataset['clicks'] = clicks
        dataset['click_label'] = (clicks > 0).astype(int)  # Binary click indicator
        
        # Add some metadata
        dataset['sample_id'] = range(len(dataset))
        dataset['generated_at'] = pd.Timestamp.now()
        
        logger.info(f"Generated dataset with shape: {dataset.shape}")
        logger.info(f"CTR range: {dataset['observed_ctr'].min():.4f} - {dataset['observed_ctr'].max():.4f}")
        logger.info(f"Average CTR: {dataset['observed_ctr'].mean():.4f}")
        
        return dataset
    
    def generate_ab_test_data(
        self, 
        control_headline: str, 
        variant_headlines: List[str],
        n_impressions: int = 10000
    ) -> pd.DataFrame:
        """
        Generate A/B test simulation data.
        
        Args:
            control_headline: Control headline
            variant_headlines: List of variant headlines
            n_impressions: Total impressions to simulate
            
        Returns:
            DataFrame with A/B test results
        """
        all_headlines = [control_headline] + variant_headlines
        
        # Extract features
        features_df = feature_extractor.extract_features_batch(all_headlines)
        
        # Calculate true CTR
        true_ctr = self._calculate_true_ctr(features_df)
        
        # Split impressions (50% control, 50% split among variants)
        control_impressions = n_impressions // 2
        variant_impressions = n_impressions // (2 * len(variant_headlines))
        
        impressions = [control_impressions] + [variant_impressions] * len(variant_headlines)
        
        # Generate clicks
        clicks = []
        observed_ctr = []
        
        for i, (ctr, imp) in enumerate(zip(true_ctr, impressions)):
            # Add some noise
            noisy_ctr = np.clip(ctr + np.random.normal(0, self.config.noise_level), 0.001, 0.1)
            click_count = np.random.binomial(imp, noisy_ctr)
            clicks.append(click_count)
            observed_ctr.append(click_count / imp)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'headline': all_headlines,
            'variant_type': ['control'] + [f'variant_{i+1}' for i in range(len(variant_headlines))],
            'true_ctr': true_ctr,
            'observed_ctr': observed_ctr,
            'impressions': impressions,
            'clicks': clicks,
            'is_control': [True] + [False] * len(variant_headlines)
        })
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get ground truth feature importance for analysis."""
        importance_df = pd.DataFrame([
            {'feature': feature, 'weight': weight, 'abs_weight': abs(weight)}
            for feature, weight in self.feature_weights.items()
        ])
        
        importance_df = importance_df.sort_values('abs_weight', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def save_synthetic_data(self, dataset: pd.DataFrame, filepath: str):
        """Save synthetic dataset to file."""
        dataset.to_csv(filepath, index=False)
        logger.info(f"Saved synthetic dataset to {filepath}")
    
    def load_synthetic_data(self, filepath: str) -> pd.DataFrame:
        """Load synthetic dataset from file."""
        dataset = pd.read_csv(filepath)
        logger.info(f"Loaded synthetic dataset from {filepath}")
        return dataset


# Global simulator instance
ctr_simulator = CTRSimulator()


def generate_training_data(n_samples: int = 10000, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to generate training data.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Optional path to save the dataset
        
    Returns:
        Generated dataset
    """
    config = CTRSimulationConfig(n_samples=n_samples)
    simulator = CTRSimulator(config)
    dataset = simulator.generate_synthetic_data()
    
    if save_path:
        simulator.save_synthetic_data(dataset, save_path)
    
    return dataset


def simulate_ab_test(
    control_headline: str, 
    variant_headlines: List[str], 
    n_impressions: int = 10000
) -> pd.DataFrame:
    """
    Convenience function to simulate A/B test.
    
    Args:
        control_headline: Control headline
        variant_headlines: List of variant headlines
        n_impressions: Total impressions to simulate
        
    Returns:
        A/B test simulation results
    """
    return ctr_simulator.generate_ab_test_data(control_headline, variant_headlines, n_impressions)
