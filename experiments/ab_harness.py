"""
A/B Testing Framework
Provides statistical testing capabilities for headline variants.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    test_id: str
    variant_a: str
    variant_b: str
    impressions_a: int
    impressions_b: int
    clicks_a: int
    clicks_b: int
    ctr_a: float
    ctr_b: float
    lift: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    test_duration_days: int
    created_at: datetime
    status: str  # 'running', 'completed', 'stopped'


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    variant_a: str
    variant_b: str
    target_impressions: int
    significance_level: float = 0.05
    power: float = 0.8
    min_sample_size: int = 1000
    max_duration_days: int = 30
    early_stopping: bool = True
    created_at: Optional[datetime] = None


class ABTestHarness:
    """A/B testing framework for headline optimization."""
    
    def __init__(self, results_file: Optional[str] = None):
        """
        Initialize the A/B testing harness.
        
        Args:
            results_file: Path to store test results
        """
        self.results_file = results_file or "data/ab_test_results.json"
        self.tests: Dict[str, ABTestResult] = {}
        self.configs: Dict[str, ABTestConfig] = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        # Load existing results
        self._load_results()
    
    def _load_results(self):
        """Load existing test results from file."""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                
                # Load test results
                for test_id, test_data in data.get('tests', {}).items():
                    test_data['created_at'] = datetime.fromisoformat(test_data['created_at'])
                    self.tests[test_id] = ABTestResult(**test_data)
                
                # Load test configs
                for test_id, config_data in data.get('configs', {}).items():
                    if config_data.get('created_at'):
                        config_data['created_at'] = datetime.fromisoformat(config_data['created_at'])
                    self.configs[test_id] = ABTestConfig(**config_data)
                
                logger.info(f"Loaded {len(self.tests)} test results and {len(self.configs)} configs")
                
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    
    def _save_results(self):
        """Save test results to file."""
        try:
            data = {
                'tests': {},
                'configs': {}
            }
            
            # Save test results
            for test_id, test in self.tests.items():
                test_dict = test.__dict__.copy()
                test_dict['created_at'] = test.created_at.isoformat()
                data['tests'][test_id] = test_dict
            
            # Save test configs
            for test_id, config in self.configs.items():
                config_dict = config.__dict__.copy()
                if config.created_at:
                    config_dict['created_at'] = config.created_at.isoformat()
                data['configs'][test_id] = config_dict
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save results: {e}")
    
    def create_test(self, test_id: str, variant_a: str, variant_b: str,
                   target_impressions: int, **kwargs) -> ABTestConfig:
        """
        Create a new A/B test configuration.
        
        Args:
            test_id: Unique identifier for the test
            variant_a: First headline variant
            variant_b: Second headline variant
            target_impressions: Target number of impressions per variant
            **kwargs: Additional configuration parameters
            
        Returns:
            ABTestConfig object
        """
        config = ABTestConfig(
            test_id=test_id,
            variant_a=variant_a,
            variant_b=variant_b,
            target_impressions=target_impressions,
            significance_level=kwargs.get('significance_level', settings.significance_level),
            power=kwargs.get('power', 0.8),
            min_sample_size=kwargs.get('min_sample_size', settings.min_sample_size),
            max_duration_days=kwargs.get('max_duration_days', 30),
            early_stopping=kwargs.get('early_stopping', True),
            created_at=datetime.now()
        )
        
        self.configs[test_id] = config
        self._save_results()
        
        logger.info(f"Created A/B test: {test_id}")
        return config
    
    def calculate_sample_size(self, baseline_ctr: float, mde: float = 0.1,
                            significance_level: float = 0.05, power: float = 0.8) -> int:
        """
        Calculate required sample size for A/B test.
        
        Args:
            baseline_ctr: Baseline click-through rate
            mde: Minimum detectable effect (as proportion)
            significance_level: Significance level (alpha)
            power: Statistical power (1 - beta)
            
        Returns:
            Required sample size per variant
        """
        # Effect size
        effect_size = mde * baseline_ctr
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation for proportions
        p1 = baseline_ctr
        p2 = baseline_ctr + effect_size
        p_pooled = (p1 + p2) / 2
        
        n = ((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (effect_size ** 2)
        
        return int(np.ceil(n))
    
    def simulate_test_data(self, variant_a: str, variant_b: str, 
                          impressions_a: int, impressions_b: int,
                          true_ctr_a: float, true_ctr_b: float) -> Dict[str, Any]:
        """
        Simulate A/B test data.
        
        Args:
            variant_a: First variant headline
            variant_b: Second variant headline
            impressions_a: Number of impressions for variant A
            impressions_b: Number of impressions for variant B
            true_ctr_a: True CTR for variant A
            true_ctr_b: True CTR for variant B
            
        Returns:
            Dictionary with simulated test data
        """
        # Simulate clicks using binomial distribution
        clicks_a = np.random.binomial(impressions_a, true_ctr_a)
        clicks_b = np.random.binomial(impressions_b, true_ctr_b)
        
        # Calculate observed CTRs
        ctr_a = clicks_a / impressions_a if impressions_a > 0 else 0
        ctr_b = clicks_b / impressions_b if impressions_b > 0 else 0
        
        return {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'impressions_a': impressions_a,
            'impressions_b': impressions_b,
            'clicks_a': clicks_a,
            'clicks_b': clicks_b,
            'ctr_a': ctr_a,
            'ctr_b': ctr_b,
            'true_ctr_a': true_ctr_a,
            'true_ctr_b': true_ctr_b
        }
    
    def run_statistical_test(self, clicks_a: int, impressions_a: int,
                           clicks_b: int, impressions_b: int,
                           significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Run statistical tests on A/B test data.
        
        Args:
            clicks_a: Clicks for variant A
            impressions_a: Impressions for variant A
            clicks_b: Clicks for variant B
            impressions_b: Impressions for variant B
            significance_level: Significance level for testing
            
        Returns:
            Dictionary with test results
        """
        # Calculate CTRs
        ctr_a = clicks_a / impressions_a if impressions_a > 0 else 0
        ctr_b = clicks_b / impressions_b if impressions_b > 0 else 0
        
        # Calculate lift
        lift = (ctr_b - ctr_a) / ctr_a if ctr_a > 0 else 0
        
        # Chi-square test for proportions
        contingency_table = np.array([
            [clicks_a, impressions_a - clicks_a],
            [clicks_b, impressions_b - clicks_b]
        ])
        
        chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
        
        # Two-proportion z-test
        p_pooled = (clicks_a + clicks_b) / (impressions_a + impressions_b)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/impressions_a + 1/impressions_b))
        z_score = (ctr_b - ctr_a) / se if se > 0 else 0
        p_value_z = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval for difference
        se_diff = np.sqrt(ctr_a * (1 - ctr_a) / impressions_a + ctr_b * (1 - ctr_b) / impressions_b)
        margin_error = stats.norm.ppf(1 - significance_level / 2) * se_diff
        ci_lower = (ctr_b - ctr_a) - margin_error
        ci_upper = (ctr_b - ctr_a) + margin_error
        
        # Determine significance
        is_significant = p_value_z < significance_level
        
        return {
            'ctr_a': ctr_a,
            'ctr_b': ctr_b,
            'lift': lift,
            'p_value_chi2': p_value_chi2,
            'p_value_z': p_value_z,
            'z_score': z_score,
            'is_significant': is_significant,
            'confidence_interval': (ci_lower, ci_upper),
            'chi2_statistic': chi2,
            'degrees_of_freedom': dof
        }
    
    def run_test(self, test_id: str, impressions_a: int, impressions_b: int,
                clicks_a: int, clicks_b: int) -> ABTestResult:
        """
        Run an A/B test and return results.
        
        Args:
            test_id: Test identifier
            impressions_a: Impressions for variant A
            impressions_b: Impressions for variant B
            clicks_a: Clicks for variant A
            clicks_b: Clicks for variant B
            
        Returns:
            ABTestResult object
        """
        if test_id not in self.configs:
            raise ValueError(f"Test configuration not found: {test_id}")
        
        config = self.configs[test_id]
        
        # Run statistical test
        test_results = self.run_statistical_test(
            clicks_a, impressions_a, clicks_b, impressions_b,
            config.significance_level
        )
        
        # Calculate test duration
        test_duration = (datetime.now() - config.created_at).days if config.created_at else 0
        
        # Create result
        result = ABTestResult(
            test_id=test_id,
            variant_a=config.variant_a,
            variant_b=config.variant_b,
            impressions_a=impressions_a,
            impressions_b=impressions_b,
            clicks_a=clicks_a,
            clicks_b=clicks_b,
            ctr_a=test_results['ctr_a'],
            ctr_b=test_results['ctr_b'],
            lift=test_results['lift'],
            p_value=test_results['p_value_z'],
            is_significant=test_results['is_significant'],
            confidence_interval=test_results['confidence_interval'],
            test_duration_days=test_duration,
            created_at=config.created_at or datetime.now(),
            status='completed'
        )
        
        # Store result
        self.tests[test_id] = result
        self._save_results()
        
        logger.info(f"Completed A/B test: {test_id}")
        return result
    
    def simulate_test(self, test_id: str, true_ctr_a: float, true_ctr_b: float,
                     impressions_a: Optional[int] = None, 
                     impressions_b: Optional[int] = None) -> ABTestResult:
        """
        Simulate an A/B test with given true CTRs.
        
        Args:
            test_id: Test identifier
            true_ctr_a: True CTR for variant A
            true_ctr_b: True CTR for variant B
            impressions_a: Impressions for variant A (uses config target if None)
            impressions_b: Impressions for variant B (uses config target if None)
            
        Returns:
            ABTestResult object
        """
        if test_id not in self.configs:
            raise ValueError(f"Test configuration not found: {test_id}")
        
        config = self.configs[test_id]
        
        # Use target impressions if not specified
        if impressions_a is None:
            impressions_a = config.target_impressions
        if impressions_b is None:
            impressions_b = config.target_impressions
        
        # Simulate data
        simulated_data = self.simulate_test_data(
            config.variant_a, config.variant_b,
            impressions_a, impressions_b,
            true_ctr_a, true_ctr_b
        )
        
        # Run test
        result = self.run_test(
            test_id,
            simulated_data['impressions_a'],
            simulated_data['impressions_b'],
            simulated_data['clicks_a'],
            simulated_data['clicks_b']
        )
        
        return result
    
    def get_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """
        Get results for a specific test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            ABTestResult object or None if not found
        """
        return self.tests.get(test_id)
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """
        List all tests with summary information.
        
        Returns:
            List of test summaries
        """
        test_summaries = []
        
        for test_id, result in self.tests.items():
            summary = {
                'test_id': test_id,
                'variant_a': result.variant_a,
                'variant_b': result.variant_b,
                'ctr_a': result.ctr_a,
                'ctr_b': result.ctr_b,
                'lift': result.lift,
                'p_value': result.p_value,
                'is_significant': result.is_significant,
                'status': result.status,
                'created_at': result.created_at.isoformat() if result.created_at else None
            }
            test_summaries.append(summary)
        
        return test_summaries
    
    def plot_test_results(self, test_id: str, save_path: Optional[str] = None):
        """
        Plot A/B test results.
        
        Args:
            test_id: Test identifier
            save_path: Path to save plot (optional)
        """
        if test_id not in self.tests:
            raise ValueError(f"Test results not found: {test_id}")
        
        result = self.tests[test_id]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. CTR comparison
        variants = ['Variant A', 'Variant B']
        ctrs = [result.ctr_a, result.ctr_b]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(variants, ctrs, color=colors)
        ax1.set_title('CTR Comparison')
        ax1.set_ylabel('Click-Through Rate')
        ax1.set_ylim(0, max(ctrs) * 1.2)
        
        # Add value labels on bars
        for bar, ctr in zip(bars, ctrs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{ctr:.3f}', ha='center', va='bottom')
        
        # 2. Impressions and clicks
        metrics = ['Impressions', 'Clicks']
        values_a = [result.impressions_a, result.clicks_a]
        values_b = [result.impressions_b, result.clicks_b]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, values_a, width, label='Variant A', color='skyblue')
        ax2.bar(x + width/2, values_b, width, label='Variant B', color='lightcoral')
        
        ax2.set_title('Impressions and Clicks')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        
        # 3. Confidence interval
        ci_lower, ci_upper = result.confidence_interval
        ax3.barh(['CTR Difference'], [ci_upper - ci_lower], left=ci_lower, 
                color='lightgreen', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Confidence Interval for CTR Difference')
        ax3.set_xlabel('CTR Difference')
        ax3.text(0, 0, f'[{ci_lower:.4f}, {ci_upper:.4f}]', 
                ha='center', va='center', fontweight='bold')
        
        # 4. Test summary
        ax4.axis('off')
        summary_text = f"""
        Test ID: {test_id}
        Lift: {result.lift:.1%}
        P-value: {result.p_value:.4f}
        Significant: {'Yes' if result.is_significant else 'No'}
        Duration: {result.test_duration_days} days
        """
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Test results plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, test_id: str) -> str:
        """
        Generate a text report for an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Formatted report string
        """
        if test_id not in self.tests:
            return f"Test results not found: {test_id}"
        
        result = self.tests[test_id]
        
        report = f"""
A/B Test Report
===============

Test ID: {test_id}
Created: {result.created_at.strftime('%Y-%m-%d %H:%M:%S') if result.created_at else 'N/A'}
Duration: {result.test_duration_days} days

Variants:
---------
Variant A: {result.variant_a}
Variant B: {result.variant_b}

Results:
--------
Variant A: {result.impressions_a:,} impressions, {result.clicks_a:,} clicks, {result.ctr_a:.3f} CTR
Variant B: {result.impressions_b:,} impressions, {result.clicks_b:,} clicks, {result.ctr_b:.3f} CTR

Statistical Analysis:
--------------------
Lift: {result.lift:.1%}
P-value: {result.p_value:.4f}
Significant: {'Yes' if result.is_significant else 'No'}
Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]

Conclusion:
-----------
{'Variant B is significantly better than Variant A' if result.is_significant and result.lift > 0 else
 'Variant A is significantly better than Variant B' if result.is_significant and result.lift < 0 else
 'No significant difference between variants'}
        """
        
        return report.strip()


def main():
    """Test the A/B testing framework."""
    harness = ABTestHarness()
    
    # Create a test
    test_id = "test_001"
    config = harness.create_test(
        test_id=test_id,
        variant_a="Get 50% Off Today Only!",
        variant_b="Transform Your Life in 30 Days",
        target_impressions=10000
    )
    
    print(f"Created test: {test_id}")
    
    # Simulate test with different CTRs
    result = harness.simulate_test(
        test_id=test_id,
        true_ctr_a=0.05,  # 5% CTR
        true_ctr_b=0.06   # 6% CTR (20% lift)
    )
    
    print(f"Test completed: {result.lift:.1%} lift, p-value: {result.p_value:.4f}")
    print(f"Significant: {result.is_significant}")
    
    # Generate report
    report = harness.generate_report(test_id)
    print("\n" + report)
    
    # List all tests
    tests = harness.list_tests()
    print(f"\nTotal tests: {len(tests)}")


if __name__ == "__main__":
    main()
