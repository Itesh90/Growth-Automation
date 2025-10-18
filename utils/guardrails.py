"""
Content Filtering and Guardrails
Ensures generated headlines meet quality, safety, and compliance standards.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json

from better_profanity import profanity
from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class ContentLevel(Enum):
    """Content safety levels."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


class ComplianceType(Enum):
    """Types of compliance checks."""
    PROFANITY = "profanity"
    SPAM = "spam"
    MISLEADING = "misleading"
    REGULATORY = "regulatory"
    BRAND_SAFETY = "brand_safety"
    ACCESSIBILITY = "accessibility"


@dataclass
class ContentCheck:
    """Result of a content check."""
    check_type: ComplianceType
    level: ContentLevel
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ContentFilterResult:
    """Result of content filtering."""
    is_safe: bool
    level: ContentLevel
    checks: List[ContentCheck]
    filtered_content: str
    warnings: List[str]
    blocked_reasons: List[str]


class ContentGuardrails:
    """Content filtering and safety system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the content guardrails system.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "utils/guardrails_config.json"
        self.config = self._load_config()
        
        # Initialize profanity filter
        profanity.load_censor_words()
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Load compliance rules
        self._load_compliance_rules()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "max_length": 200,
            "min_length": 10,
            "max_caps_ratio": 0.5,
            "max_exclamation_ratio": 0.3,
            "max_question_ratio": 0.2,
            "max_digit_ratio": 0.3,
            "max_special_char_ratio": 0.2,
            "spam_keywords": [
                "click here", "buy now", "free money", "make money fast",
                "guaranteed", "no risk", "limited time", "act now",
                "exclusive offer", "secret", "insider", "proven"
            ],
            "misleading_indicators": [
                "100% free", "no cost", "no obligation", "risk-free",
                "guaranteed results", "instant success", "overnight",
                "miracle", "breakthrough", "revolutionary"
            ],
            "regulatory_keywords": [
                "cure", "treat", "heal", "medical advice", "prescription",
                "investment advice", "financial advice", "legal advice"
            ],
            "brand_safety_keywords": [
                "hate", "violence", "discrimination", "illegal",
                "dangerous", "harmful", "toxic"
            ]
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def _compile_patterns(self):
        """Compile regex patterns for content checking."""
        self.patterns = {
            'excessive_caps': re.compile(r'[A-Z]{3,}'),
            'excessive_punctuation': re.compile(r'[!?]{2,}'),
            'excessive_digits': re.compile(r'\d{4,}'),
            'url_pattern': re.compile(r'https?://\S+'),
            'email_pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_pattern': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'currency_pattern': re.compile(r'\$\d+'),
            'percentage_pattern': re.compile(r'\d+%'),
            'all_caps': re.compile(r'^[A-Z\s!?.,]+$'),
            'excessive_spaces': re.compile(r'\s{3,}'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'html_tags': re.compile(r'<[^>]+>'),
            'special_chars': re.compile(r'[^\w\s!?.,-]')
        }
    
    def _load_compliance_rules(self):
        """Load compliance rules and regulations."""
        self.compliance_rules = {
            'ftc_guidelines': {
                'description': 'FTC advertising guidelines',
                'rules': [
                    'No false or misleading claims',
                    'Clear disclosure of material connections',
                    'Substantiation of claims'
                ]
            },
            'ada_compliance': {
                'description': 'ADA accessibility guidelines',
                'rules': [
                    'Clear and simple language',
                    'Appropriate contrast ratios',
                    'Screen reader compatibility'
                ]
            },
            'gdpr_compliance': {
                'description': 'GDPR privacy guidelines',
                'rules': [
                    'Clear consent mechanisms',
                    'Data protection notices',
                    'Right to be forgotten'
                ]
            }
        }
    
    def check_profanity(self, text: str) -> ContentCheck:
        """
        Check for profanity and inappropriate language.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        if profanity.contains_profanity(text):
            return ContentCheck(
                check_type=ComplianceType.PROFANITY,
                level=ContentLevel.BLOCKED,
                message="Content contains profanity or inappropriate language",
                details={'profane_words': profanity.censor(text)}
            )
        
        return ContentCheck(
            check_type=ComplianceType.PROFANITY,
            level=ContentLevel.SAFE,
            message="No profanity detected"
        )
    
    def check_spam(self, text: str) -> ContentCheck:
        """
        Check for spam-like content.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        text_lower = text.lower()
        spam_keywords = self.config.get('spam_keywords', [])
        
        found_keywords = []
        for keyword in spam_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            return ContentCheck(
                check_type=ComplianceType.SPAM,
                level=ContentLevel.WARNING,
                message="Content contains potential spam keywords",
                details={'spam_keywords': found_keywords}
            )
        
        return ContentCheck(
            check_type=ComplianceType.SPAM,
            level=ContentLevel.SAFE,
            message="No spam indicators detected"
        )
    
    def check_misleading(self, text: str) -> ContentCheck:
        """
        Check for misleading or deceptive content.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        text_lower = text.lower()
        misleading_indicators = self.config.get('misleading_indicators', [])
        
        found_indicators = []
        for indicator in misleading_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        if found_indicators:
            return ContentCheck(
                check_type=ComplianceType.MISLEADING,
                level=ContentLevel.WARNING,
                message="Content may be misleading or deceptive",
                details={'misleading_indicators': found_indicators}
            )
        
        return ContentCheck(
            check_type=ComplianceType.MISLEADING,
            level=ContentLevel.SAFE,
            message="No misleading content detected"
        )
    
    def check_regulatory(self, text: str) -> ContentCheck:
        """
        Check for regulatory compliance issues.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        text_lower = text.lower()
        regulatory_keywords = self.config.get('regulatory_keywords', [])
        
        found_keywords = []
        for keyword in regulatory_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            return ContentCheck(
                check_type=ComplianceType.REGULATORY,
                level=ContentLevel.BLOCKED,
                message="Content may violate regulatory guidelines",
                details={'regulatory_keywords': found_keywords}
            )
        
        return ContentCheck(
            check_type=ComplianceType.REGULATORY,
            level=ContentLevel.SAFE,
            message="No regulatory issues detected"
        )
    
    def check_brand_safety(self, text: str) -> ContentCheck:
        """
        Check for brand safety issues.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        text_lower = text.lower()
        brand_safety_keywords = self.config.get('brand_safety_keywords', [])
        
        found_keywords = []
        for keyword in brand_safety_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            return ContentCheck(
                check_type=ComplianceType.BRAND_SAFETY,
                level=ContentLevel.BLOCKED,
                message="Content may harm brand safety",
                details={'brand_safety_keywords': found_keywords}
            )
        
        return ContentCheck(
            check_type=ComplianceType.BRAND_SAFETY,
            level=ContentLevel.SAFE,
            message="No brand safety issues detected"
        )
    
    def check_formatting(self, text: str) -> ContentCheck:
        """
        Check for formatting issues.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        issues = []
        
        # Check length
        if len(text) > self.config.get('max_length', 200):
            issues.append(f"Text too long ({len(text)} chars, max {self.config.get('max_length', 200)})")
        
        if len(text) < self.config.get('min_length', 10):
            issues.append(f"Text too short ({len(text)} chars, min {self.config.get('min_length', 10)})")
        
        # Check caps ratio
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > self.config.get('max_caps_ratio', 0.5):
            issues.append(f"Too many capital letters ({caps_ratio:.1%}, max {self.config.get('max_caps_ratio', 0.5):.1%})")
        
        # Check exclamation ratio
        exclamation_ratio = text.count('!') / len(text) if text else 0
        if exclamation_ratio > self.config.get('max_exclamation_ratio', 0.3):
            issues.append(f"Too many exclamation marks ({exclamation_ratio:.1%}, max {self.config.get('max_exclamation_ratio', 0.3):.1%})")
        
        # Check question ratio
        question_ratio = text.count('?') / len(text) if text else 0
        if question_ratio > self.config.get('max_question_ratio', 0.2):
            issues.append(f"Too many question marks ({question_ratio:.1%}, max {self.config.get('max_question_ratio', 0.2):.1%})")
        
        # Check for excessive patterns
        if self.patterns['excessive_caps'].search(text):
            issues.append("Excessive capital letters")
        
        if self.patterns['excessive_punctuation'].search(text):
            issues.append("Excessive punctuation")
        
        if self.patterns['excessive_spaces'].search(text):
            issues.append("Excessive spaces")
        
        if self.patterns['repeated_chars'].search(text):
            issues.append("Repeated characters")
        
        if issues:
            return ContentCheck(
                check_type=ComplianceType.ACCESSIBILITY,
                level=ContentLevel.WARNING,
                message="Formatting issues detected",
                details={'issues': issues}
            )
        
        return ContentCheck(
            check_type=ComplianceType.ACCESSIBILITY,
            level=ContentLevel.SAFE,
            message="No formatting issues detected"
        )
    
    def check_accessibility(self, text: str) -> ContentCheck:
        """
        Check for accessibility compliance.
        
        Args:
            text: Text to check
            
        Returns:
            ContentCheck result
        """
        issues = []
        
        # Check for HTML tags
        if self.patterns['html_tags'].search(text):
            issues.append("HTML tags not allowed")
        
        # Check for URLs
        if self.patterns['url_pattern'].search(text):
            issues.append("URLs not allowed in headlines")
        
        # Check for email addresses
        if self.patterns['email_pattern'].search(text):
            issues.append("Email addresses not allowed in headlines")
        
        # Check for phone numbers
        if self.patterns['phone_pattern'].search(text):
            issues.append("Phone numbers not allowed in headlines")
        
        # Check for special characters
        special_char_ratio = len(self.patterns['special_chars'].findall(text)) / len(text) if text else 0
        if special_char_ratio > self.config.get('max_special_char_ratio', 0.2):
            issues.append(f"Too many special characters ({special_char_ratio:.1%}, max {self.config.get('max_special_char_ratio', 0.2):.1%})")
        
        if issues:
            return ContentCheck(
                check_type=ComplianceType.ACCESSIBILITY,
                level=ContentLevel.WARNING,
                message="Accessibility issues detected",
                details={'issues': issues}
            )
        
        return ContentCheck(
            check_type=ComplianceType.ACCESSIBILITY,
            level=ContentLevel.SAFE,
            message="No accessibility issues detected"
        )
    
    def filter_content(self, text: str) -> ContentFilterResult:
        """
        Filter content through all safety checks.
        
        Args:
            text: Text to filter
            
        Returns:
            ContentFilterResult with filtering results
        """
        checks = []
        warnings = []
        blocked_reasons = []
        
        # Run all checks
        check_functions = [
            self.check_profanity,
            self.check_spam,
            self.check_misleading,
            self.check_regulatory,
            self.check_brand_safety,
            self.check_formatting,
            self.check_accessibility
        ]
        
        for check_func in check_functions:
            try:
                check_result = check_func(text)
                checks.append(check_result)
                
                if check_result.level == ContentLevel.WARNING:
                    warnings.append(check_result.message)
                elif check_result.level == ContentLevel.BLOCKED:
                    blocked_reasons.append(check_result.message)
                    
            except Exception as e:
                logger.error(f"Error in content check {check_func.__name__}: {e}")
        
        # Determine overall safety level
        if any(check.level == ContentLevel.BLOCKED for check in checks):
            level = ContentLevel.BLOCKED
            is_safe = False
        elif any(check.level == ContentLevel.WARNING for check in checks):
            level = ContentLevel.WARNING
            is_safe = True
        else:
            level = ContentLevel.SAFE
            is_safe = True
        
        # Apply content filtering
        filtered_content = self._apply_filtering(text, checks)
        
        return ContentFilterResult(
            is_safe=is_safe,
            level=level,
            checks=checks,
            filtered_content=filtered_content,
            warnings=warnings,
            blocked_reasons=blocked_reasons
        )
    
    def _apply_filtering(self, text: str, checks: List[ContentCheck]) -> str:
        """
        Apply filtering based on check results.
        
        Args:
            text: Original text
            checks: List of check results
            
        Returns:
            Filtered text
        """
        filtered_text = text
        
        # Apply profanity filtering
        profanity_check = next((check for check in checks if check.check_type == ComplianceType.PROFANITY), None)
        if profanity_check and profanity_check.level == ContentLevel.BLOCKED:
            filtered_text = profanity.censor(filtered_text)
        
        # Remove excessive patterns
        filtered_text = self.patterns['excessive_spaces'].sub(' ', filtered_text)
        filtered_text = self.patterns['repeated_chars'].sub(r'\1\1', filtered_text)
        
        # Remove HTML tags
        filtered_text = self.patterns['html_tags'].sub('', filtered_text)
        
        # Remove URLs and emails
        filtered_text = self.patterns['url_pattern'].sub('[URL]', filtered_text)
        filtered_text = self.patterns['email_pattern'].sub('[EMAIL]', filtered_text)
        filtered_text = self.patterns['phone_pattern'].sub('[PHONE]', filtered_text)
        
        return filtered_text.strip()
    
    def batch_filter(self, texts: List[str]) -> List[ContentFilterResult]:
        """
        Filter multiple texts in batch.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            List of ContentFilterResult objects
        """
        results = []
        for text in texts:
            result = self.filter_content(text)
            results.append(result)
        
        return results
    
    def get_safe_content(self, texts: List[str]) -> List[str]:
        """
        Get only safe content from a list of texts.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            List of safe texts
        """
        results = self.batch_filter(texts)
        safe_texts = []
        
        for i, result in enumerate(results):
            if result.is_safe:
                safe_texts.append(result.filtered_content)
        
        return safe_texts
    
    def get_filtering_summary(self, results: List[ContentFilterResult]) -> Dict[str, Any]:
        """
        Get summary of filtering results.
        
        Args:
            results: List of filtering results
            
        Returns:
            Summary dictionary
        """
        total = len(results)
        safe = sum(1 for r in results if r.is_safe)
        warnings = sum(1 for r in results if r.level == ContentLevel.WARNING)
        blocked = sum(1 for r in results if r.level == ContentLevel.BLOCKED)
        
        # Count check types
        check_counts = {}
        for result in results:
            for check in result.checks:
                check_type = check.check_type.value
                if check_type not in check_counts:
                    check_counts[check_type] = {'safe': 0, 'warning': 0, 'blocked': 0}
                check_counts[check_type][check.level.value] += 1
        
        return {
            'total_texts': total,
            'safe_texts': safe,
            'warning_texts': warnings,
            'blocked_texts': blocked,
            'safety_rate': safe / total if total > 0 else 0,
            'check_breakdown': check_counts
        }


def main():
    """Test the content guardrails system."""
    guardrails = ContentGuardrails()
    
    # Test texts
    test_texts = [
        "Get 50% Off Today Only!",
        "This is a fucking amazing deal!",  # Profanity
        "CLICK HERE NOW!!! FREE MONEY!!!",  # Spam + excessive caps
        "100% FREE - NO RISK GUARANTEED",  # Misleading
        "Cure your illness with this product",  # Regulatory
        "Transform Your Life in 30 Days",
        "Free Shipping on All Orders"
    ]
    
    print("Testing Content Guardrails System...")
    
    # Test individual filtering
    for text in test_texts:
        result = guardrails.filter_content(text)
        print(f"\nText: {text}")
        print(f"Safe: {result.is_safe}, Level: {result.level.value}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        if result.blocked_reasons:
            print(f"Blocked: {result.blocked_reasons}")
        if result.filtered_content != text:
            print(f"Filtered: {result.filtered_content}")
    
    # Test batch filtering
    print("\n" + "="*50)
    print("Batch Filtering Results:")
    results = guardrails.batch_filter(test_texts)
    summary = guardrails.get_filtering_summary(results)
    
    print(f"Total texts: {summary['total_texts']}")
    print(f"Safe texts: {summary['safe_texts']}")
    print(f"Warning texts: {summary['warning_texts']}")
    print(f"Blocked texts: {summary['blocked_texts']}")
    print(f"Safety rate: {summary['safety_rate']:.1%}")
    
    # Test safe content extraction
    safe_texts = guardrails.get_safe_content(test_texts)
    print(f"\nSafe texts: {safe_texts}")


if __name__ == "__main__":
    main()
