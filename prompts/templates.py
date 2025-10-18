"""
Production Prompt Templates
High-quality prompts for different headline generation scenarios.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class Tone(Enum):
    """Available tones for headline generation."""
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"
    INFORMATIVE = "informative"
    ACTION_ORIENTED = "action-oriented"
    SOCIAL_PROOF = "social proof"
    EMOTIONAL = "emotional"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    AUTHORITATIVE = "authoritative"


class Placement(Enum):
    """Available ad placements."""
    FACEBOOK = "facebook"
    GOOGLE_ADS = "google_ads"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    EMAIL = "email"
    DISPLAY = "display"
    SEARCH = "search"


@dataclass
class PromptTemplate:
    """Template for headline generation prompts."""
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    examples: List[Dict[str, str]]
    tone: Optional[Tone] = None
    placement: Optional[Placement] = None
    max_tokens: int = 1000
    temperature: float = 0.7


class PromptManager:
    """Manages production prompt templates."""
    
    def __init__(self, templates_file: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            templates_file: Path to templates configuration file
        """
        self.templates_file = templates_file or "prompts/templates.json"
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Ensure prompts directory exists
        os.makedirs(os.path.dirname(self.templates_file), exist_ok=True)
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates from file."""
        try:
            if os.path.exists(self.templates_file):
                with open(self.templates_file, 'r') as f:
                    data = json.load(f)
                
                for template_name, template_data in data.items():
                    self.templates[template_name] = PromptTemplate(**template_data)
                
                logger.info(f"Loaded {len(self.templates)} prompt templates")
            else:
                # Create default templates
                self._create_default_templates()
                self._save_templates()
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self._create_default_templates()
    
    def _save_templates(self):
        """Save templates to file."""
        try:
            data = {}
            for name, template in self.templates.items():
                data[name] = template.__dict__
            
            with open(self.templates_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving templates: {e}")
    
    def _create_default_templates(self):
        """Create default prompt templates."""
        # E-commerce template
        self.templates["ecommerce_urgent"] = PromptTemplate(
            name="ecommerce_urgent",
            description="Urgent e-commerce headlines for sales and promotions",
            system_prompt="""You are an expert copywriter specializing in high-converting e-commerce headlines. 
            Your headlines should create urgency, highlight value, and drive immediate action. 
            Focus on benefits, scarcity, and emotional triggers that motivate purchases.""",
            user_prompt_template="""Generate {n_variants} compelling e-commerce headlines for: {base_copy}
            
            Tone: Urgent and action-oriented
            Placement: {placement}
            Key elements to include:
            - Clear value proposition
            - Urgency or scarcity
            - Action-oriented language
            - Emotional appeal
            
            Make each headline unique and test different approaches.""",
            examples=[
                {
                    "base_copy": "Summer sale on clothing",
                    "headlines": [
                        "🔥 50% Off Summer Styles - Ends Tonight!",
                        "Last Chance: Summer Sale - 50% Off Everything",
                        "Summer Styles 50% Off - Limited Time Only!"
                    ]
                }
            ],
            tone=Tone.URGENT,
            placement=Placement.FACEBOOK
        )
        
        # SaaS template
        self.templates["saas_professional"] = PromptTemplate(
            name="saas_professional",
            description="Professional SaaS headlines for B2B software",
            system_prompt="""You are a B2B marketing expert creating headlines for software products. 
            Your headlines should emphasize efficiency, ROI, and professional benefits. 
            Focus on business outcomes and professional credibility.""",
            user_prompt_template="""Generate {n_variants} professional SaaS headlines for: {base_copy}
            
            Tone: Professional and authoritative
            Placement: {placement}
            Key elements to include:
            - Business benefits
            - Efficiency gains
            - Professional credibility
            - Clear value proposition
            
            Make each headline unique and focus on different business outcomes.""",
            examples=[
                {
                    "base_copy": "Project management software",
                    "headlines": [
                        "Streamline Your Projects - 30% More Efficient",
                        "Professional Project Management Made Simple",
                        "Boost Team Productivity with Smart Project Tools"
                    ]
                }
            ],
            tone=Tone.PROFESSIONAL,
            placement=Placement.LINKEDIN
        )
        
        # Mobile app template
        self.templates["mobile_app_casual"] = PromptTemplate(
            name="mobile_app_casual",
            description="Casual mobile app headlines for consumer apps",
            system_prompt="""You are a mobile app marketing specialist creating headlines for consumer apps. 
            Your headlines should be casual, engaging, and highlight the fun or useful aspects of the app. 
            Focus on user benefits and emotional connection.""",
            user_prompt_template="""Generate {n_variants} casual mobile app headlines for: {base_copy}
            
            Tone: Casual and engaging
            Placement: {placement}
            Key elements to include:
            - User benefits
            - Fun or useful aspects
            - Emotional connection
            - Easy to understand
            
            Make each headline unique and test different emotional appeals.""",
            examples=[
                {
                    "base_copy": "Fitness tracking app",
                    "headlines": [
                        "Get Fit, Have Fun - Track Your Progress",
                        "Your Personal Fitness Coach in Your Pocket",
                        "Make Fitness Fun - Download Now!"
                    ]
                }
            ],
            tone=Tone.CASUAL,
            placement=Placement.INSTAGRAM
        )
        
        # Social proof template
        self.templates["social_proof"] = PromptTemplate(
            name="social_proof",
            description="Headlines that leverage social proof and testimonials",
            system_prompt="""You are a conversion expert specializing in social proof marketing. 
            Your headlines should build trust and credibility by highlighting user success, 
            testimonials, or community validation. Focus on trust signals and social validation.""",
            user_prompt_template="""Generate {n_variants} social proof headlines for: {base_copy}
            
            Tone: Trust-building and credible
            Placement: {placement}
            Key elements to include:
            - User testimonials or success stories
            - Community validation
            - Trust signals
            - Credibility indicators
            
            Make each headline unique and test different trust signals.""",
            examples=[
                {
                    "base_copy": "Online course platform",
                    "headlines": [
                        "Join 10,000+ Students Who've Transformed Their Careers",
                        "Rated 4.9/5 by 5,000+ Happy Students",
                        "Trusted by Professionals Worldwide - See Results"
                    ]
                }
            ],
            tone=Tone.SOCIAL_PROOF,
            placement=Placement.FACEBOOK
        )
        
        # Emotional template
        self.templates["emotional"] = PromptTemplate(
            name="emotional",
            description="Emotional headlines that connect with user feelings",
            system_prompt="""You are a psychological marketing expert creating emotionally resonant headlines. 
            Your headlines should tap into deep emotions, aspirations, and personal transformation. 
            Focus on emotional triggers and personal connection.""",
            user_prompt_template="""Generate {n_variants} emotional headlines for: {base_copy}
            
            Tone: Emotional and aspirational
            Placement: {placement}
            Key elements to include:
            - Emotional triggers
            - Personal transformation
            - Aspirational language
            - Deep connection
            
            Make each headline unique and test different emotional appeals.""",
            examples=[
                {
                    "base_copy": "Personal development program",
                    "headlines": [
                        "Transform Your Life - Start Your Journey Today",
                        "Unlock Your Potential - Discover What's Possible",
                        "Your Best Self Awaits - Take the First Step"
                    ]
                }
            ],
            tone=Tone.EMOTIONAL,
            placement=Placement.FACEBOOK
        )
        
        # Action-oriented template
        self.templates["action_oriented"] = PromptTemplate(
            name="action_oriented",
            description="Action-oriented headlines that drive immediate response",
            system_prompt="""You are a direct response copywriter creating action-oriented headlines. 
            Your headlines should create immediate action through clear calls-to-action, 
            urgency, and compelling offers. Focus on driving immediate response.""",
            user_prompt_template="""Generate {n_variants} action-oriented headlines for: {base_copy}
            
            Tone: Direct and action-focused
            Placement: {placement}
            Key elements to include:
            - Clear call-to-action
            - Immediate benefits
            - Urgency or scarcity
            - Direct language
            
            Make each headline unique and test different action triggers.""",
            examples=[
                {
                    "base_copy": "Free trial offer",
                    "headlines": [
                        "Start Your Free Trial Now - No Credit Card Required",
                        "Get Instant Access - Try Free for 30 Days",
                        "Claim Your Free Trial - Limited Time Offer"
                    ]
                }
            ],
            tone=Tone.ACTION_ORIENTED,
            placement=Placement.GOOGLE_ADS
        )
        
        logger.info(f"Created {len(self.templates)} default templates")
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        Get a specific template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            PromptTemplate object or None if not found
        """
        return self.templates.get(template_name)
    
    def get_templates_by_tone(self, tone: Tone) -> List[PromptTemplate]:
        """
        Get templates filtered by tone.
        
        Args:
            tone: Tone to filter by
            
        Returns:
            List of matching templates
        """
        return [template for template in self.templates.values() 
                if template.tone == tone]
    
    def get_templates_by_placement(self, placement: Placement) -> List[PromptTemplate]:
        """
        Get templates filtered by placement.
        
        Args:
            placement: Placement to filter by
            
        Returns:
            List of matching templates
        """
        return [template for template in self.templates.values() 
                if template.placement == placement]
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template information
        """
        templates_info = []
        for name, template in self.templates.items():
            templates_info.append({
                'name': name,
                'description': template.description,
                'tone': template.tone.value if template.tone else None,
                'placement': template.placement.value if template.placement else None,
                'examples_count': len(template.examples)
            })
        
        return templates_info
    
    def format_prompt(self, template_name: str, base_copy: str, n_variants: int = 5,
                     placement: str = "facebook", **kwargs) -> Dict[str, str]:
        """
        Format a prompt using a template.
        
        Args:
            template_name: Name of the template to use
            base_copy: Base copy to generate headlines for
            n_variants: Number of variants to generate
            placement: Ad placement
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with system and user prompts
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Format user prompt
        user_prompt = template.user_prompt_template.format(
            base_copy=base_copy,
            n_variants=n_variants,
            placement=placement,
            **kwargs
        )
        
        return {
            'system_prompt': template.system_prompt,
            'user_prompt': user_prompt,
            'max_tokens': template.max_tokens,
            'temperature': template.temperature
        }
    
    def get_best_template(self, tone: str, placement: str) -> Optional[PromptTemplate]:
        """
        Get the best template for given tone and placement.
        
        Args:
            tone: Desired tone
            placement: Ad placement
            
        Returns:
            Best matching template or None
        """
        # Try to find exact match
        for template in self.templates.values():
            if (template.tone and template.tone.value == tone.lower() and
                template.placement and template.placement.value == placement.lower()):
                return template
        
        # Try to find tone match
        for template in self.templates.values():
            if template.tone and template.tone.value == tone.lower():
                return template
        
        # Try to find placement match
        for template in self.templates.values():
            if template.placement and template.placement.value == placement.lower():
                return template
        
        # Return first available template
        return list(self.templates.values())[0] if self.templates else None
    
    def add_template(self, template: PromptTemplate):
        """
        Add a new template.
        
        Args:
            template: Template to add
        """
        self.templates[template.name] = template
        self._save_templates()
        logger.info(f"Added template: {template.name}")
    
    def remove_template(self, template_name: str):
        """
        Remove a template.
        
        Args:
            template_name: Name of template to remove
        """
        if template_name in self.templates:
            del self.templates[template_name]
            self._save_templates()
            logger.info(f"Removed template: {template_name}")
    
    def update_template(self, template_name: str, **updates):
        """
        Update an existing template.
        
        Args:
            template_name: Name of template to update
            **updates: Fields to update
        """
        if template_name in self.templates:
            template = self.templates[template_name]
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            self._save_templates()
            logger.info(f"Updated template: {template_name}")


# Global prompt manager instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global prompt manager instance (singleton pattern).
    
    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def format_headline_prompt(template_name: str, base_copy: str, n_variants: int = 5,
                          placement: str = "facebook", **kwargs) -> Dict[str, str]:
    """
    Convenience function for formatting headline prompts.
    
    Args:
        template_name: Name of the template to use
        base_copy: Base copy to generate headlines for
        n_variants: Number of variants to generate
        placement: Ad placement
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with system and user prompts
    """
    manager = get_prompt_manager()
    return manager.format_prompt(template_name, base_copy, n_variants, placement, **kwargs)


def main():
    """Test the prompt templates system."""
    manager = PromptManager()
    
    print("Testing Prompt Templates System...")
    print(f"Loaded {len(manager.templates)} templates")
    
    # List templates
    templates = manager.list_templates()
    print("\nAvailable templates:")
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")
    
    # Test template selection
    template = manager.get_best_template("urgent", "facebook")
    if template:
        print(f"\nBest template for urgent/facebook: {template.name}")
    
    # Test prompt formatting
    try:
        prompt = manager.format_prompt(
            "ecommerce_urgent",
            "Summer sale on clothing",
            n_variants=3,
            placement="facebook"
        )
        print(f"\nFormatted prompt:")
        print(f"System: {prompt['system_prompt'][:100]}...")
        print(f"User: {prompt['user_prompt'][:100]}...")
    except Exception as e:
        print(f"Error formatting prompt: {e}")


if __name__ == "__main__":
    main()
