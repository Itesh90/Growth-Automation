"""
LLM-powered headline generation service with caching and retry logic.
Generates high-CTR headline variants using OpenAI/Anthropic APIs.
"""

import json
import hashlib
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import asyncio
from datetime import datetime

import openai
import anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis
from pydantic import BaseModel, Field, validator

from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@dataclass
class HeadlineResult:
    """Result of headline generation."""
    headline: str
    reasoning: str
    char_count: int
    generation_time: float
    cost_usd: float
    model_used: str


class HeadlineRequest(BaseModel):
    """Request model for headline generation."""
    base_copy: str = Field(..., min_length=10, max_length=500)
    tone: str = Field(..., pattern="^(Professional|Casual|Urgent|Playful)$")
    n_variants: int = Field(default=10, ge=1, le=20)
    max_length: int = Field(default=60, ge=20, le=100)
    placement_type: str = Field(default="Email Subject", pattern="^(Email Subject|Social Ad|Display Ad)$")
    
    @validator('base_copy')
    def validate_base_copy(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Base copy must be at least 10 characters")
        return v.strip()


class HeadlineGenerator:
    """LLM-powered headline generation with caching and fallback."""
    
    def __init__(self):
        self.redis_client = None
        self._init_redis()
        self._init_llm_clients()
        
    def _init_redis(self):
        """Initialize Redis client for caching."""
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None
    
    def _init_llm_clients(self):
        """Initialize LLM API clients."""
        if settings.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            
        if settings.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except Exception as e:
                logger.warning(f"Anthropic client initialization failed: {e}")
                self.anthropic_client = None
        else:
            self.anthropic_client = None
            
        if settings.gemini_api_key:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(settings.gemini_model)
            except Exception as e:
                logger.warning(f"Gemini client initialization failed: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            
        if not self.openai_client and not self.anthropic_client and not self.gemini_model:
            logger.warning("No LLM API keys configured. Only sample data will be available.")
            # Don't raise error - allow system to work with sample data
    
    def _get_cache_key(self, request: HeadlineRequest) -> str:
        """Generate cache key for request."""
        content = f"{request.base_copy}:{request.tone}:{request.n_variants}:{request.max_length}:{request.placement_type}"
        return f"gen:{hashlib.md5(content.encode()).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached generation result."""
        if not self.redis_client:
            return None
            
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache generation result."""
        if not self.redis_client:
            return
            
        try:
            self.redis_client.setex(
                cache_key, 
                settings.redis_ttl, 
                json.dumps(result)
            )
            logger.info(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _build_prompt(self, request: HeadlineRequest) -> str:
        """Build the few-shot prompt for headline generation."""
        return f"""You are an expert marketing copywriter specializing in high-CTR headlines.

Given ad copy, generate {request.n_variants} headline variations that:
- Are ≤{request.max_length} characters
- Match the {request.tone} tone
- Include action verbs and concrete benefits
- Avoid jargon, hype, or false urgency
- Are optimized for {request.placement_type}

Examples of high-performing headlines:
1. Base: "Project management software" → "Ship projects 2x faster with AI planning"
2. Base: "Online yoga classes" → "Flexible yoga—15 min sessions you'll actually do"
3. Base: "Tax filing service" → "Max refund guaranteed. File in under 20 minutes."
4. Base: "Fitness app" → "Get fit in 20 minutes. No gym required."
5. Base: "Email marketing tool" → "Double your open rates with AI subject lines"

Base copy: {request.base_copy}
Tone: {request.tone}
Placement: {request.placement_type}

Return valid JSON array only:
[{{"headline": "...", "reasoning": "why this works"}}]"""
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((Exception,))
    )
    async def _generate_with_openai(self, prompt: str, request: HeadlineRequest) -> List[Dict[str, Any]]:
        """Generate headlines using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        start_time = datetime.now()
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert marketing copywriter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            headlines = json.loads(content)
            
            # Validate and enhance results
            enhanced_headlines = []
            for item in headlines:
                if isinstance(item, dict) and "headline" in item:
                    enhanced_headlines.append({
                        "headline": item["headline"],
                        "reasoning": item.get("reasoning", "Generated by AI"),
                        "char_count": len(item["headline"]),
                        "generation_time": generation_time,
                        "cost_usd": self._calculate_openai_cost(response.usage.total_tokens),
                        "model_used": settings.openai_model
                    })
            
            return enhanced_headlines
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise ValueError("Invalid JSON response from OpenAI")
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((Exception,))
    )
    async def _generate_with_anthropic(self, prompt: str, request: HeadlineRequest) -> List[Dict[str, Any]]:
        """Generate headlines using Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        start_time = datetime.now()
        
        try:
            response = self.anthropic_client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Parse response
            content = response.content[0].text.strip()
            
            # Extract JSON from response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            headlines = json.loads(content)
            
            # Validate and enhance results
            enhanced_headlines = []
            for item in headlines:
                if isinstance(item, dict) and "headline" in item:
                    enhanced_headlines.append({
                        "headline": item["headline"],
                        "reasoning": item.get("reasoning", "Generated by AI"),
                        "char_count": len(item["headline"]),
                        "generation_time": generation_time,
                        "cost_usd": self._calculate_anthropic_cost(response.usage.input_tokens, response.usage.output_tokens),
                        "model_used": settings.anthropic_model
                    })
            
            return enhanced_headlines
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Anthropic response: {e}")
            raise ValueError("Invalid JSON response from Anthropic")
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((Exception,))
    )
    async def _generate_with_gemini(self, prompt: str, request: HeadlineRequest) -> List[Dict[str, Any]]:
        """Generate headlines using Gemini API."""
        if not self.gemini_model:
            raise ValueError("Gemini model not initialized")
        
        start_time = datetime.now()
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.gemini_temperature,
                    max_output_tokens=settings.gemini_max_tokens,
                )
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Parse response
            content = response.text.strip()
            
            # Extract JSON from response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            headlines = json.loads(content)
            
            # Validate and enhance results
            enhanced_headlines = []
            for item in headlines:
                if isinstance(item, dict) and "headline" in item:
                    enhanced_headlines.append({
                        "headline": item["headline"],
                        "reasoning": item.get("reasoning", "Generated by AI"),
                        "char_count": len(item["headline"]),
                        "generation_time": generation_time,
                        "cost_usd": self._calculate_gemini_cost(len(prompt), len(content)),
                        "model_used": settings.gemini_model
                    })
            
            return enhanced_headlines
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            raise ValueError("Invalid JSON response from Gemini")
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def _calculate_openai_cost(self, total_tokens: int) -> float:
        """Calculate cost for OpenAI API usage."""
        return total_tokens * settings.cost_per_token_openai
    
    def _calculate_anthropic_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic API usage."""
        # Simplified cost calculation - adjust based on actual pricing
        return (input_tokens + output_tokens) * settings.cost_per_token_anthropic
    
    def _calculate_gemini_cost(self, input_chars: int, output_chars: int) -> float:
        """Calculate cost for Gemini API usage."""
        # Gemini pricing is per character - simplified calculation
        # Adjust based on actual Gemini pricing
        input_cost = input_chars * 0.0000005  # $0.50 per 1M characters
        output_cost = output_chars * 0.0000015  # $1.50 per 1M characters
        return input_cost + output_cost
    
    def _generate_sample_headlines(self, request: HeadlineRequest) -> List[Dict[str, Any]]:
        """Generate sample headlines when no API keys are available."""
        sample_headlines = {
            "Professional": [
                f"Transform Your {request.base_copy.split()[-1]} with Our Solution",
                f"Professional {request.base_copy.split()[-1]} Services",
                f"Expert {request.base_copy.split()[-1]} Solutions",
                f"Premium {request.base_copy.split()[-1]} Platform",
                f"Advanced {request.base_copy.split()[-1]} Technology"
            ],
            "Casual": [
                f"Hey! Check out this awesome {request.base_copy.split()[-1]}",
                f"Your {request.base_copy.split()[-1]} just got way better",
                f"Finally, a {request.base_copy.split()[-1]} that actually works",
                f"Ready to upgrade your {request.base_copy.split()[-1]}?",
                f"This {request.base_copy.split()[-1]} is a game changer"
            ],
            "Urgent": [
                f"Limited Time: {request.base_copy.split()[-1]} Offer",
                f"Don't Miss Out on {request.base_copy.split()[-1]}",
                f"Last Chance for {request.base_copy.split()[-1]}",
                f"Urgent: {request.base_copy.split()[-1]} Update",
                f"Act Now: {request.base_copy.split()[-1]} Deal"
            ],
            "Playful": [
                f"🎉 Your {request.base_copy.split()[-1]} adventure starts here!",
                f"✨ Magical {request.base_copy.split()[-1]} experience",
                f"🚀 Blast off with {request.base_copy.split()[-1]}",
                f"🎯 Hit the bullseye with {request.base_copy.split()[-1]}",
                f"🌈 Colorful {request.base_copy.split()[-1]} solutions"
            ]
        }
        
        tone_headlines = sample_headlines.get(request.tone, sample_headlines["Professional"])
        selected_headlines = tone_headlines[:request.n_variants]
        
        return [
            {
                "headline": headline,
                "reasoning": f"Sample headline for {request.tone} tone",
                "char_count": len(headline),
                "generation_time": 0.1,
                "cost_usd": 0.0,
                "model_used": "sample_data"
            }
            for headline in selected_headlines
        ]
    
    async def generate_headlines(self, request: HeadlineRequest) -> List[HeadlineResult]:
        """
        Generate headline variants with caching and fallback logic.
        
        Args:
            request: Headline generation request
            
        Returns:
            List of generated headlines with metadata
            
        Raises:
            ValueError: If request validation fails
            RuntimeError: If all LLM providers fail
        """
        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return [HeadlineResult(**item) for item in cached_result]
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Try only the model with valid API key (prioritize Gemini since user has it)
        headlines = None
        last_error = None
        
        # Check which API keys are actually valid (not placeholder values)
        has_valid_openai = settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here"
        has_valid_anthropic = settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_api_key_here"
        has_valid_gemini = settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here"
        
        # Use Gemini first since user has a valid key
        if has_valid_gemini and self.gemini_model:
            try:
                headlines = await self._generate_with_gemini(prompt, request)
                logger.info(f"Generated {len(headlines)} headlines using Gemini")
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini generation failed: {e}")
        
        # Fallback to other valid APIs only if Gemini fails
        if not headlines and has_valid_openai and self.openai_client:
            try:
                headlines = await self._generate_with_openai(prompt, request)
                logger.info(f"Generated {len(headlines)} headlines using OpenAI")
            except Exception as e:
                last_error = e
                logger.warning(f"OpenAI generation failed: {e}")
        
        if not headlines and has_valid_anthropic and self.anthropic_client:
            try:
                headlines = await self._generate_with_anthropic(prompt, request)
                logger.info(f"Generated {len(headlines)} headlines using Anthropic")
            except Exception as e:
                last_error = e
                logger.warning(f"Anthropic generation failed: {e}")
        
        if not headlines:
            # Only use sample data if no valid API keys are available
            if not (has_valid_openai or has_valid_anthropic or has_valid_gemini):
                logger.info("No valid API keys available, generating sample headlines")
                headlines = self._generate_sample_headlines(request)
            else:
                raise RuntimeError(f"All available LLM providers failed. Last error: {last_error}")
        
        # Validate results
        if len(headlines) < request.n_variants:
            logger.warning(f"Generated {len(headlines)} headlines, requested {request.n_variants}")
        
        # Filter by max length
        filtered_headlines = [
            h for h in headlines 
            if h["char_count"] <= request.max_length
        ]
        
        if not filtered_headlines:
            raise ValueError("No headlines generated within length constraints")
        
        # Cache results
        self._cache_result(cache_key, filtered_headlines)
        
        # Convert to HeadlineResult objects
        results = [HeadlineResult(**item) for item in filtered_headlines]
        
        # Log generation metrics
        total_cost = sum(h.cost_usd for h in results)
        avg_time = sum(h.generation_time for h in results) / len(results)
        
        logger.info(f"Generated {len(results)} headlines. Total cost: ${total_cost:.4f}, Avg time: {avg_time:.2f}s")
        
        return results


# Global generator instance (lazy initialization)
_generator = None

def get_generator():
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = HeadlineGenerator()
    return _generator


async def generate_headlines(
    base_copy: str,
    tone: str = "Professional",
    n: int = 10,
    max_length: int = 60,
    placement_type: str = "Email Subject"
) -> List[HeadlineResult]:
    """
    Convenience function for headline generation.
    
    Args:
        base_copy: Base ad copy to generate headlines from
        tone: Tone for headlines (Professional, Casual, Urgent, Playful)
        n: Number of variants to generate
        max_length: Maximum character length
        placement_type: Where headlines will be used
        
    Returns:
        List of generated headlines
    """
    request = HeadlineRequest(
        base_copy=base_copy,
        tone=tone,
        n_variants=n,
        max_length=max_length,
        placement_type=placement_type
    )
    
    generator = get_generator()
    return await generator.generate_headlines(request)
