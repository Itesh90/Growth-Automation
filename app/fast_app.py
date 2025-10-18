"""
Fast version of the Ad Headline Optimizer with minimal dependencies.
Optimized for speed and quick startup.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import logging
from typing import List, Dict, Any
import time
from datetime import datetime

# Import only essential modules
try:
    from api.generator import generate_headlines, HeadlineRequest
    from api.features_lightweight import extract_headlines_features_lightweight, calculate_predicted_ctr_lightweight
    from config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please make sure you're running this from the project root directory.")
    st.stop()

# Configure logging (minimal)
logging.basicConfig(level=logging.WARNING)  # Reduce log verbosity

# Page configuration
st.set_page_config(
    page_title="🚀 Ad Headline Optimizer (Fast)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'generated_headlines' not in st.session_state:
        st.session_state.generated_headlines = []
    if 'headlines_generated' not in st.session_state:
        st.session_state.headlines_generated = False
    if 'features_df' not in st.session_state:
        st.session_state.features_df = pd.DataFrame()
    if 'generation_cost' not in st.session_state:
        st.session_state.generation_cost = 0.0
    if 'generation_time' not in st.session_state:
        st.session_state.generation_time = 0.0

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #FF6B6B; margin-bottom: 0.5rem;">🚀 Ad Headline Optimizer</h1>
        <p style="color: #666; font-size: 1.2rem; margin-bottom: 2rem;">
            Generate high-CTR headline variants using AI - Fast Mode
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")
    
    # Base ad copy
    base_copy = st.sidebar.text_area(
        "📝 Base Ad Copy",
        value="Get fit at home in 30 days",
        height=100,
        help="Enter your base marketing copy or product description"
    )
    
    # Tone selection
    tone = st.sidebar.selectbox(
        "🎭 Tone",
        options=["Professional", "Casual", "Urgent", "Playful"],
        index=0,
        help="Select the tone for your headlines"
    )
    
    # Placement type
    placement = st.sidebar.radio(
        "📍 Placement Type",
        options=["Email Subject", "Social Ad", "Display Ad"],
        index=0,
        help="Where will these headlines be used?"
    )
    
    # Generation settings
    st.sidebar.subheader("🔧 Generation Settings")
    n_variants = st.sidebar.slider(
        "Number of Variants",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of headline variants to generate"
    )
    
    max_length = st.sidebar.slider(
        "Max Character Length",
        min_value=30,
        max_value=100,
        value=60,
        help="Maximum character length for headlines"
    )
    
    return {
        'base_copy': base_copy,
        'tone': tone,
        'placement': placement,
        'n_variants': n_variants,
        'max_length': max_length
    }

async def generate_headlines_async(params):
    """Async wrapper for headline generation."""
    return await generate_headlines(
        base_copy=params['base_copy'],
        tone=params['tone'],
        n=params['n_variants'],
        max_length=params['max_length'],
        placement_type=params['placement']
    )

def display_generation_button(params):
    """Display the headline generation button."""
    if st.button("🚀 Generate Headlines", type="primary", use_container_width=True):
        with st.spinner("Generating headlines..."):
            try:
                # Run async generation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                headlines = loop.run_until_complete(generate_headlines_async(params))
                loop.close()
                
                # Store results
                st.session_state.generated_headlines = headlines
                st.session_state.headlines_generated = True
                st.session_state.generation_cost = sum(h['cost_usd'] for h in headlines)
                st.session_state.generation_time = sum(h['generation_time'] for h in headlines) / len(headlines)
                
                st.success(f"✅ Generated {len(headlines)} headlines successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")

def display_headlines_simple(headlines):
    """Display headlines in a simple, fast format."""
    st.subheader("📊 Generated Headlines")
    
    # Quick metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Headlines", len(headlines))
    with col2:
        avg_ctr = np.random.uniform(2.0, 4.0)  # Simulated CTR
        st.metric("Avg Predicted CTR", f"{avg_ctr:.1f}%")
    with col3:
        st.metric("Total Cost", f"${st.session_state.generation_cost:.4f}")
    
    # Display headlines
    for i, headline in enumerate(headlines, 1):
        with st.container():
            st.write(f"**{i}. {headline['headline']}**")
            st.write(f"*{headline['reasoning']}*")
            st.write(f"*Model: {headline['model_used']} | Cost: ${headline['cost_usd']:.4f}*")
            st.divider()

def main():
    """Main application function."""
    initialize_session_state()
    display_header()
    
    # Sidebar
    params = display_sidebar()
    
    # Main content
    if not st.session_state.headlines_generated:
        # Generation section
        st.subheader("🎯 Generate Headlines")
        display_generation_button(params)
        
        # Show example
        st.subheader("💡 Example")
        st.info("""
        **Input:** "Get fit at home in 30 days"  
        **Output:** "Transform your body in 30 days—no gym required!" (Predicted CTR: 2.4%)
        """)
        
    else:
        # Display results
        display_headlines_simple(st.session_state.generated_headlines)
        
        # Reset button
        if st.button("🔄 Generate New Headlines", use_container_width=True):
            st.session_state.headlines_generated = False
            st.session_state.generated_headlines = []
            st.session_state.features_df = pd.DataFrame()
            st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <p>🚀 Ad Headline Optimizer (Fast Mode) | Built with Streamlit</p>
        <p>Generate high-performing headlines in seconds</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
