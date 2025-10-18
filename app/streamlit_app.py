"""
Main Streamlit UI for the Ad Headline Optimizer.
Provides a clean, professional interface for headline generation and optimization.
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
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json

# Import our modules
try:
    from api.generator import generate_headlines, HeadlineRequest
    from api.features_lightweight import extract_headlines_features_lightweight, calculate_predicted_ctr_lightweight
    from models.ctr_simulator import simulate_ab_test
    from config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please make sure you're running this from the project root directory.")
    st.stop()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🚀 Ad Headline Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .headline-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'headlines_generated' not in st.session_state:
        st.session_state.headlines_generated = False
    if 'generated_headlines' not in st.session_state:
        st.session_state.generated_headlines = []
    if 'features_df' not in st.session_state:
        st.session_state.features_df = pd.DataFrame()
    if 'ab_test_results' not in st.session_state:
        st.session_state.ab_test_results = pd.DataFrame()
    if 'generation_cost' not in st.session_state:
        st.session_state.generation_cost = 0.0
    if 'generation_time' not in st.session_state:
        st.session_state.generation_time = 0.0


def display_header():
    """Display the main header."""
    st.markdown('<h1 class="main-header">🚀 Ad Headline Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Generate high-CTR headline variants using AI, predict performance, and validate with A/B testing
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with input controls."""
    st.sidebar.header("🎯 Configuration")
    
    # Base copy input
    st.sidebar.subheader("Base Ad Copy")
    base_copy = st.sidebar.text_area(
        "Enter your base ad copy:",
        value="Get fit at home in 30 days",
        height=100,
        max_chars=500,
        help="The base message you want to turn into compelling headlines"
    )
    
    # Tone selection
    st.sidebar.subheader("Tone & Style")
    tone = st.sidebar.selectbox(
        "Select tone:",
        ["Professional", "Casual", "Urgent", "Playful"],
        index=0,
        help="The emotional tone for your headlines"
    )
    
    # Placement type
    placement = st.sidebar.radio(
        "Placement type:",
        ["Email Subject", "Social Ad", "Display Ad"],
        index=0,
        help="Where these headlines will be used"
    )
    
    # Generation parameters
    st.sidebar.subheader("Generation Settings")
    n_variants = st.sidebar.slider(
        "Number of variants:",
        min_value=5,
        max_value=20,
        value=10,
        help="How many headline variants to generate"
    )
    
    max_length = st.sidebar.slider(
        "Max character length:",
        min_value=20,
        max_value=100,
        value=60,
        help="Maximum character length for headlines"
    )
    
    # Diversity filter
    st.sidebar.subheader("Diversity Filter")
    apply_diversity_filter = st.sidebar.checkbox(
        "Apply diversity filter",
        value=True,
        help="Remove similar headlines to ensure variety"
    )
    
    diversity_threshold = st.sidebar.slider(
        "Similarity threshold:",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.1,
        help="Lower values = more diverse headlines"
    )
    
    return {
        'base_copy': base_copy,
        'tone': tone,
        'placement': placement,
        'n_variants': n_variants,
        'max_length': max_length,
        'apply_diversity_filter': apply_diversity_filter,
        'diversity_threshold': diversity_threshold
    }


async def generate_headlines_async(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Async wrapper for headline generation."""
    try:
        headlines = await generate_headlines(
            base_copy=params['base_copy'],
            tone=params['tone'],
            n=params['n_variants'],
            max_length=params['max_length'],
            placement_type=params['placement']
        )
        
        # Convert to dict format
        return [
            {
                'headline': h.headline,
                'reasoning': h.reasoning,
                'char_count': h.char_count,
                'generation_time': h.generation_time,
                'cost_usd': h.cost_usd,
                'model_used': h.model_used
            }
            for h in headlines
        ]
    except Exception as e:
        logger.error(f"Headline generation failed: {e}")
        raise


def display_generation_button(params: Dict[str, Any]):
    """Display the generate button and handle generation."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Generate Headlines", type="primary", use_container_width=True):
            if not params['base_copy'].strip():
                st.error("Please enter base ad copy")
                return
            
            # Show progress
            with st.spinner("Generating headlines with AI..."):
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
                    
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    logger.error(f"Generation error: {e}")


def apply_diversity_filter(headlines: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Apply diversity filter to remove similar headlines."""
    if len(headlines) <= 3:
        return headlines
    
    # Simple diversity filter based on character similarity
    filtered = [headlines[0]]  # Start with first headline
    
    for headline in headlines[1:]:
        is_diverse = True
        for existing in filtered:
            # Calculate simple similarity (can be improved with embeddings)
            similarity = len(set(headline['headline'].lower().split()) & 
                           set(existing['headline'].lower().split())) / \
                        len(set(headline['headline'].lower().split()) | 
                           set(existing['headline'].lower().split()))
            
            if similarity > threshold:
                is_diverse = False
                break
        
        if is_diverse:
            filtered.append(headline)
    
    return filtered


def extract_features_for_headlines(headlines: List[Dict[str, Any]], placement: str) -> pd.DataFrame:
    """Extract features for generated headlines using lightweight methods."""
    headline_texts = [h['headline'] for h in headlines]
    context = {'placement_type': placement}
    
    # Use lightweight feature extraction
    features_df = extract_headlines_features_lightweight(headline_texts, context)
    
    # Add metadata
    for i, headline in enumerate(headlines):
        features_df.loc[i, 'reasoning'] = headline['reasoning']
        features_df.loc[i, 'cost_usd'] = headline['cost_usd']
        features_df.loc[i, 'model_used'] = headline['model_used']
    
    return features_df


def calculate_predicted_ctr(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate predicted CTR using lightweight heuristics."""
    return calculate_predicted_ctr_lightweight(features_df)


def display_headlines_table(features_df: pd.DataFrame, apply_filter: bool, threshold: float):
    """Display the headlines table with predictions."""
    if features_df.empty:
        return
    
    # Apply diversity filter if requested
    if apply_filter and len(features_df) > 3:
        filtered_df = apply_diversity_filter(features_df.to_dict('records'), threshold)
        filtered_df = pd.DataFrame(filtered_df)
        st.info(f"🔍 Diversity filter applied: {len(features_df)} → {len(filtered_df)} headlines")
    else:
        filtered_df = features_df
    
    st.subheader("📊 Generated Headlines & Predictions")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Headlines", len(filtered_df))
    
    with col2:
        avg_ctr = filtered_df['predicted_ctr_pct'].mean()
        st.metric("Avg Predicted CTR", f"{avg_ctr:.2f}%")
    
    with col3:
        best_ctr = filtered_df['predicted_ctr_pct'].max()
        st.metric("Best Predicted CTR", f"{best_ctr:.2f}%")
    
    with col4:
        total_cost = filtered_df['cost_usd'].sum()
        st.metric("Total Cost", f"${total_cost:.4f}")
    
    # Display table
    display_df = filtered_df[['headline', 'predicted_ctr_pct', 'char_count', 'reasoning']].copy()
    display_df.columns = ['Headline', 'Predicted CTR (%)', 'Characters', 'AI Reasoning']
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df = display_df[['Rank', 'Headline', 'Predicted CTR (%)', 'Characters', 'AI Reasoning']]
    
    # Format the table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Add copy buttons
    st.subheader("📋 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy Top 3", use_container_width=True):
            top_3 = filtered_df.head(3)['headline'].tolist()
            st.code('\n'.join(top_3), language=None)
    
    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"headlines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🧪 Run A/B Test", use_container_width=True):
            if len(filtered_df) >= 2:
                st.session_state.run_ab_test = True
                st.rerun()
            else:
                st.warning("Need at least 2 headlines for A/B testing")


def display_ab_test_simulation(features_df: pd.DataFrame):
    """Display A/B test simulation results."""
    if not st.session_state.get('run_ab_test', False):
        return
    
    st.subheader("🧪 A/B Test Simulation")
    
    # Select headlines for testing
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Control Headline:**")
        control_options = features_df['headline'].tolist()
        control_idx = st.selectbox("Control:", range(len(control_options)), format_func=lambda x: control_options[x])
    
    with col2:
        st.write("**Select Variant Headlines:**")
        variant_options = [h for i, h in enumerate(control_options) if i != control_idx]
        selected_variants = st.multiselect(
            "Variants (select 1-3):",
            variant_options,
            default=variant_options[:2] if len(variant_options) >= 2 else variant_options
        )
    
    if st.button("🚀 Run A/B Test Simulation", type="primary"):
        if not selected_variants:
            st.error("Please select at least one variant headline")
            return
        
        with st.spinner("Running A/B test simulation..."):
            try:
                # Run simulation
                control_headline = control_options[control_idx]
                results = simulate_ab_test(
                    control_headline=control_headline,
                    variant_headlines=selected_variants,
                    n_impressions=10000
                )
                
                st.session_state.ab_test_results = results
                
                # Display results
                st.success("✅ A/B test simulation completed!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                control_result = results[results['is_control']].iloc[0]
                
                with col1:
                    st.metric("Control CTR", f"{control_result['observed_ctr']*100:.2f}%")
                
                with col2:
                    best_variant = results[~results['is_control']].loc[results[~results['is_control']]['observed_ctr'].idxmax()]
                    st.metric("Best Variant CTR", f"{best_variant['observed_ctr']*100:.2f}%")
                
                with col3:
                    uplift = ((best_variant['observed_ctr'] - control_result['observed_ctr']) / control_result['observed_ctr']) * 100
                    st.metric("Uplift", f"{uplift:.1f}%")
                
                with col4:
                    st.metric("Total Impressions", f"{results['impressions'].sum():,}")
                
                # Results table
                st.subheader("📊 A/B Test Results")
                display_results = results[['headline', 'variant_type', 'observed_ctr', 'impressions', 'clicks']].copy()
                display_results['observed_ctr'] = (display_results['observed_ctr'] * 100).round(2)
                display_results.columns = ['Headline', 'Type', 'CTR (%)', 'Impressions', 'Clicks']
                
                st.dataframe(display_results, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    results,
                    x='variant_type',
                    y='observed_ctr',
                    title='CTR by Variant',
                    labels={'observed_ctr': 'CTR', 'variant_type': 'Variant'},
                    color='is_control',
                    color_discrete_map={True: '#1f77b4', False: '#ff7f0e'}
                )
                fig.update_layout(yaxis_tickformat='.2%')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ A/B test simulation failed: {str(e)}")
                logger.error(f"A/B test error: {e}")


def display_footer():
    """Display footer with additional information."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🚀 Ad Headline Optimizer | Built with Streamlit, OpenAI, and LightGBM</p>
        <p>Generate high-performing headlines in seconds, not weeks</p>
    </div>
    """, unsafe_allow_html=True)


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
        # Process generated headlines
        if st.session_state.features_df.empty:
            with st.spinner("Extracting features and predicting CTR..."):
                try:
                    st.session_state.features_df = extract_features_for_headlines(
                        st.session_state.generated_headlines,
                        params['placement']
                    )
                    st.session_state.features_df = calculate_predicted_ctr(st.session_state.features_df)
                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    # Create a simple fallback DataFrame
                    headlines_data = []
                    for i, headline in enumerate(st.session_state.generated_headlines):
                        headlines_data.append({
                            'headline': headline['headline'],
                            'predicted_ctr_pct': np.random.uniform(1.5, 4.0),  # Random CTR
                            'char_count': headline['char_count'],
                            'reasoning': headline['reasoning'],
                            'cost_usd': headline['cost_usd'],
                            'model_used': headline['model_used']
                        })
                    st.session_state.features_df = pd.DataFrame(headlines_data)
        
        # Display results - always show headlines
        if not st.session_state.features_df.empty:
            display_headlines_table(
                st.session_state.features_df,
                params['apply_diversity_filter'],
                params['diversity_threshold']
            )
        else:
            # Fallback: Display headlines directly
            st.subheader("📊 Generated Headlines")
            
            # Show which model was used
            if st.session_state.generated_headlines:
                model_used = st.session_state.generated_headlines[0]['model_used']
                st.info(f"🤖 Generated using: **{model_used}**")
            
            for i, headline in enumerate(st.session_state.generated_headlines, 1):
                with st.container():
                    st.write(f"**{i}. {headline['headline']}**")
                    st.write(f"*Reasoning: {headline['reasoning']}*")
                    st.write(f"*Model: {headline['model_used']} | Cost: ${headline['cost_usd']:.4f}*")
                    st.divider()
        
        # A/B test simulation
        display_ab_test_simulation(st.session_state.features_df)
        
        # Reset button
        if st.button("🔄 Generate New Headlines"):
            st.session_state.headlines_generated = False
            st.session_state.generated_headlines = []
            st.session_state.features_df = pd.DataFrame()
            st.session_state.ab_test_results = pd.DataFrame()
            st.rerun()
    
    display_footer()


if __name__ == "__main__":
    main()
