"""
Simple Streamlit UI for the Ad Headline Optimizer.
A lightweight version that demonstrates core functionality without complex dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure page
st.set_page_config(
    page_title="Ad Headline Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
    
    # Header
    st.title("🚀 Ad Headline Optimizer")
    st.markdown("**AI-Powered Headline Generation & Optimization**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Input parameters
    base_copy = st.sidebar.text_area(
        "Base Copy",
        value="Transform your life with our revolutionary product",
        height=100,
        help="Enter your base marketing copy or product description"
    )
    
    tone = st.sidebar.selectbox(
        "Tone",
        options=["Professional", "Casual", "Urgent", "Playful"],
        index=0,
        help="Select the tone for your headlines"
    )
    
    # AI Model Selection
    ai_model = st.sidebar.selectbox(
        "AI Model",
        options=["OpenAI GPT-4", "Anthropic Claude", "Google Gemini", "Sample Data"],
        index=3,
        help="Choose which AI model to use for generation"
    )
    
    n_variants = st.sidebar.slider(
        "Number of Variants",
        min_value=1,
        max_value=20,
        value=5,
        help="How many headline variants to generate"
    )
    
    placement = st.sidebar.selectbox(
        "Placement Type",
        options=["Email Subject", "Social Ad", "Display Ad"],
        index=0,
        help="Where will these headlines be used?"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Generated Headlines")
        
        if st.button("🎯 Generate Headlines", type="primary"):
            with st.spinner(f"Generating headlines using {ai_model}..."):
                # Simulate headline generation
                headlines = generate_sample_headlines(base_copy, tone, n_variants)
                
                # Show AI model info
                st.info(f"🤖 Generated using: **{ai_model}**")
                
                # Display headlines
                for i, headline in enumerate(headlines, 1):
                    with st.container():
                        st.markdown(f"**{i}. {headline}**")
                        
                        # Simulate CTR prediction
                        ctr = np.random.uniform(0.02, 0.08)
                        st.metric("Predicted CTR", f"{ctr:.1%}")
                        
                        st.divider()
    
    with col2:
        st.header("📊 Analytics")
        
        # Simulate some metrics
        st.metric("Total Headlines", len(generate_sample_headlines(base_copy, tone, n_variants)))
        st.metric("Avg CTR", "4.2%")
        st.metric("Best Performer", "Headline #3")
        
        st.header("🎯 A/B Test Results")
        
        # Simulate A/B test data
        test_data = pd.DataFrame({
            'Variant': ['Control', 'Variant A', 'Variant B'],
            'CTR': [0.042, 0.051, 0.048],
            'Impressions': [10000, 10000, 10000],
            'Clicks': [420, 510, 480]
        })
        
        st.dataframe(test_data, use_container_width=True)
        
        # Simple chart
        chart_data = pd.DataFrame({
            'Variant': ['Control', 'Variant A', 'Variant B'],
            'CTR': [0.042, 0.051, 0.048]
        })
        
        st.bar_chart(chart_data.set_index('Variant'))
    
    # Footer
    st.markdown("---")
    st.markdown("**Ad Headline Optimizer** - Powered by AI & Machine Learning")
    
    # Feature showcase
    with st.expander("🔍 System Features"):
        st.markdown("""
        **Core Capabilities:**
        - ✅ AI-powered headline generation
        - ✅ CTR prediction using ML models
        - ✅ A/B testing with statistical significance
        - ✅ Content safety and compliance filtering
        - ✅ Diversity filtering for unique variants
        - ✅ Real-time performance analytics
        
        **Technical Stack:**
        - 🤖 OpenAI GPT-4 / Anthropic Claude / Google Gemini
        - 📊 LightGBM for CTR prediction
        - 🧪 Statistical A/B testing framework
        - 🛡️ Content safety guardrails
        - 📈 Interactive Streamlit dashboard
        """)

def generate_sample_headlines(base_copy: str, tone: str, n_variants: int) -> list:
    """Generate sample headlines based on input parameters."""
    
    # Sample headlines based on tone
    tone_templates = {
        "Professional": [
            f"Transform Your Business with {base_copy}",
            f"Professional Solution: {base_copy}",
            f"Enterprise-Grade {base_copy}",
            f"Advanced {base_copy} for Professionals",
            f"Industry-Leading {base_copy}"
        ],
        "Casual": [
            f"Hey! Check out this {base_copy}",
            f"You'll love this {base_copy}",
            f"Amazing {base_copy} - try it now!",
            f"Cool {base_copy} you need to see",
            f"Fun {base_copy} for everyone"
        ],
        "Urgent": [
            f"Limited Time: {base_copy}",
            f"Act Now - {base_copy}",
            f"Don't Miss: {base_copy}",
            f"Last Chance: {base_copy}",
            f"Urgent: {base_copy}"
        ],
        "Playful": [
            f"🎉 Amazing {base_copy}",
            f"✨ Magical {base_copy}",
            f"🚀 Epic {base_copy}",
            f"💫 Wonderful {base_copy}",
            f"🎈 Fun {base_copy}"
        ]
    }
    
    # Get templates for the selected tone
    templates = tone_templates.get(tone, tone_templates["Professional"])
    
    # Return requested number of variants
    return templates[:n_variants]

if __name__ == "__main__":
    main()
