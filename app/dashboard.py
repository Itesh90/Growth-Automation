"""
Results Visualization Dashboard
Interactive dashboard for viewing A/B test results and model performance.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta

from experiments.ab_harness import ABTestHarness
from api.predictor import CTRPredictor
from utils.diversity_filter import DiversityFilter
from config import settings


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class Dashboard:
    """Interactive dashboard for headline optimization results."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.ab_harness = ABTestHarness()
        self.predictor = CTRPredictor()
        self.diversity_filter = DiversityFilter()
        
        # Set page config
        st.set_page_config(
            page_title="Ad Headline Optimizer Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_sidebar(self):
        """Render the sidebar with navigation."""
        st.sidebar.title("📊 Dashboard")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "A/B Tests", "Model Performance", "Headline Analysis", "Diversity Analysis"]
        )
        
        # Filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Campaign filter
        campaigns = self._get_campaigns()
        selected_campaigns = st.sidebar.multiselect(
            "Campaigns",
            options=campaigns,
            default=campaigns
        )
        
        return page, date_range, selected_campaigns
    
    def _get_campaigns(self) -> List[str]:
        """Get list of available campaigns."""
        # This would typically come from the database
        return ["E-commerce Sale", "SaaS Trial", "Mobile App", "All Campaigns"]
    
    def render_overview(self, date_range, selected_campaigns):
        """Render the overview page."""
        st.title("📊 Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Tests",
                value=len(self.ab_harness.tests),
                delta="+2 this week"
            )
        
        with col2:
            st.metric(
                label="Significant Tests",
                value=len([t for t in self.ab_harness.tests.values() if t.is_significant]),
                delta="+1 this week"
            )
        
        with col3:
            st.metric(
                label="Avg Lift",
                value=f"{np.mean([t.lift for t in self.ab_harness.tests.values()]):.1%}",
                delta="+0.5%"
            )
        
        with col4:
            st.metric(
                label="Model Accuracy",
                value="87.3%",
                delta="+2.1%"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_test_timeline()
        
        with col2:
            self._render_lift_distribution()
        
        # Recent tests table
        st.subheader("Recent A/B Tests")
        self._render_recent_tests_table()
    
    def _render_test_timeline(self):
        """Render test timeline chart."""
        st.subheader("Test Timeline")
        
        if not self.ab_harness.tests:
            st.info("No tests available")
            return
        
        # Prepare data
        test_data = []
        for test in self.ab_harness.tests.values():
            test_data.append({
                'Date': test.created_at.date() if test.created_at else datetime.now().date(),
                'Test ID': test.test_id,
                'Lift': test.lift,
                'Significant': test.is_significant
            })
        
        df = pd.DataFrame(test_data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='Date',
            y='Lift',
            color='Significant',
            hover_data=['Test ID'],
            title="Test Results Over Time",
            color_discrete_map={True: 'green', False: 'red'}
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Lift (%)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_lift_distribution(self):
        """Render lift distribution chart."""
        st.subheader("Lift Distribution")
        
        if not self.ab_harness.tests:
            st.info("No tests available")
            return
        
        lifts = [test.lift for test in self.ab_harness.tests.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=lifts,
            nbinsx=20,
            name='Lift Distribution',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Distribution of Test Lifts",
            xaxis_title="Lift (%)",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_recent_tests_table(self):
        """Render recent tests table."""
        if not self.ab_harness.tests:
            st.info("No tests available")
            return
        
        # Prepare data
        test_data = []
        for test in self.ab_harness.tests.values():
            test_data.append({
                'Test ID': test.test_id,
                'Variant A': test.variant_a[:50] + "..." if len(test.variant_a) > 50 else test.variant_a,
                'Variant B': test.variant_b[:50] + "..." if len(test.variant_b) > 50 else test.variant_b,
                'Lift': f"{test.lift:.1%}",
                'P-value': f"{test.p_value:.4f}",
                'Significant': "Yes" if test.is_significant else "No",
                'Status': test.status,
                'Created': test.created_at.strftime('%Y-%m-%d') if test.created_at else 'N/A'
            })
        
        df = pd.DataFrame(test_data)
        
        # Sort by creation date
        df = df.sort_values('Created', ascending=False)
        
        st.dataframe(df, use_container_width=True)
    
    def render_ab_tests(self, date_range, selected_campaigns):
        """Render the A/B tests page."""
        st.title("🧪 A/B Tests")
        
        # Test creation section
        with st.expander("Create New Test", expanded=False):
            self._render_test_creation_form()
        
        # Test results
        if self.ab_harness.tests:
            st.subheader("Test Results")
            
            # Test selection
            test_ids = list(self.ab_harness.tests.keys())
            selected_test = st.selectbox("Select Test", test_ids)
            
            if selected_test:
                self._render_test_details(selected_test)
        else:
            st.info("No A/B tests available. Create a test to get started.")
    
    def _render_test_creation_form(self):
        """Render test creation form."""
        with st.form("create_test_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                test_id = st.text_input("Test ID", value=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                variant_a = st.text_area("Variant A", placeholder="Enter headline for variant A")
                target_impressions = st.number_input("Target Impressions", min_value=1000, value=10000)
            
            with col2:
                campaign = st.selectbox("Campaign", self._get_campaigns())
                variant_b = st.text_area("Variant B", placeholder="Enter headline for variant B")
                significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
            
            submitted = st.form_submit_button("Create Test")
            
            if submitted:
                if variant_a and variant_b:
                    try:
                        config = self.ab_harness.create_test(
                            test_id=test_id,
                            variant_a=variant_a,
                            variant_b=variant_b,
                            target_impressions=target_impressions,
                            significance_level=significance_level
                        )
                        st.success(f"Test created successfully: {test_id}")
                    except Exception as e:
                        st.error(f"Error creating test: {e}")
                else:
                    st.error("Please fill in both variants")
    
    def _render_test_details(self, test_id: str):
        """Render detailed test results."""
        test = self.ab_harness.get_test_results(test_id)
        if not test:
            st.error("Test not found")
            return
        
        # Test summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CTR A", f"{test.ctr_a:.3f}")
        
        with col2:
            st.metric("CTR B", f"{test.ctr_b:.3f}")
        
        with col3:
            st.metric("Lift", f"{test.lift:.1%}")
        
        with col4:
            st.metric("P-value", f"{test.p_value:.4f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_ctr_comparison_chart(test)
        
        with col2:
            self._render_confidence_interval_chart(test)
        
        # Test report
        st.subheader("Test Report")
        report = self.ab_harness.generate_report(test_id)
        st.text(report)
    
    def _render_ctr_comparison_chart(self, test):
        """Render CTR comparison chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Variant A', 'Variant B'],
            y=[test.ctr_a, test.ctr_b],
            marker_color=['lightblue', 'lightcoral'],
            text=[f"{test.ctr_a:.3f}", f"{test.ctr_b:.3f}"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="CTR Comparison",
            xaxis_title="Variant",
            yaxis_title="Click-Through Rate",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_confidence_interval_chart(self, test):
        """Render confidence interval chart."""
        ci_lower, ci_upper = test.confidence_interval
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[0, 0],
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=10),
            name='Confidence Interval'
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Difference")
        
        fig.update_layout(
            title="Confidence Interval for CTR Difference",
            xaxis_title="CTR Difference",
            yaxis_title="",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance(self, date_range, selected_campaigns):
        """Render model performance page."""
        st.title("🤖 Model Performance")
        
        if not self.predictor.is_model_loaded():
            st.warning("No model loaded. Please train a model first.")
            return
        
        # Model info
        model_info = self.predictor.get_model_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_info.get('model_type', 'Unknown'))
        
        with col2:
            st.metric("Features", model_info.get('feature_count', 0))
        
        with col3:
            st.metric("Training Date", model_info.get('training_date', 'Unknown')[:10] if model_info.get('training_date') else 'Unknown')
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = self.predictor.get_feature_importance(top_n=20)
        
        if feature_importance:
            df = pd.DataFrame(feature_importance)
            
            fig = px.bar(
                df,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 20 Feature Importance"
            )
            
            fig.update_layout(
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data available")
        
        # Training metrics
        if model_info.get('training_metrics'):
            st.subheader("Training Metrics")
            metrics = model_info['training_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            
            with col3:
                st.metric("R²", f"{metrics.get('r2', 0):.4f}")
            
            with col4:
                st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
    
    def render_headline_analysis(self, date_range, selected_campaigns):
        """Render headline analysis page."""
        st.title("📝 Headline Analysis")
        
        # Headline input
        st.subheader("Analyze Headlines")
        
        headlines_input = st.text_area(
            "Enter headlines (one per line)",
            placeholder="Get 50% Off Today Only!\nTransform Your Life in 30 Days\nFree Shipping on All Orders",
            height=150
        )
        
        if headlines_input:
            headlines = [h.strip() for h in headlines_input.split('\n') if h.strip()]
            
            if headlines:
                # Analyze headlines
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("CTR Predictions")
                    if self.predictor.is_model_loaded():
                        predictions = self.predictor.predict_ctr(headlines)
                        
                        df = pd.DataFrame({
                            'Headline': headlines,
                            'Predicted CTR': [f"{p:.3f}" for p in predictions]
                        })
                        
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No model loaded for predictions")
                
                with col2:
                    st.subheader("Headline Ranking")
                    if self.predictor.is_model_loaded():
                        ranked = self.predictor.rank_headlines(headlines)
                        
                        df = pd.DataFrame(ranked)
                        df = df[['rank', 'headline', 'predicted_ctr']]
                        df.columns = ['Rank', 'Headline', 'CTR']
                        
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No model loaded for ranking")
                
                # Comparison analysis
                if self.predictor.is_model_loaded():
                    st.subheader("Comparison Analysis")
                    comparison = self.predictor.compare_headlines(headlines)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean CTR", f"{comparison['statistics']['mean_ctr']:.3f}")
                    
                    with col2:
                        st.metric("Max CTR", f"{comparison['statistics']['max_ctr']:.3f}")
                    
                    with col3:
                        st.metric("Range", f"{comparison['statistics']['range_ctr']:.3f}")
                    
                    # Best and worst headlines
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Best Headline")
                        best = comparison['best_headline']
                        st.success(f"**{best['headline']}**\n\nCTR: {best['predicted_ctr']:.3f}")
                    
                    with col2:
                        st.subheader("Worst Headline")
                        worst = comparison['worst_headline']
                        st.error(f"**{worst['headline']}**\n\nCTR: {worst['predicted_ctr']:.3f}")
    
    def render_diversity_analysis(self, date_range, selected_campaigns):
        """Render diversity analysis page."""
        st.title("🎯 Diversity Analysis")
        
        # Headline input
        st.subheader("Analyze Headline Diversity")
        
        headlines_input = st.text_area(
            "Enter headlines (one per line)",
            placeholder="Get 50% Off Today Only!\nGet 50% Off Today Only!\nTransform Your Life in 30 Days\nFree Shipping on All Orders",
            height=150
        )
        
        if headlines_input:
            headlines = [h.strip() for h in headlines_input.split('\n') if h.strip()]
            
            if headlines:
                # Diversity analysis
                analysis = self.diversity_filter.analyze_diversity(headlines)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Diversity Score", f"{analysis['diversity_score']:.3f}")
                
                with col2:
                    st.metric("Max Similarity", f"{analysis['max_similarity']:.3f}")
                
                with col3:
                    st.metric("Mean Similarity", f"{analysis['mean_similarity']:.3f}")
                
                with col4:
                    st.metric("Meets Threshold", "Yes" if analysis['meets_diversity_threshold'] else "No")
                
                # Filtered headlines
                st.subheader("Diversity Filtering")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Headlines")
                    st.write(f"Count: {len(headlines)}")
                    for i, headline in enumerate(headlines, 1):
                        st.write(f"{i}. {headline}")
                
                with col2:
                    st.subheader("Filtered Headlines")
                    filtered = self.diversity_filter.filter_similar_headlines(headlines)
                    st.write(f"Count: {len(filtered)}")
                    for i, headline in enumerate(filtered, 1):
                        st.write(f"{i}. {headline}")
                
                # Recommendations
                st.subheader("Recommendations")
                recommendations = self.diversity_filter.get_diversity_recommendations(headlines)
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.success("No recommendations - headlines are diverse enough!")
    
    def run(self):
        """Run the dashboard."""
        # Render sidebar
        page, date_range, selected_campaigns = self.render_sidebar()
        
        # Render main content
        if page == "Overview":
            self.render_overview(date_range, selected_campaigns)
        elif page == "A/B Tests":
            self.render_ab_tests(date_range, selected_campaigns)
        elif page == "Model Performance":
            self.render_model_performance(date_range, selected_campaigns)
        elif page == "Headline Analysis":
            self.render_headline_analysis(date_range, selected_campaigns)
        elif page == "Diversity Analysis":
            self.render_diversity_analysis(date_range, selected_campaigns)


def main():
    """Run the dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
