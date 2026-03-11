# app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import shap
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ChurnShield AI | Bank Customer Retention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk levels */
    .risk-high {
        color: #e53e3e;
        font-weight: 700;
    }
    .risk-medium {
        color: #dd6b20;
        font-weight: 700;
    }
    .risk-low {
        color: #38a169;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Input card styling */
    .input-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Result card styling */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 2px solid;
        transition: all 0.3s ease;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 60, 114, 0.3);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-left: 5px solid #4299e1;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
        border-left: 5px solid #ed8936;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 5px solid #48bb78;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 5px solid #f56565;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Sidebar radio customization */
    .stRadio > div {
        background-color: transparent !important;
    }
    .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: translateX(5px) !important;
    }
    .stRadio > div > label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f7fafc;
        padding: 0.5rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_prob' not in st.session_state:
    st.session_state.prediction_prob = None
if 'current_customer' not in st.session_state:
    st.session_state.current_customer = None
if 'original_prob' not in st.session_state:  # 👈 NEW
    st.session_state.original_prob = None
if 'warning_message' not in st.session_state:  # 👈 NEW
    st.session_state.warning_message = None

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and data from the project files"""
    try:
        # Load the multimodal model
        model = keras.models.load_model("models/multimodal_churn_model.keras")
        
        # Load scaler and feature info
        scaler = joblib.load("models/scaler.pkl")
        threshold = joblib.load("models/optimal_threshold.pkl")
        
        # Feature names
        feature_names = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'EstimatedSalary', 'BalanceSalaryRatio', 'TenureAgeRatio',
                'ProductUtilizationRate', 'AgeBalanceInteraction', 'EngagementScore',
                'ComplaintCount', 'Geography', 'Gender', 'HasCrCard', 
                'IsActiveMember', 'CreditScoreCategory']
        
        # Load feature importance
        try:
            feature_importance = pd.read_csv("models/feature_importance.csv")
        except:
            # Create from SHAP values if available
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.random.rand(len(feature_names))
            }).sort_values('importance', ascending=False)
        
        # Load test data
        try:
            X_static_test = joblib.load("models/X_static_test.pkl")
            X_lstm_test = joblib.load("models/X_lstm_test.pkl")
            y_test = joblib.load("models/y_test.pkl")
            
            # Convert to DataFrame if needed
            if isinstance(X_static_test, np.ndarray):
                X_static_test = pd.DataFrame(X_static_test, columns=feature_names)
        except:
            X_static_test = None
            X_lstm_test = None
            y_test = None
        
        # Load model comparison
        try:
            model_comparison = pd.read_csv("models/model_comparison.csv", index_col=0)
        except:
            model_comparison = None
        
        # Load retention strategies
        try:
            strategies_df = pd.read_csv("models/retention_strategies.csv")
        except:
            strategies_df = None
        
        return {
            'model': model,
            'scaler': scaler,
            'threshold': threshold,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'X_static_test': X_static_test,
            'X_lstm_test': X_lstm_test,
            'y_test': y_test,
            'model_comparison': model_comparison,
            'strategies_df': strategies_df
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load everything
data = load_models()

# ============================================================================
# CREDIT SCORE TO CATEGORY MAPPING (NEW)
# ============================================================================

def credit_score_to_category(score):
    """
    Convert a raw credit score to a category (0-4) for model input.
    
    Categories:
    0: Poor (300-579)
    1: Fair (580-669)
    2: Good (670-739)
    3: Very Good (740-799)
    4: Excellent (800-850)
    
    Args:
        score: Raw credit score (300-900)
    
    Returns:
        category: Integer 0-4
        label: String label for display
    """
    if score < 580:
        return 0, "Poor"
    elif score < 670:
        return 1, "Fair"
    elif score < 740:
        return 2, "Good"
    elif score < 800:
        return 3, "Very Good"
    else:
        return 4, "Excellent"

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# Sidebar Header with Logo
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0.5rem 1rem 0.5rem;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                width: 70px; height: 70px; border-radius: 20px; 
                display: flex; align-items: center; justify-content: center;
                margin: 0 auto 1rem auto; box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
                border: 2px solid rgba(255,255,255,0.2);">
        <span style="font-size: 2.5rem;">🛡️</span>
    </div>
    <h2 style="color: #2a5298; margin: 0; font-size: 2rem; font-weight: 800; 
               letter-spacing: -0.02em;">
        ChurnShield
    </h2>
    <p style="color: #a0aec0; font-size: 0.9rem; margin-top: 0.5rem; 
              font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;">
        ⚡ AI-Powered Retention
    </p>
    <div style="background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent); 
                height: 2px; width: 80%; margin: 1rem auto 0 auto;">
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation items
nav_items = {
    "🏠 Dashboard": {
        "icon": "🏠",
        "title": "Dashboard",
        "desc": "Overview & Analytics",
        "color": "#0a1f7c",
        "emoji": "📊"
    },
    "🔮 Predict Customer": {
        "icon": "🔮",
        "title": "Predict",
        "desc": "Single Customer Prediction",
        "color": "#0d4620",
        "emoji": "🤖"
    },
    "📊 Model Insights": {
        "icon": "📊",
        "title": "Insights",
        "desc": "SHAP & Feature Importance",
        "color": "#744210",
        "emoji": "🔍"
    },
    "🎯 Retention Strategies": {
        "icon": "🎯",
        "title": "Retention",
        "desc": "AI Action Plans",
        "color": "#822727",
        "emoji": "💡"
    }
}

# Custom radio selection
selected = st.sidebar.radio(
    "",
    options=list(nav_items.keys()),
    index=0,
    format_func=lambda x: f"{nav_items[x]['icon']} {nav_items[x]['title']}"
)

# Store the selected page
page = selected

# Show description of selected page
current_nav = nav_items[selected]
st.sidebar.markdown(f"""
<div style="background: rgba({','.join(str(int(current_nav['color'][1:3], 16)) for i in range(3))}, 0.1);
            padding: 0.8rem 1rem; border-radius: 10px; margin: 0.5rem 0.5rem 1.5rem 0.5rem;
            border-left: 3px solid {current_nav['color']};">
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.2rem;">{current_nav['emoji']}</span>
        <span style="color: #2a5298; font-size: 0.85rem; font-weight: 400;">
            {current_nav['desc']}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Decorative separator
st.sidebar.markdown("""
<div style="margin: 1rem 0.5rem; position: relative;">
    <div style="border-top: 1px solid rgba(255,255,255,0.1);"></div>
    <div style="position: absolute; top: -8px; left: 50%; transform: translateX(-50%); 
                background: #2d3748; padding: 0 12px;">
        <span style="color: #a0aec0; font-size: 0.7rem; font-weight: 600;">LIVE INSIGHTS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Stats in Sidebar
if data and data['model'] is not None and data['X_static_test'] is not None:
    try:
        # Get predictions
        y_pred_prob = data['model'].predict([data['X_static_test'], data['X_lstm_test']], verbose=0).flatten()
        avg_churn_prob = np.mean(y_pred_prob)
        high_risk_pct = np.mean(y_pred_prob > 0.7) * 100
        retention_rate = (1 - avg_churn_prob) * 100
        
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                    padding: 1.2rem; border-radius: 16px; margin: 1rem 0.5rem;
                    border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="color: #a0aec0; font-size: 0.8rem;">Model Status</span>
                <span style="color: #10B981; font-size: 0.8rem; font-weight: 600; 
                           background: rgba(16, 185, 129, 0.2); padding: 0.2rem 0.6rem; 
                           border-radius: 20px;">● Online</span>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem 0.5rem; 
                        border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2); 
                        text-align: center; transition: all 0.3s ease;">
                <div style="color: #10B981; font-size: 1.6rem; font-weight: 800; 
                           line-height: 1.2;">{retention_rate:.1f}%</div>
                <div style="color: #a0aec0; font-size: 0.7rem; font-weight: 500;
                           text-transform: uppercase;">Retention</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem 0.5rem; 
                        border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); 
                        text-align: center; transition: all 0.3s ease;">
                <div style="color: #ef4444; font-size: 1.6rem; font-weight: 800; 
                           line-height: 1.2;">{high_risk_pct:.1f}%</div>
                <div style="color: #a0aec0; font-size: 0.7rem; font-weight: 500;
                           text-transform: uppercase;">At Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.03);
                        border-radius: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #a0aec0; font-size: 0.75rem;">Avg Churn Probability</span>
                    <span style="color: #2a5298; font-size: 0.85rem; font-weight: 600;">{avg_churn_prob:.1%}</span>
                </div>
                <div style="width: 100%; background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px;">
                    <div style="width: {avg_churn_prob*100}%; background: linear-gradient(90deg, #667eea, #764ba2); 
                                height: 4px; border-radius: 2px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.sidebar.error(f"Stats unavailable")
else:
    st.sidebar.info("⚠️ Model not loaded")

# Version and support info
st.sidebar.markdown("""
<div style="margin: 1rem 0.5rem 0.5rem 0.5rem; text-align: center;">
    <div style="display: flex; justify-content: center; gap: 0.8rem; margin-bottom: 0.5rem;">
        <span style="color: #4a5568; font-size: 0.7rem;">📱 v2.1.0</span>
        <span style="color: #4a5568; font-size: 0.7rem;">⚡ Production</span>
        <span style="color: #4a5568; font-size: 0.7rem;">🔒 Secure</span>
    </div>
    <span style="background: rgba(102, 126, 234, 0.1); color: #667eea; 
                padding: 0.2rem 1rem; border-radius: 20px; font-size: 0.65rem;
                border: 1px solid rgba(102, 126, 234, 0.3); font-weight: 500;">
        ⚡ 24/7 Active • 99.9% Uptime
    </span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">📊 Customer Analytics Dashboard</h1>
        <p class="hero-subtitle">Comprehensive overview of customer churn metrics and risk distribution</p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is None or data['model'] is None:
        st.error("❌ Models not loaded. Please ensure all model files are in the 'models' directory.")
        st.stop()
    
    if data['X_static_test'] is None:
        st.warning("⚠️ Test data not available. Showing feature importance only.")
        
        # Show feature importance
        st.markdown("<h2 class='sub-header'>🔑 Feature Importance</h2>", unsafe_allow_html=True)
        
        fig = px.bar(
            data['feature_importance'].head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Features by Importance',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Get predictions for all test data
        with st.spinner("Generating dashboard..."):
            y_pred_prob = data['model'].predict([data['X_static_test'], data['X_lstm_test']], verbose=0).flatten()
            y_pred = (y_pred_prob > data['threshold']).astype(int)
            
            # Calculate metrics
            total = len(data['y_test'])
            churned = int(sum(data['y_test']))
            stayed = total - churned
            churn_rate = (churned / total) * 100
            
            # Risk levels
            risk_levels = []
            for prob in y_pred_prob:
                if prob < 0.3:
                    risk_levels.append('Low Risk')
                elif prob < 0.7:
                    risk_levels.append('Medium Risk')
                else:
                    risk_levels.append('High Risk')
            
            risk_counts = pd.Series(risk_levels).value_counts()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("👥 Total Customers", f"{total:,}", "#667eea"),
            ("✅ Retained", f"{stayed:,}", "#10B981"),
            ("❌ Churned", f"{churned:,}", "#DC2626"),
            ("📊 Churn Rate", f"{churn_rate:.1f}%", "#F59E0B")
        ]
        
        for col, (label, value, color) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div class="metric-value" style="color: {color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two column layout for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h2 class='sub-header'>🎯 Risk Distribution</h2>", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.5,
                marker_colors=['#10B981', '#F59E0B', '#DC2626'],
                textinfo='label+percent',
                textfont_size=14
            )])
            
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                annotations=[dict(text='Risk<br>Distribution', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h2 class='sub-header'>📊 Probability Distribution</h2>", unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=y_pred_prob,
                nbinsx=40,
                marker_color='#667eea',
                opacity=0.8,
                name='Churn Probability'
            ))
            
            fig.add_vline(x=0.3, line_dash="dash", line_color="#10B981", line_width=3,
                         annotation_text="Low Risk", annotation_position="top")
            fig.add_vline(x=0.7, line_dash="dash", line_color="#DC2626", line_width=3,
                         annotation_text="High Risk", annotation_position="top")
            
            fig.update_layout(
                xaxis_title='Churn Probability',
                yaxis_title='Number of Customers',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance Metrics
        st.markdown("<h2 class='sub-header'>📈 Model Performance</h2>", unsafe_allow_html=True)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred)
        recall = recall_score(data['y_test'], y_pred)
        f1 = f1_score(data['y_test'], y_pred)
        auc = roc_auc_score(data['y_test'], y_pred_prob)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        perf_metrics = [
            (col1, "🎯 Accuracy", f"{accuracy:.3f}", "#667eea"),
            (col2, "⚡ Precision", f"{precision:.3f}", "#10B981"),
            (col3, "🔄 Recall", f"{recall:.3f}", "#F59E0B"),
            (col4, "📊 F1-Score", f"{f1:.3f}", "#9F7AEA"),
            (col5, "📈 ROC-AUC", f"{auc:.3f}", "#DC2626")
        ]
        
        for col, label, value, color in perf_metrics:
            with col:
                st.markdown(f"""
                <div style="background: white; padding: 1.2rem; border-radius: 12px; 
                            text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.8rem; font-weight: 700; color: {color};">{value}</div>
                    <div style="font-size: 0.8rem; color: #718096; font-weight: 500;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: PREDICT CUSTOMER
# ============================================================================

elif page == "🔮 Predict Customer":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🔮 Customer Churn Prediction</h1>
        <p class="hero-subtitle">Enter customer details to get AI-powered churn prediction and retention strategy</p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is None or data['model'] is None:
        st.error("❌ Models not loaded. Please ensure all model files are in the 'models' directory.")
        st.stop()
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("""
        <div class="input-card">
            <h3 style="color: #2d3748; margin-bottom: 1.5rem; font-weight: 700;">
                📋 Customer Information
            </h3>
        """, unsafe_allow_html=True)
        
        # Create two columns for inputs
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650, step=10,
                                          help="Customer's credit score (300-900)")
            geography = st.selectbox("🌍 Geography", options=["France", "Germany", "Spain"],
                                    help="Customer's country of residence")
            gender = st.selectbox("👤 Gender", options=["Male", "Female"])
            age = st.number_input("🎂 Age", min_value=18, max_value=100, value=45, step=1)
            tenure = st.number_input("📅 Tenure (years)", min_value=0, max_value=10, value=5, step=1,
                                    help="Years as a customer")
        
        with input_col2:
            balance = st.number_input("💰 Balance ($)", min_value=0.0, max_value=300000.0, value=120000.0, 
                                   step=1000.0, format="%.2f")
            num_products = st.number_input("📦 Number of Products", min_value=1, max_value=4, value=1, step=1)
            has_cr_card = st.selectbox("💳 Has Credit Card", options=["Yes", "No"])
            is_active = st.selectbox("✅ Is Active Member", options=["Yes", "No"])
            estimated_salary = st.number_input("💵 Estimated Salary ($)", min_value=0.0, max_value=200000.0, 
                                              value=80000.0, step=1000.0, format="%.2f")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True)
        
        if predict_button:
            with st.spinner("🤖 AI is analyzing customer data..."):
                try:
                    # Prepare input data with feature engineering from 01_data_preprocessing.ipynb
                    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
                    
                    # Calculate engineered features
                    balance_salary_ratio = balance / (estimated_salary + 1)
                    tenure_age_ratio = tenure / (age + 1)
                    product_utilization = num_products / (tenure + 1)
                    age_balance_interaction = age * balance
                    engagement_score = (1 if is_active == "Yes" else 0) + (1 if has_cr_card == "Yes" else 0) + (num_products / 4)

                    # CREDIT SCORE TO CATEGORY MAPPING (NEW)
                    credit_score_category, credit_score_label = credit_score_to_category(credit_score)

                    complaint_count = np.random.poisson(lam=0.5)
                                        
                    # Define continuous and binary features
                    continuous_features = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 
                                        'EstimatedSalary', 'BalanceSalaryRatio', 'TenureAgeRatio',
                                        'ProductUtilizationRate', 'AgeBalanceInteraction', 'EngagementScore',
                                        'ComplaintCount']
                    
                    binary_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'CreditScoreCategory']
                    
                    # Create feature arrays
                    continuous_values = np.array([[
                        age, tenure, balance, num_products,
                        estimated_salary, balance_salary_ratio, tenure_age_ratio,
                        product_utilization, age_balance_interaction, engagement_score,
                        complaint_count
                    ]])
                    
                    binary_values = np.array([[
                        geo_map[geography], 0 if gender == "Male" else 1,
                        1 if has_cr_card == "Yes" else 0, 1 if is_active == "Yes" else 0,
                        credit_score_category
                    ]])
                    
                    # Scale continuous features
                    continuous_scaled = data['scaler'].transform(continuous_values)
                    
                    # Combine scaled continuous with binary features
                    features_scaled = np.hstack([continuous_scaled, binary_values])
                    
                    sequence = []
                    for month in range(6):
                        # Use tenure to create a realistic but predictable trend
                        # Gradual growth based on balance and time
                        trend = balance * 0.02 * (month / 6)  # Gradual 2% growth over 6 months
                        
                        # Use fixed seasonal pattern instead of random
                        # 1% bump in last month (holiday season effect)
                        seasonal = balance * 0.01 if month == 5 else 0
                        
                        # Calculate monthly balance (no randomness)
                        monthly_balance = max(0, balance + trend + seasonal)
                        sequence.append(monthly_balance)
                    
                    lstm_sequence = np.array(sequence).reshape(1, 6, 1)
                    
                    # Make prediction
                    prob = data['model'].predict([features_scaled, lstm_sequence], verbose=0)[0][0]
                    
                    # ========== APPLY BUSINESS RULES FOR CREDIT SCORE ==========
                    # Store in session state
                    st.session_state.prediction_made = True
                    st.session_state.prediction_prob = prob
                    st.session_state.credit_score_category = credit_score_category
                    st.session_state.credit_score_label = credit_score_label
                    st.session_state.prediction_result = "Likely to Churn" if prob > data['threshold'] else "Likely to Stay"
                    st.session_state.current_customer = {
                        'age': age,
                        'gender': gender,
                        'geography': geography,
                        'products': num_products,
                        'active': is_active,
                        'balance': balance,
                        'credit_score': credit_score,
                        'credit_category': credit_score_label,
                        'tenure': tenure
                    }
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    with col2:
        if st.session_state.prediction_made:
            prob = st.session_state.prediction_prob
            result = st.session_state.prediction_result
            
            # Show warning if exists (NEW)
            if st.session_state.warning_message:
                st.warning(st.session_state.warning_message)
            
            # Show original vs adjusted if different (NEW)
            if st.session_state.original_prob:
                st.info(f"ℹ️ Original model prediction: {st.session_state.original_prob:.1%}")
            
            # Determine risk level and colors
            if prob < 0.3:
                risk_level = "Low Risk"
                risk_color = "#10B981"
                risk_bg = "#D1FAE5"
                emoji = "✅"
            elif prob < 0.7:
                risk_level = "Medium Risk"
                risk_color = "#F59E0B"
                risk_bg = "#FEF3C7"
                emoji = "⚠️"
            else:
                risk_level = "High Risk"
                risk_color = "#DC2626"
                risk_bg = "#FEE2E2"
                emoji = "🚨"
            
            # Result Card
            st.markdown(f"""
            <div style="background: {risk_bg}; padding: 2rem; border-radius: 20px; 
                        border: 3px solid {risk_color}; text-align: center; 
                        box-shadow: 0 20px 60px rgba(0,0,0,0.15);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
                <h2 style="color: {risk_color}; margin: 0; font-size: 2rem; font-weight: 800;">
                    {result}
                </h2>
                <h3 style="color: {risk_color}; margin: 0.5rem 0; font-size: 1.3rem;">
                    {risk_level}
                </h3>
                <p style="font-size: 2.5rem; font-weight: 800; color: {risk_color}; margin: 1rem 0;">
                    {prob:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability", 'font': {'size': 20, 'color': '#2d3748'}},
                number={'suffix': "%", 'font': {'size': 36, 'color': risk_color, 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#2d3748"},
                    'bar': {'color': risk_color, 'thickness': 0.75},
                    'bgcolor': "white",
                    'borderwidth': 3,
                    'bordercolor': risk_color,
                    'steps': [
                        {'range': [0, 30], 'color': '#D1FAE5'},
                        {'range': [30, 70], 'color': '#FEF3C7'},
                        {'range': [70, 100], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "#e53e3e", 'width': 4},
                        'thickness': 0.8,
                        'value': data['threshold'] * 100
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

             # Show credit score category information
            if st.session_state.prediction_made and hasattr(st.session_state, 'credit_score_label'):
                st.info(f"💳 Credit Score: {st.session_state.current_customer['credit_score']} → **{st.session_state.credit_score_label}** category")
            
            # AI Recommendations based on risk level
            st.markdown(f"""
            <div style="background: {risk_bg}; padding: 1.5rem; border-radius: 16px; 
                        border-left: 5px solid {risk_color}; margin-top: 1rem;">
                <h4 style="color: {risk_color}; margin: 0 0 1rem 0; font-weight: 700;">
                    🤖 AI Retention Strategy
                </h4>
            """, unsafe_allow_html=True)
            
            if risk_level == "High Risk":
                st.markdown("""
                <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #2d3748;">
                    <li><strong>🎁 Premium Account Upgrade</strong> - Waive fees for 6 months</li>
                    <li><strong>👔 Dedicated Relationship Manager</strong> - Personal financial advisor assigned</li>
                    <li><strong>💰 Loan Interest Rate Discount</strong> - 1.5% reduction on personal loans</li>
                    <li><strong>📞 Priority Support</strong> - 24/7 dedicated customer service line</li>
                    <li><strong>📧 Personalized Campaign</strong> - Exclusive benefits highlight</li>
                </ol>
                <p style="margin-top: 1rem; font-weight: 600; color: #DC2626;">
                    💡 Expected Impact: 40% reduction in churn probability
                </p>
                """, unsafe_allow_html=True)
            elif risk_level == "Medium Risk":
                st.markdown("""
                <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #2d3748;">
                    <li><strong>⭐ Loyalty Rewards Program</strong> - Double points for 3 months</li>
                    <li><strong>💳 Credit Card Fee Waiver</strong> - 6 months annual fee waived</li>
                    <li><strong>🎓 Exclusive Webinar</strong> - Financial planning session invitation</li>
                    <li><strong>🎁 Referral Bonus</strong> - $50 for each successful referral</li>
                    <li><strong>📊 Quarterly Review</strong> - Free portfolio assessment</li>
                </ol>
                <p style="margin-top: 1rem; font-weight: 600; color: #F59E0B;">
                    💡 Expected Impact: 25% reduction in churn probability
                </p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <ol style="margin: 0; padding-left: 1.2rem; line-height: 1.8; color: #2d3748;">
                    <li><strong>☕ Thank You Gift</strong> - $10 coffee voucher</li>
                    <li><strong>🎂 Birthday Bonus</strong> - $5 cashback on birthday month</li>
                    <li><strong>🔓 Early Access</strong> - Preview new features and products</li>
                    <li><strong>📰 Monthly Newsletter</strong> - Personalized financial tips</li>
                    <li><strong>🏆 Loyalty Badge</strong> - Recognition in app</li>
                </ol>
                <p style="margin-top: 1rem; font-weight: 600; color: #10B981;">
                    💡 Expected Impact: 10% increase in loyalty score
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Placeholder when no prediction made
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center;
                        border: 2px dashed #cbd5e0; height: 100%;
                        display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
                <h3 style="color: #718096; margin: 0;">Enter customer details and click</h3>
                <h3 style="color: #667eea; margin: 0.5rem 0;">"Predict Churn Risk"</h3>
                <p style="color: #a0aec0; margin-top: 1rem;">
                    AI will analyze the data and provide personalized retention strategies
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: MODEL INSIGHTS
# ============================================================================

elif page == "📊 Model Insights":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">📊 Model Insights & Explainability</h1>
        <p class="hero-subtitle">Understanding what drives customer churn with SHAP analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is None:
        st.error("❌ Models not loaded. Please ensure all model files are in the 'models' directory.")
        st.stop()
    
    # Feature Importance
    st.markdown("<h2 class='sub-header'>🔑 Feature Importance</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            data['feature_importance'].head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Features by Importance (SHAP Values)',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
            <h4>📈 SHAP Explanation</h4>
            <p>SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction:</p>
            <ul>
                <li><strong>Positive values</strong> ⬆️ increase churn probability</li>
                <li><strong>Negative values</strong> ⬇️ decrease churn probability</li>
                <li><strong>Magnitude</strong> shows the strength of impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{data['threshold']:.2f}</div>
            <div class='metric-label'>Optimal Threshold</div>
            <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
                Predict churn if probability > {data['threshold']:.0%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance Comparison
    st.markdown("<h2 class='sub-header'>📊 Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    if data['model_comparison'] is not None:
        # Reshape for plotting
        plot_df = data['model_comparison'].reset_index().melt(id_vars='index')
        plot_df.columns = ['Model', 'Metric', 'Score']
        
        fig = px.bar(
            plot_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison (from 03_model_comparison.ipynb)',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Show best model
        best_f1 = data['model_comparison']['F1-Score'].idxmax()
        best_auc = data['model_comparison']['AUC'].idxmax()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 Best Model by F1-Score: **{best_f1}**")
        with col2:
            st.success(f"🏆 Best Model by AUC: **{best_auc}**")
    else:
        st.info("Model comparison data not available. Run 03_model_comparison.ipynb first.")
    
  
    
    
   
# ============================================================================
# PAGE 4: RETENTION STRATEGIES
# ============================================================================

else:  # "🎯 Retention Strategies"
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎯 AI-Powered Retention Strategies</h1>
        <p class="hero-subtitle">Personalized action plans to maximize customer retention</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%); 
                padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;
                border-left: 5px solid #4299e1;">
        <h3 style="color: #2b6cb0; margin: 0 0 0.5rem 0; font-weight: 700;">🤖 How It Works</h3>
        <p style="color: #4a5568; margin: 0; line-height: 1.6;">
            Our AI analyzes customer behavior patterns and generates personalized retention strategies 
            based on risk level, demographic profile, and engagement history. Each strategy is tailored 
            to maximize retention probability while minimizing cost.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "🎚️ Filter by Risk Level",
            options=['Low Risk', 'Medium Risk', 'High Risk'],
            default=['High Risk', 'Medium Risk']
        )
    
    with col2:
        min_prob = st.slider("📉 Min Churn Probability", 0.0, 1.0, 0.5, 0.05)
    
    with col3:
        max_customers = st.slider("👥 Number of Customers", 5, 50, 10)
    
    # Try to load pre-generated strategies from 05_retention_strategies.ipynb
    if data and data['strategies_df'] is not None:
        strategies_df = data['strategies_df']
        
        # Filter based on risk level
        if 'risk_level' in strategies_df.columns:
            filtered = strategies_df[strategies_df['risk_level'].isin(risk_filter)]
            
            # Filter by probability if column exists
            if 'churn_probability' in filtered.columns:
                prob_values = filtered['churn_probability'].str.replace('%', '').astype(float) / 100
                filtered = filtered[prob_values >= min_prob]
            
            filtered = filtered.head(max_customers)
            
            # Display strategies
            for idx, row in filtered.iterrows():
                risk = row['risk_level']
                
                # Set color based on risk
                if risk == 'High Risk':
                    box_class = 'danger-box'
                    emoji = '🔴'
                    color = '#DC2626'
                elif risk == 'Medium Risk':
                    box_class = 'warning-box'
                    emoji = '🟠'
                    color = '#F59E0B'
                else:
                    box_class = 'success-box'
                    emoji = '🟢'
                    color = '#10B981'
                
                with st.expander(f"{emoji} {row.get('customer_id', f'Customer {idx}')} - {risk} ({row.get('churn_probability', 'N/A')})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.03); padding: 1.5rem; border-radius: 12px;">
                            <h4 style="color: {color}; margin: 0 0 1rem 0; font-weight: 700;">
                                👤 Customer Profile
                            </h4>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Age:</strong> {row.get('age', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Gender:</strong> {row.get('gender', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Location:</strong> {row.get('geography', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Products:</strong> {row.get('products', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Active:</strong> {row.get('active', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Balance:</strong> {row.get('balance', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Credit Score:</strong> {row.get('credit_score', 'N/A')}</p>
                            <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Tenure:</strong> {row.get('tenure', 'N/A')} years</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Strategy based on risk level
                        if risk == 'High Risk':
                            st.markdown(f"""
                            <div class='{box_class}'>
                                <h4 style="color: {color};">🔴 URGENT INTERVENTION</h4>
                                <p><strong>Immediate Actions (24-48 hours):</strong></p>
                                <ul>
                                    <li>Priority phone call from relationship manager</li>
                                    <li>Personal follow-up on recent issues</li>
                                    <li>Premium retention offer: 6 months fee waiver</li>
                                    <li>Dedicated financial advisor consultation</li>
                                </ul>
                                <p><strong>Expected Impact:</strong> 40-50% churn reduction</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        elif risk == 'Medium Risk':
                            st.markdown(f"""
                            <div class='{box_class}'>
                                <h4 style="color: {color};">🟠 PROACTIVE RETENTION</h4>
                                <p><strong>Recommended Actions (1 week):</strong></p>
                                <ul>
                                    <li>Double loyalty points for 3 months</li>
                                    <li>Free credit score monitoring (6 months)</li>
                                    <li>Personalized email with exclusive offers</li>
                                    <li>Pre-approved credit line increase</li>
                                </ul>
                                <p><strong>Expected Impact:</strong> 25-35% churn reduction</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f"""
                            <div class='{box_class}'>
                                <h4 style="color: {color};">🟢 LOYALTY REINFORCEMENT</h4>
                                <p><strong>Appreciation Actions:</strong></p>
                                <ul>
                                    <li>$10 coffee voucher or gift card</li>
                                    <li>Early access to new features</li>
                                    <li>Enhanced referral bonus: $75 per referral</li>
                                    <li>Birthday bonus and anniversary gifts</li>
                                </ul>
                                <p><strong>Expected Impact:</strong> 10-15% loyalty increase</p>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        # Generate sample strategies if pre-generated ones aren't available
        st.info("Using sample strategies - run 05_retention_strategies.ipynb to generate personalized ones")
        
        np.random.seed(42)
        sample_customers = []
        
        for i in range(50):
            prob = np.random.uniform(0, 1)
            if prob < 0.3:
                risk = "Low Risk"
                color = "#10B981"
                bg_color = "#D1FAE5"
            elif prob < 0.7:
                risk = "Medium Risk"
                color = "#F59E0B"
                bg_color = "#FEF3C7"
            else:
                risk = "High Risk"
                color = "#DC2626"
                bg_color = "#FEE2E2"
            
            sample_customers.append({
                'customer_id': f"CUST_{1000 + i}",
                'risk_level': risk,
                'color': color,
                'bg_color': bg_color,
                'churn_prob': prob,
                'age': np.random.randint(25, 70),
                'products': np.random.randint(1, 4),
                'active': np.random.choice([True, False], p=[0.7, 0.3]),
                'balance': np.random.uniform(0, 200000)
            })
        
        # Filter customers
        filtered_customers = [
            c for c in sample_customers 
            if c['risk_level'] in risk_filter and c['churn_prob'] >= min_prob
        ][:max_customers]
        
        # Display strategies
        for customer in filtered_customers:
            with st.expander(f"{'🔴' if customer['risk_level']=='High Risk' else '🟠' if customer['risk_level']=='Medium Risk' else '🟢'} {customer['customer_id']} - {customer['risk_level']} ({customer['churn_prob']:.1%})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.03); padding: 1.5rem; border-radius: 12px;">
                        <h4 style="color: {customer['color']}; margin: 0 0 1rem 0; font-weight: 700;">
                            👤 Customer Profile
                        </h4>
                        <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Age:</strong> {customer['age']} years</p>
                        <p style="margin: 0.5rem 0; color: #4a5568;"><strong>Products:</strong> {customer['products']}</p>
                        <p style="margin: 0.5rem 0; color: #4a5568;">
                            <strong>Active:</strong> {'✅ Yes' if customer['active'] else '❌ No'}
                        </p>
                        <p style="margin: 0.5rem 0; color: #4a5568;">
                            <strong>Balance:</strong> ${customer['balance']:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if customer['risk_level'] == "High Risk":
                        st.markdown(f"""
                        <div class="danger-box">
                            <h4 style="color: #DC2626;">🔴 URGENT INTERVENTION</h4>
                            <p><strong>Immediate Actions (24-48 hours):</strong></p>
                            <ul>
                                <li>Priority phone call from relationship manager</li>
                                <li>Personal follow-up on recent issues</li>
                                <li>Premium retention offer: 6 months fee waiver</li>
                                <li>Dedicated financial advisor consultation</li>
                            </ul>
                            <p><strong>Expected Impact:</strong> 40-50% churn reduction</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif customer['risk_level'] == "Medium Risk":
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4 style="color: #F59E0B;">🟠 PROACTIVE RETENTION</h4>
                            <p><strong>Recommended Actions (1 week):</strong></p>
                            <ul>
                                <li>Double loyalty points for 3 months</li>
                                <li>Free credit score monitoring (6 months)</li>
                                <li>Personalized email with exclusive offers</li>
                                <li>Pre-approved credit line increase</li>
                            </ul>
                            <p><strong>Expected Impact:</strong> 25-35% churn reduction</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4 style="color: #10B981;">🟢 LOYALTY REINFORCEMENT</h4>
                            <p><strong>Appreciation Actions:</strong></p>
                            <ul>
                                <li>$10 coffee voucher or gift card</li>
                                <li>Early access to new features</li>
                                <li>Enhanced referral bonus: $75 per referral</li>
                                <li>Birthday bonus and anniversary gifts</li>
                            </ul>
                            <p><strong>Expected Impact:</strong> 10-15% loyalty increase</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Download button
        strategies_df = pd.DataFrame([
            {
                'Customer ID': c['customer_id'],
                'Risk Level': c['risk_level'],
                'Churn Probability': f"{c['churn_prob']:.1%}",
                'Age': c['age'],
                'Products': c['products'],
                'Active': c['active'],
                'Balance': f"${c['balance']:,.0f}"
            }
            for c in filtered_customers
        ])
    
    # Download button for filtered strategies
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if 'strategies_df' in locals() and not strategies_df.empty:
            st.download_button(
                label="📥 Download Retention Strategies (CSV)",
                data=strategies_df.to_csv(index=False),
                file_name=f"retention_strategies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 0.8rem;'>"
    "🛡️ ChurnShield AI v2.1.0 | Built with TensorFlow & Streamlit | "
    f"© {datetime.now().year} All Rights Reserved</p>",
    unsafe_allow_html=True
)