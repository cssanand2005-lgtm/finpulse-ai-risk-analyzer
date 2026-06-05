import streamlit as st
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from main import run_pipeline

# 1. Page Configuration
st.set_page_config(page_title="FinPulse AI Risk Analyzer", layout="wide", page_icon="📊")

st.title("📊 FinPulse: Real-Time Financial Sentiment & Stock Risk Analyzer")
st.markdown("An end-to-end AI pipeline tracking market sentiment using FinBERT and evaluating volatility correlations.")
st.markdown("---")

# 2. Sidebar Configuration for User Input
st.sidebar.header("🎯 Target Selection")
ticker_input = st.sidebar.text_input("Enter NSE Ticker Symbol:", value="RELIANCE.NS")
keyword_input = st.sidebar.text_input("Enter Company Search Keyword:", value="Reliance")

analyze_button = st.sidebar.button("⚡ Run Live AI Analysis")

# 3. Handle App Trigger
if analyze_button:
    with st.spinner("Processing live pipeline (Scraping web data & running deep learning inference)..."):
        try:
            # Call your main background pipeline!
            stock_data, news_data = run_pipeline(ticker_input, keyword_input)
            
            # --- ROW 1: METRICS & SIGNALS ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Market Data Rows", value=f"{stock_data.height} Days")
            with col2:
                st.metric(label="Articles Evaluated", value=f"{news_data.height} Headlines")
                
            with col3:
                if not news_data.is_empty():
                    avg_sentiment = news_data["ai_risk_score"].mean()
                    if avg_sentiment > 0.15:
                        st.success(f"🟢 BULLISH (Score: {avg_sentiment:.2f})")
                    elif avg_sentiment < -0.15:
                        st.error(f"🔴 HIGH RISK (Score: {avg_sentiment:.2f})")
                    else:
                        st.warning(f"🟡 NEUTRAL (Score: {avg_sentiment:.2f})")
                        
            st.markdown("---")
            
            # --- ROW 2: INTERACTIVE CHART ---
            st.subheader("📈 Historical Market Closing Trend")
            # Convert Polars back to Pandas just for Plotly mapping
            pd_stock = stock_data.to_pandas()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd_stock['Date'], y=pd_stock['Close'], mode='lines', name='Close Price', line=dict(color='#00CC96', width=2)))
            fig.update_layout(template='plotly_dark', xaxis_title='Date', yaxis_title='Price (INR)', margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # --- ROW 3: LIVE SENTIMENT BREAKDOWN ---
            st.subheader("📰 Live AI Sentiment Feed")
            pd_news = news_data.to_pandas()
            # Render a clean dataframe directly onto the webpage
            st.dataframe(pd_news[['title', 'sentiment_label', 'confidence_score']], use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")
else:
    st.info("👈 Enter a stock ticker on the sidebar and click 'Run Live AI Analysis' to launch the web dashboard!")