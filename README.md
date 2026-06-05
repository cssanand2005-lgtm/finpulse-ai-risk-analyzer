# FinPulse: Real-Time Financial Sentiment & Stock Risk Analyzer

An end-to-end AI and Data Science pipeline that tracks real-time market sentiment using fine-tuned NLP transformers and evaluates volatility correlations using high-performance data frameworks.

## 🌟 Key Features
- **Parallel Data Engineering**: Implemented utilizing the **Polars** framework for high-performance tabular data optimization.
- **Deep Learning Text Analytics**: Leverages the **FinBERT Transformer** model via **PyTorch** to extract precision directional semantic metrics from financial news headlines.
- **Live Web Scraping**: Dynamically parses real-time Google News RSS feeds and historical financial data utilizing **BeautifulSoup4** and **YFinance**.
- **Interactive UI Dashboard**: Features a dark-themed **Streamlit** frontend interface incorporating custom **Plotly** data visualizations.

## 🛠️ Project Architecture
1. `scraper.py` - Fetches raw market closing prices and live industry news feeds.
2. `model.py` - Initialized the transformer pipeline to score text segments numerically (-1 to +1).
3. `main.py` - Orchestrates the flow and implements data structuring in Polars DataFrames.
4. `app.py` - Serves the final interactive user interface web app.

## 🚀 How to Run Locally
1. Clone this repository.
2. Ensure you have your environment activated and run:
   ```bash
   pip install -r requirements.txt
