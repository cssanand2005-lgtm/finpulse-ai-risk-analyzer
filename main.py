import pandas as pd
import polars as pl
from scraper import get_stock_data, get_financial_news
from model import initialize_sentiment_analyzer, analyze_headlines_sentiment

def run_pipeline(ticker_symbol, company_keyword):
    print(f"🚀 Starting FinPulse Pipeline for: {ticker_symbol} ({company_keyword})")
    print("="*60)
    
    # 1. Fetch raw stock data
    df_stock_raw = get_stock_data(ticker_symbol, period="1mo", interval="1d")
    
    # 2. Fetch raw news headlines
    df_news_raw = get_financial_news(company_keyword)
    
    # 3. Initialize our downloaded Hugging Face model
    ai_model = initialize_sentiment_analyzer()
    
    # 4. Run text data through the AI model
    df_news_analyzed = analyze_headlines_sentiment(df_news_raw, ai_model)
    
    # 5. Bring in High-Performance Data Engineering (Polars processing)
    # Convert our analytical data into Polars dataframes
    pl_stock = pl.from_pandas(df_stock_raw)
    pl_news = pl.from_pandas(df_news_analyzed)
    
    print("\n" + "="*60)
    print("📊 DATA ENGINEERING METRICS (POLARS)")
    print("="*60)
    print(f"🔹 Total Days of Market Data Processed: {pl_stock.height}")
    print(f"🔹 Total News Headlines Evaluated by AI: {pl_news.height}")
    
    # Calculate an overall market risk signal score from the headlines
    if not pl_news.is_empty():
        avg_sentiment = pl_news["ai_risk_score"].mean()
        print(f"🔹 Average Aggregate AI Sentiment Score: {avg_sentiment:.4f}")
        
        if avg_sentiment > 0.15:
            print("🟢 MARKET SIGNAL: BULLISH / STABLE SENTIMENT")
        elif avg_sentiment < -0.15:
            print("🔴 MARKET SIGNAL: BEARISH / HIGH ACCUMULATED RISK")
        else:
            print("🟡 MARKET SIGNAL: NEUTRAL / MIXED SENTIMENT")
            
    return pl_stock, pl_news

if __name__ == "__main__":
    # Let's target a major stock to watch live data stream in
    # 'RELIANCE.NS' is the ticker for Reliance Industries on NSE
    ticker = "RELIANCE.NS"
    keyword = "Reliance"
    
    stock_data, news_data = run_pipeline(ticker, keyword)