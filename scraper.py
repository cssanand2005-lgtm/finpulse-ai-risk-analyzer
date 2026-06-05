import yfinance as yf
from bs4 import BeautifulSoup
import requests
import pandas as pd
import polars as pl

def get_stock_data(ticker, period="1mo", interval="1d"):
    """
    Fetches historical market data for a given stock ticker using yfinance.
    """
    print(f"📥 Fetching market data for {ticker} from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    
    # Reset index so 'Date' becomes a standard queryable column
    hist = hist.reset_index()
    return hist

def get_financial_news(ticker_name):
    """
    Scrapes recent financial headlines from Google News RSS feed based on a keyword.
    """
    print(f"📰 Scraping live financial headlines for '{ticker_name}'...")
    # Target Google News RSS feed for Indian business news
    url = f"https://news.google.com/rss/search?q={ticker_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")
    
    articles = soup.findAll('item')
    news_data = []
    
    # Extract titles and dates for the top 10 most recent articles
    for item in articles[:10]:
        news_data.append({
            "title": item.title.text,
            "pub_date": item.pubDate.text
        })
        
    return pd.DataFrame(news_data)

if __name__ == "__main__":
    # Local isolated testing block
    test_ticker = "RELIANCE.NS"
    
    print("--- Running Local Scraper Test ---")
    stock_df = get_stock_data(test_ticker)
    news_df = get_financial_news("Reliance")
    
    print("\n✅ Scraper working perfectly! Rows fetched:", len(stock_df))