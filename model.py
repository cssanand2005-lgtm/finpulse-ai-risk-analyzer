import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import pipeline
import pandas as pd
import polars as pl

def initialize_sentiment_analyzer():
    """
    Loads the FinBERT model from Hugging Face.
    FinBERT is a BERT model specifically fine-tuned for financial text sentiment.
    """
    print("🤖 Loading FinBERT AI Model from Hugging Face (this may take a minute on the first run)...")
    # Using the standard industry benchmark for financial sentiment analysis
    analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return analyzer

def analyze_headlines_sentiment(df_news, analyzer):
    """
    Takes a DataFrame of news headlines and scores them using the AI analyzer.
    """
    if df_news.empty:
        print("⚠️ No news headlines provided for analysis.")
        return pd.DataFrame()

    print("🧠 Analyzing text sentiment using AI pipeline...")
    headlines = df_news['title'].tolist()
    
    # Run inference through the transformer model
    predictions = analyzer(headlines)
    
    # Extract labels and numerical confidence scores
    labels = [pred['label'] for pred in predictions]
    scores = [pred['score'] for pred in predictions]
    
    # Map raw labels to a consistent continuous numerical metric (-1 to 1) for data analysis
    # positive -> 1, negative -> -1, neutral -> 0
    numerical_sentiment = []
    for pred in predictions:
        if pred['label'] == 'positive':
            numerical_sentiment.append(pred['score'])
        elif pred['label'] == 'negative':
            numerical_sentiment.append(-pred['score'])
        else:
            numerical_sentiment.append(0.0)
            
    # Add the AI outputs back into our DataFrame
    df_news['sentiment_label'] = labels
    df_news['confidence_score'] = scores
    df_news['ai_risk_score'] = numerical_sentiment
    
    return df_news

if __name__ == "__main__":
    # Let's test the AI engine with some dummy financial phrases to see if it works!
    test_data = pd.DataFrame({
        "title": [
            "Ambuja Cement profits jump by 25% exceeding market expectations",
            "Regulatory hurdle delays infrastructure project expansion",
            "Company maintains steady operations ahead of quarterly review"
        ],
        "pub_date": ["Today", "Today", "Today"]
    })
    
    # 1. Start the model
    ai_model = initialize_sentiment_analyzer()
    
    # 2. Score the dummy phrases
    df_analyzed = analyze_headlines_sentiment(test_data, ai_model)
    
    print("\n--- AI Sentiment Analysis Results ---")
    print(df_analyzed[['title', 'sentiment_label', 'ai_risk_score']])