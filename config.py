CONFIG = {
    "max_comments_per_video":5, #1000,  # Maximum comments to fetch
    "fetch_interval": 300,  # Fetch new comments every 300 seconds (5 minutes)
    "sentiment_threshold": 0.5,  # Threshold for classifying sentiment
    "batch_size": 100,  # Process comments in batches of 100
    "languages": ["en"],  # Supported languages
    "update_mode": "append",  # Append new data or replace
}

FILE_PATHS = {
    "raw_comments": "data/raw_comments.csv",
    "processed_sentiment": "data/processed_sentiment.csv",
    "processed_data": "data/processed_data.csv", 
    "model_path": "models/sentiment_model.pkl",
    "logs": "logs/",
}

SENTIMENT_LABELS = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

DASHBOARD_CONFIG = {
    "page_title": "YouTube Sentiment Analyzer",
    "page_icon": "📊",
    "layout": "wide",
    "update_interval": 5,  # Dashboard refresh interval in seconds
}