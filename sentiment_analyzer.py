"""
sentiment_analyzer.py - Perform sentiment analysis on YouTube comments
Uses VADER (best for social media) with TextBlob as alternative
"""

import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import re
import logging
from config import SENTIMENT_LABELS

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, method='vader'):
        """
        Initialize sentiment analyzer
        
        Parameters:
        - method: 'vader' (recommended for social media) or 'textblob'
        """
        self.method = method
        
        if method == 'vader':
            self.sia = SentimentIntensityAnalyzer()
        elif method == 'textblob':
            pass  # TextBlob doesn't require initialization
        else:
            raise ValueError("Method must be 'vader' or 'textblob'")
    
    def clean_text(self, text):
        """Clean text before sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep emojis and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?\'".,😊😢😂❤️]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_vader(self, text):
        """
        Perform sentiment analysis using VADER
        Returns: compound score (-1 to 1), pos, neu, neg scores
        """
        cleaned_text = self.clean_text(text)
        scores = self.sia.polarity_scores(cleaned_text)
        
        # Extract compound score (overall sentiment)
        compound = scores['compound']
        
        # Classify sentiment
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'compound': compound,
            'positive_score': scores['pos'],
            'neutral_score': scores['neu'],
            'negative_score': scores['neg'],
        }
    
    def analyze_textblob(self, text):
        """
        Perform sentiment analysis using TextBlob
        Polarity: -1 (negative) to 1 (positive)
        Subjectivity: 0 (objective) to 1 (subjective)
        """
        cleaned_text = self.clean_text(text)
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'compound': polarity,  # For compatibility
        }
    
    def analyze(self, text):
        """Analyze sentiment of given text"""
        if self.method == 'vader':
            return self.analyze_vader(text)
        elif self.method == 'textblob':
            return self.analyze_textblob(text)
    
    def analyze_batch(self, texts):
        """Analyze sentiment for multiple texts"""
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing text {i}: {e}")
                results.append({
                    'sentiment': 'neutral',
                    'compound': 0,
                    'positive_score': 0,
                    'neutral_score': 1,
                    'negative_score': 0,
                })
        
        return results
    
    def analyze_dataframe(self, df, text_column='text'):
        """
        Add sentiment analysis to a DataFrame
        
        Parameters:
        - df: pandas DataFrame
        - text_column: name of column containing text to analyze
        
        Returns:
        - DataFrame with sentiment columns added
        """
        logger.info(f"Analyzing {len(df)} comments...")
        
        # Analyze all texts
        results = self.analyze_batch(df[text_column].values)
        
        # Extract results into separate columns
        sentiments = [r['sentiment'] for r in results]
        compounds = [r['compound'] for r in results]
        
        if self.method == 'vader':
            pos_scores = [r['positive_score'] for r in results]
            neu_scores = [r['neutral_score'] for r in results]
            neg_scores = [r['negative_score'] for r in results]
            
            df['sentiment'] = sentiments
            df['compound_score'] = compounds
            df['positive_score'] = pos_scores
            df['neutral_score'] = neu_scores
            df['negative_score'] = neg_scores
        else:
            subjectivities = [r['subjectivity'] for r in results]
            
            df['sentiment'] = sentiments
            df['polarity'] = compounds
            df['subjectivity'] = subjectivities
        
        logger.info("Sentiment analysis completed")
        return df
    
    def get_sentiment_stats(self, df):
        """Get statistics about sentiments in DataFrame"""
        if 'sentiment' not in df.columns:
            return None
        
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
        
        stats = {
            'total_comments': len(df),
            'positive_count': sentiment_counts.get('positive', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'positive_percentage': sentiment_percentages.get('positive', 0),
            'neutral_percentage': sentiment_percentages.get('neutral', 0),
            'negative_percentage': sentiment_percentages.get('negative', 0),
            'average_compound': df['compound_score'].mean() if 'compound_score' in df.columns else None,
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(method='vader')
    
    # Test texts
    test_texts = [
        "This video is absolutely amazing! I loved it!",
        "Terrible content, waste of time",
        "It's okay, nothing special",
        "Best video I've seen all year!",
        "Worst thing ever made",
    ]
    
    print("Testing VADER Sentiment Analysis:")
    print("-" * 50)
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Compound Score: {result['compound']:.3f}")