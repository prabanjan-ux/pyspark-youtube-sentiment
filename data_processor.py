"""
data_processor.py - Process and prepare data for analysis using PySpark
Handles cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
import logging
import os
from config import FILE_PATHS

# --- PySpark Imports ---
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, IntegerType, FloatType, TimestampType, StructType, StructField
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logging.warning("PySpark not found. SparkDataProcessor will not be available.")

# --- NLTK Import ---
from nltk.sentiment import SentimentIntensityAnalyzer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkDataProcessor:
    """
    Process and analyze data using a PySpark DataFrame pipeline.
    """

    def __init__(self, spark: 'SparkSession'):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required for SparkDataProcessor.")
        self.spark = spark
        self._register_udfs()
        logger.info("SparkDataProcessor initialized and UDFs registered.")

    def _register_udfs(self):
        """Register User-Defined Functions (UDFs) for sentiment and language."""

        # --- VADER Sentiment UDF ---
        # We assume NLTK data is downloaded by main.py
        try:
            sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
            logger.error("Please ensure 'vader_lexicon' is downloaded (run main.py).")
            raise e

        def get_vader_compound(text):
            if text is None:
                return 0.0
            return sia.polarity_scores(text)['compound']

        self.spark.udf.register("get_compound_score", get_vader_compound, FloatType())

        # --- Language Detection UDF ---
        def detect_language(text):
            if text is None:
                return None
            try:
                from langdetect import detect
                return detect(text)
            except:
                return None # Error during detection

        self.spark.udf.register("detect_language", detect_language, StringType())

    def load_data(self, file_path):
        """Load data from CSV into a Spark DataFrame."""
        try:
            # Added multiline=True for better CSV parsing
            df = self.spark.read.csv(file_path, header=True, inferSchema=True, multiLine=True, escape='"')
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            schema = StructType([
                StructField("comment_id", StringType(), True),
                StructField("author", StringType(), True),
                StructField("text", StringType(), True),
                StructField("like_count", IntegerType(), True),
                StructField("published_at", StringType(), True),
            ])
            df = self.spark.createDataFrame([], schema)
            return df

        # Normalize column names
        rename_map = {
            'publishedAt': 'timestamp',
            'published_at': 'timestamp',
            'like_count': 'likes'
        }

        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)

        # Ensure base columns exist
        if 'likes' not in df.columns:
            df = df.withColumn('likes', F.lit(0).cast(IntegerType()))
        if 'reply_count' not in df.columns:
             df = df.withColumn('reply_count', F.lit(0).cast(IntegerType()))

        return df

    def analyze_sentiment(self, df, text_col='text'):
        """Apply VADER sentiment analysis using Spark UDF."""
        logger.info("Analyzing sentiment with Spark VADER UDF...")
        df = df.withColumn('compound_score', F.call_udf("get_compound_score", F.col(text_col)))

        df = df.withColumn('sentiment',
                           F.when(F.col('compound_score') >= 0.05, 'positive')
                           .when(F.col('compound_score') <= -0.05, 'negative')
                           .otherwise('neutral'))
        return df

    def remove_duplicates(self, df):
        """Remove duplicate comments based on text and timestamp."""
        time_col = 'timestamp'
        original_count = df.count()
        df = df.dropDuplicates(subset=['text', time_col])
        removed = original_count - df.count()
        logger.info(f"Removed {removed} duplicate comments")
        return df

    def filter_by_language(self, df, language='en'):
        """Filter comments by language using Spark UDF."""
        try:
            df_with_lang = df.withColumn("language", F.call_udf("detect_language", F.col('text')))
            df_filtered = df_with_lang.filter(F.col('language') == language)
            count = df_filtered.count()
            logger.info(f"Filtered to {count} {language} comments")
            return df_filtered
        except Exception as e:
            logger.warning(f"langdetect UDF failed: {e}. Skipping language filtering.")
            return df

    def remove_spam_comments(self, df, min_length=3):
        """Remove very short or spam-like comments."""
        original_count = df.count()
        df = df.filter(F.length(F.col('text')) >= min_length)
        removed = original_count - df.count()
        logger.info(f"Removed {removed} spam/short comments")
        return df

    def add_timestamp_features(self, df, timestamp_col='timestamp'):
        """Add time-based features from timestamp."""
        # Ensure correct timestamp type
        df = df.withColumn(timestamp_col, F.to_timestamp(F.col(timestamp_col)))

        df = df.withColumn('hour', F.hour(F.col(timestamp_col)))
        df = df.withColumn('day_of_week', F.date_format(F.col(timestamp_col), 'E')) # 'E' = Mon, Tue, etc.
        df = df.withColumn('date', F.to_date(F.col(timestamp_col)))

        return df

    def add_text_features(self, df, text_col='text'):
        """Add text-based features using Spark SQL functions."""
        df = df.na.fill("", [text_col]) # Fill null text to avoid errors

        df = df.withColumn('text_length', F.length(F.col(text_col)))
        df = df.withColumn('word_count', F.size(F.split(F.col(text_col), ' ')))

        df = df.withColumn('has_caps',
                           F.when(F.col(text_col) == F.upper(F.col(text_col)), 1)
                           .otherwise(0))

        df = df.withColumn('has_exclamation',
                           F.when(F.col(text_col).contains('!'), 1)
                           .otherwise(0))

        df = df.withColumn('has_question',
                           F.when(F.col(text_col).rlike(r'\?'), 1) # rlike for regex
                           .otherwise(0))

        df = df.withColumn('has_emoji',
                           F.when(F.col(text_col).rlike(r'[😊😢😂❤️👍👎]'), 1)
                           .otherwise(0))
        return df

    def add_engagement_features(self, df):
        """Add engagement-based features."""
        # Ensure columns are numeric and fill nulls
        df = df.withColumn('likes', F.col('likes').cast(IntegerType()))
        df = df.withColumn('reply_count', F.col('reply_count').cast(IntegerType()))
        df = df.na.fill(0, subset=['likes', 'reply_count'])

        # Calculate score
        df = df.withColumn('engagement_score', (F.col('likes') * 0.7 + F.col('reply_count') * 0.3))

        # Categorize (using CaseWhen)
        df = df.withColumn('engagement_level',
                           F.when(F.col('engagement_score') <= 1, 'low')
                           .when((F.col('engagement_score') > 1) & (F.col('engagement_score') <= 5), 'medium')
                           .otherwise('high'))
        return df

    def create_time_windows(self, df, window_hours=1, timestamp_col='timestamp'):
        """Add a time_window column by flooring the timestamp."""
        seconds_in_window = window_hours * 3600
        df = df.withColumn('time_window',
           F.from_unixtime(
               (F.floor(F.unix_timestamp(F.col(timestamp_col)) / seconds_in_window) * seconds_in_window)
           ).cast(TimestampType())
        )
        return df

    def full_preprocessing_pipeline(self, df):
        """Execute complete Spark preprocessing pipeline."""
        logger.info("Starting Spark preprocessing pipeline...")

        if df.limit(1).count() == 0:
            logger.warning("Input DataFrame is empty, skipping pipeline.")
            return df

        # Clean and prep
        df = self.remove_duplicates(df)
        df = self.remove_spam_comments(df)

        # Sentiment Analysis
        df = self.analyze_sentiment(df)

        # Feature Engineering
        df = self.add_timestamp_features(df)
        df = self.add_text_features(df)
        df = self.add_engagement_features(df)
        df = self.create_time_windows(df)

        logger.info("Spark preprocessing pipeline completed")
        return df

    def save_processed_data(self, df, output_path=FILE_PATHS['processed_sentiment']):
        """Save processed Spark DataFrame to a single CSV file."""
        logger.info(f"Saving processed Spark DataFrame to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as a single CSV file for the dashboard
        # This is not scalable for huge data, but matches your original goal.
        # Use .toPandas() as a stable way to write a single file.
        try:
            pandas_df = df.toPandas()
            pandas_df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving Spark data: {e}")
            logger.info("Trying alternative save method (coalesce)...")
            try:
                # Fallback to coalesce
                df.coalesce(1).write.mode('overwrite').option('header', 'true').csv(output_path + "_temp_spark")
                logger.warning(f"Data saved to folder: {output_path}_temp_spark. Manual rename needed.")
            except Exception as e2:
                logger.error(f"Coalesce save method also failed: {e2}")

        return output_path

    def get_summary_statistics(self, df):
        """Get summary statistics using Spark aggregations."""
        logger.info("Calculating summary statistics with Spark...")

        if df.limit(1).count() == 0:
            return {'error': 'No data to summarize.'}

        # Cache the DataFrame as it will be used for multiple aggregations
        df.cache()

        stats_df = df.agg(
            F.count(F.lit(1)).alias('total_comments'),
            F.countDistinct('author').alias('total_authors'),
            F.mean('text_length').alias('average_text_length'),
            F.mean('word_count').alias('average_word_count'),
            F.sum('likes').alias('total_likes'),
            F.sum('reply_count').alias('total_replies'),
            F.min('timestamp').alias('min_timestamp'),
            F.max('timestamp').alias('max_timestamp'),
            F.mean('compound_score').alias('average_compound_score')
        ).collect()[0]

        stats = stats_df.asDict()

        if stats['min_timestamp']:
            stats['date_range'] = f"{stats.pop('min_timestamp')} to {stats.pop('max_timestamp')}"
        else:
            stats['date_range'] = "N/A"
            stats.pop('max_timestamp', None)

        # Get sentiment distribution
        sentiment_dist = df.groupBy('sentiment').count().collect()
        stats['sentiment_distribution'] = {row['sentiment']: row['count'] for row in sentiment_dist}

        # Uncache
        df.unpersist()

        return stats