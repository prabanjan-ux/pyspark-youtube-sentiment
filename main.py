"""
main.py - Main orchestrator script for data collection and Spark processing
"""

import schedule
import time
import logging
from datetime import datetime
from youtube_fetcher import YouTubeFetcher
from data_processor import SparkDataProcessor
from config import CONFIG, FILE_PATHS

# --- New Imports ---
import sys
import os
import nltk
# --- End New Imports ---

# --- PySpark Import ---
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- New Function ---
def setup_nltk_data():
    """Download required NLTK data before starting Spark."""
    try:
        nltk.data.find('vader_lexicon')
        logger.info("NLTK 'vader_lexicon' already downloaded.")
    except LookupError:
        logger.info("Downloading NLTK 'vader_lexicon'...")
        nltk.download('vader_lexicon')
        logger.info("NLTK 'vader_lexicon' download complete.")

    # Add 'punkt' as well, as it's sometimes needed
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' already downloaded.")
    except LookupError:
        logger.info("Downloading NLTK 'punkt'...")
        nltk.download('punkt')
        logger.info("NLTK 'punkt' download complete.")
# --- End New Function ---


class SentimentAnalysisOrchestrator:
    """Orchestrate the entire sentiment analysis pipeline using PySpark."""

    def __init__(self, video_url):
        if not SPARK_AVAILABLE:
            logger.error("PySpark not found. Cannot run orchestrator.")
            raise ImportError("PySpark is required for this application.")

        self.video_url = video_url
        self.fetcher = YouTubeFetcher()
        self.file_paths = FILE_PATHS # Added this line

        # --- THIS IS THE CRITICAL FIX ---
        # Get the path to the Python executable in the current venv
        python_executable = sys.executable

        # Set this path for both the driver and the worker
        os.environ['PYSPARK_PYTHON'] = python_executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_executable

        # Add a log message to prove it's working
        logger.info(f"CRITICAL: PYSPARK_PYTHON set to {os.environ.get('PYSPARK_PYTHON')}")

        # Initialize Spark Session
        self.spark = SparkSession.builder \
            .appName("YouTubeSentimentAnalysis") \
            .master("local[*]") \
            .config("spark.pyspark.python", python_executable) \
            .config("spark.pyspark.driver.python", python_executable) \
            .getOrCreate()
        # --- END CRITICAL FIX ---

        # Initialize Spark Processor
        self.processor = SparkDataProcessor(self.spark)
        logger.info("SentimentAnalysisOrchestrator initialized successfully.")

    def fetch_and_analyze(self):
        """Execute complete pipeline: fetch -> load to Spark -> analyze -> save"""
        try:
            logger.info(f"Starting pipeline for: {self.video_url}")

            # Step 1: Fetch comments (still uses Pandas to write to CSV)
            logger.info("Step 1: Fetching comments...")
            new_comments_df = self.fetcher.save_comments_to_csv(self.video_url)

            if new_comments_df is None or new_comments_df.empty:
                logger.warning("No new comments fetched. Proceeding with existing data if available.")

            # Step 2: Load ALL raw comments into Spark
            logger.info("Step 2: Loading data into Spark...")
            raw_data_path = self.file_paths["raw_comments"]
            df_spark = self.processor.load_data(raw_data_path)

            # Swapped to a safer check
            if df_spark.limit(1).count() == 0:
                logger.warning("No data loaded into Spark. Aborting pipeline.")
                return False

            # Step 3: Process data using Spark pipeline
            logger.info("Step 3: Processing data with Spark...")
            df_processed = self.processor.full_preprocessing_pipeline(df_spark)

            # Step 4: Save processed data (from Spark to CSV)
            logger.info("Step 4: Saving data...")
            self.processor.save_processed_data(df_processed)

            # Step 5: Get and log statistics
            logger.info("Step 5: Calculating statistics...")
            stats = self.processor.get_summary_statistics(df_processed)
            logger.info(f"Summary: {stats.get('total_comments')} comments processed")
            if 'sentiment_distribution' in stats:
                logger.info(f"Sentiments: {stats['sentiment_distribution']}")

            logger.info("Pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
            return False

    def schedule_periodic_fetch(self, interval_minutes=CONFIG['fetch_interval'] // 60):
        logger.info(f"Scheduling fetch every {interval_minutes} minutes")
        schedule.every(interval_minutes).minutes.do(self.fetch_and_analyze)

    def run_scheduler(self):
        logger.info("Starting scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)

    def stop_spark(self):
        """Stop the Spark session."""
        logger.info("Stopping Spark session...")
        self.spark.stop()
        logger.info("Spark session stopped successfully.")


def run_single_analysis(video_url):
    """Run single analysis without scheduling"""
    if not SPARK_AVAILABLE:
        logger.error("PySpark not found. Please install pyspark.")
        return

    logger.info("Running single analysis...")
    orchestrator = None
    try:
        # --- Call NLTK setup here ---
        setup_nltk_data()
        orchestrator = SentimentAnalysisOrchestrator(video_url)
        return orchestrator.fetch_and_analyze()
    finally:
        if orchestrator:
            orchestrator.stop_spark()


def run_continuous_analysis(video_url, interval_minutes=5):
    """Run continuous analysis with scheduling"""
    if not SPARK_AVAILABLE:
        logger.error("PySpark not found. Please install pyspark.")
        return

    logger.info(f"Starting continuous analysis with {interval_minutes} minute intervals")
    orchestrator = None
    try:
        # --- Call NLTK setup here ---
        setup_nltk_data()
        orchestrator = SentimentAnalysisOrchestrator(video_url)
        orchestrator.fetch_and_analyze()
        orchestrator.schedule_periodic_fetch(interval_minutes)
        orchestrator.run_scheduler()
    finally:
        if orchestrator:
            orchestrator.stop_spark()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YouTube Sentiment Analysis Orchestrator"
    )
    parser.add_argument(
        "video_url",
        type=str,
        help="YouTube video URL or video ID"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode with periodic updates"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Update interval in minutes (default: 5)"
    )

    args = parser.parse_args()

    if not SPARK_AVAILABLE:
        logger.error("PySpark is not installed. Please install it with: pip install pyspark")
    else:
        try:
            if args.continuous:
                run_continuous_analysis(args.video_url, args.interval)
            else:
                run_single_analysis(args.video_url)
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)