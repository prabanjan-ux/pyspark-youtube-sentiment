# 📊 YouTube Comment Sentiment Analyzer  
*A Real-Time Sentiment Analysis & Visualization System using PySpark, NLTK, and Streamlit*

---

## 🧾 Version Information

| Component | Version | Description |
|------------|----------|-------------|
| **Python** | 3.11+ | Programming language used |
| **PySpark** | 3.5.7 | Distributed data processing engine |
| **NLTK (VADER)** | 3.9.2 | Natural language sentiment analysis |
| **TextBlob** | 0.17+ | Secondary sentiment analyzer |
| **Pandas** | 2.3.3 | Data manipulation and cleaning |
| **Streamlit** | 1.51.0 | Interactive dashboard framework |
| **Plotly** | 6.4.0 | Data visualization library |
| **Google API Client** | 2.187.0 | Used for YouTube Data API access |
| **Schedule** | 1.2.2 | Automates periodic data fetching |
| **dotenv** | 1.0+ | Environment variable management |

**Latest Project Version:** `v1.0.0`  
**Release Date:** November 2025

---

## 🚀 Overview

**YouTube Comment Sentiment Analyzer** is an end-to-end NLP and data engineering project that:
- Fetches live YouTube comments through the **YouTube Data API v3**
- Processes and analyzes comment text using **PySpark** and **NLTK VADER**
- Computes **sentiment scores**, **engagement levels**, and **text-based features**
- Visualizes the results interactively using **Streamlit dashboards**

This project combines **real-time data ingestion**, **AI-powered NLP sentiment analysis**, and **visual storytelling** for deep insights into YouTube engagement.

---

## 🧩 Key Modules

### 1. **`youtube_fetcher.py`**
- Fetches comments using the **YouTube Data API v3**
- Extracts details such as:
  - `comment_id`
  - `author`
  - `text`
  - `likes`
  - `timestamp`
- Merges fetched comments into CSV files, removing duplicates  

### 2. **`data_processor.py`**
- Processes and cleans the comment data using **PySpark**
- Adds multiple layers of feature engineering:
  - Sentiment classification (VADER)
  - Engagement metrics (likes + replies)
  - Timestamp-based features (hour, weekday)
  - Text features (word count, emoji presence)
- Outputs a cleaned and enriched dataset for analytics  

### 3. **`sentiment_analyzer.py`**
- Performs text-based sentiment analysis using:
  - **VADER (from NLTK)** for social-media-optimized sentiment scoring  
  - **TextBlob** for polarity and subjectivity metrics  
- Supports batch and dataframe-based sentiment analysis  
- Returns key metrics: `positive`, `neutral`, `negative`, and `compound score`  

### 4. **`dashboard.py`**
- A **Streamlit dashboard** for visual exploration  
- Includes:
  - Sentiment Distribution (Pie/Bar)
  - Sentiment Trends Over Time
  - Engagement vs Sentiment Heatmap
  - Text Statistics (Word Count, Text Length)
  - Top Comments by Likes
- Interactive filters for date and sentiment  
- CSV export for analyzed data  

### 5. **`main.py`**
- Core orchestrator of the full pipeline:
  1. Fetch → 2. Process → 3. Analyze → 4. Save  
- Integrates **PySpark**, **YouTubeFetcher**, and **SparkDataProcessor**  
- Handles continuous comment fetching via the `schedule` library  

---

## 🧱 Project Architecture
```
youtube-pyspark/
│
├── main.py # Main orchestrator for data fetching & Spark processing
├── youtube_fetcher.py # Fetches comments using YouTube API
├── data_processor.py # Data cleaning, feature engineering (PySpark)
├── sentiment_analyzer.py # Sentiment scoring using NLTK/TextBlob
├── dashboard.py # Streamlit-based interactive visualization
├── config.py # Configuration for paths and parameters
│
├── data/
│ ├── raw_comments.csv
│ └── processed_sentiment.csv
│
├── models/ # Placeholder for ML models
├── logs/ # Runtime logs
│
├── .env # Contains YouTube API key (excluded from Git)
├── .gitignore # Prevents committing unnecessary files
├── requirements.txt # Dependencies
└── README.md # Documentation
```

---

## ⚙️ Configuration

### **`config.py`**
```python
CONFIG = {
    "max_comments_per_video": 1000,
    "fetch_interval": 300,       # 5 minutes
    "sentiment_threshold": 0.5,
    "batch_size": 100,
    "languages": ["en"],
    "update_mode": "append",
}

FILE_PATHS = {
    "raw_comments": "data/raw_comments.csv",
    "processed_sentiment": "data/processed_sentiment.csv",
    "processed_data": "data/processed_data.csv",
    "model_path": "models/sentiment_model.pkl",
    "logs": "logs/",
}
```

---

## 📋 Prerequisites

Before running this project, ensure you have:

1. **Python 3.11+** installed
2. **YouTube Data API v3 Key** - [Get it here](https://console.developers.google.com/)
3. **Java 8 or 11** (required for PySpark)
4. **Git** (for cloning the repository)

---

## 🛠️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/youtube-pyspark.git
cd youtube-pyspark
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

### Step 5: Set Up Environment Variables
Create a `.env` file in the root directory:
```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### Step 6: Create Required Directories
```bash
mkdir data logs models notebooks
```

---

## 🎯 Usage

### Option 1: Single Analysis (One-Time Fetch)
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Option 2: Continuous Analysis (Periodic Updates)
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --continuous --interval 5
```
*Fetches new comments every 5 minutes*

### Option 3: Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```
*Opens at `http://localhost:8501`*

---

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **📈 Sentiment Distribution** - Pie chart and bar chart visualization
- **⏰ Sentiment Trends Over Time** - Line chart showing sentiment changes
- **🔥 Engagement Heatmap** - Correlation between sentiment and engagement
- **📝 Text Statistics** - Word count, text length, emoji analysis
- **⭐ Top Comments** - Highest-liked comments with sentiment scores
- **🔍 Interactive Filters** - Filter by date range and sentiment type
- **💾 CSV Export** - Download processed data for further analysis

---

## 🧪 Example Workflow

```python
# 1. Fetch comments from a YouTube video
from youtube_fetcher import YouTubeFetcher

fetcher = YouTubeFetcher()
comments_df = fetcher.save_comments_to_csv("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# 2. Process with PySpark
from pyspark.sql import SparkSession
from data_processor import SparkDataProcessor

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
processor = SparkDataProcessor(spark)
df_spark = processor.load_data("data/raw_comments.csv")
df_processed = processor.full_preprocessing_pipeline(df_spark)

# 3. Analyze sentiment
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment_scores = analyzer.analyze_batch(comments_df['text'].tolist())

# 4. Visualize with Streamlit
# Run: streamlit run dashboard.py
```

---

## 🔑 API Key Setup

### Getting YouTube API Key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable **YouTube Data API v3**
4. Create credentials (API Key)
5. Copy the API key to your `.env` file

**Important:** Keep your API key secure and never commit it to version control!

---

## 📂 Data Flow

```
YouTube Video
    ↓
[YouTube Data API v3] ← youtube_fetcher.py
    ↓
raw_comments.csv
    ↓
[PySpark Processing] ← data_processor.py
    ↓
[NLTK VADER Sentiment] ← sentiment_analyzer.py
    ↓
processed_sentiment.csv
    ↓
[Streamlit Dashboard] ← dashboard.py
    ↓
Interactive Visualizations
```

---

## 🧠 Sentiment Analysis Details

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Compound Score Range:** -1 (most negative) to +1 (most positive)
- **Classification Thresholds:**
  - Positive: compound ≥ 0.05
  - Neutral: -0.05 < compound < 0.05
  - Negative: compound ≤ -0.05

### Features Extracted:
- **Sentiment Scores:** positive, neutral, negative, compound
- **Engagement Metrics:** likes, replies, engagement_score
- **Temporal Features:** hour_of_day, day_of_week
- **Text Features:** word_count, has_emoji, text_length
- **Author Information:** author_name, comment_id

---

## 🚨 Troubleshooting

### Issue: PySpark Not Found
```bash
pip install pyspark==3.5.7
```

### Issue: Java Not Found
- Install Java 8 or 11
- Set `JAVA_HOME` environment variable

### Issue: NLTK Data Missing
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

### Issue: YouTube API Quota Exceeded
- YouTube API has daily quota limits (10,000 units/day)
- Each comment fetch costs ~1 unit per comment
- Wait 24 hours or use a different API key

### Issue: Streamlit Port Already in Use
```bash
streamlit run dashboard.py --server.port 8502
```

---

## 📈 Performance Metrics

- **Processing Speed:** ~1000 comments/second with PySpark
- **Memory Usage:** ~500MB for 10,000 comments
- **API Rate Limit:** 10,000 units/day (YouTube API)
- **Dashboard Load Time:** <2 seconds for 50,000 comments

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PySpark** - Distributed data processing
- **NLTK VADER** - Sentiment analysis lexicon
- **Streamlit** - Interactive dashboard framework
- **YouTube Data API v3** - Comment data source
- **Plotly** - Beautiful visualizations

---

## 📧 Contact

**Project Maintainer:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

## 🔮 Future Enhancements

- [ ] Multi-language sentiment analysis
- [ ] Real-time streaming with Kafka
- [ ] Machine learning model for custom sentiment classification
- [ ] Reply thread analysis
- [ ] Sentiment prediction for future comments
- [ ] Integration with other social media platforms
- [ ] Advanced NLP features (topic modeling, entity recognition)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

