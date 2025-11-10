# YouTube Comments Sentiment Analysis with Kafka

This project fetches YouTube comments and streams them to Apache Kafka for real-time processing.

## Features

- Fetch YouTube comments using YouTube Data API v3
- Stream comments to Kafka topic in real-time
- Kafka consumer example for processing comments
- Graceful fallback if Kafka is unavailable

## Prerequisites

- Python 3.8+
- Apache Kafka (optional, for streaming)
- YouTube Data API key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Install and start Kafka:
   - Download Kafka from https://kafka.apache.org/downloads
   - Start Zookeeper:
     ```bash
     # Linux/Mac
     bin/zookeeper-server-start.sh config/zookeeper.properties
     
     # Windows
     bin\windows\zookeeper-server-start.bat config\zookeeper.properties
     ```
   - Start Kafka broker:
     ```bash
     # Linux/Mac
     bin/kafka-server-start.sh config/server.properties
     
     # Windows
     bin\windows\kafka-server-start.bat config\server.properties
     ```

## Usage

### Fetch Comments and Stream to Kafka

```bash
python app.py
```

This will:
- Fetch comments from the configured YouTube video
- Stream each comment to the `youtube-comments` Kafka topic
- Display progress and summary

### Consume Comments from Kafka

In a separate terminal:

```bash
python kafka_consumer.py
```

This will consume and display comments from the Kafka topic in real-time.

### Disable Kafka Streaming

To fetch comments without Kafka, edit `app.py`:

```python
df = fetch_youtube_comments(
    video_id=video_id,
    api_key=api_key,
    kafka_topic=None,  # Set to None to disable Kafka
)
```

## Configuration

### app.py

- `api_key`: Your YouTube Data API key
- `video_id`: YouTube video ID to fetch comments from
- `kafka_topic`: Kafka topic name (default: `youtube-comments`)
- `kafka_bootstrap_servers`: Kafka broker address (default: `localhost:9092`)

### kafka_consumer.py

- `topic`: Kafka topic to consume from
- `bootstrap_servers`: Kafka broker address
- `group_id`: Consumer group ID

## Data Schema

Each comment message contains:

```json
{
  "comment_id": "unique_comment_id",
  "text": "comment text",
  "author": "author name",
  "published_at": "2024-01-01T12:00:00Z",
  "likes": 10,
  "video_id": "video_id"
}
```

## Troubleshooting

### Kafka Connection Failed

If you see "Kafka connection failed", the app will continue without streaming. Make sure:
- Kafka is running on `localhost:9092`
- Firewall allows connections to port 9092

### YouTube API Quota

The YouTube Data API has quota limits. If you exceed them:
- Wait 24 hours for quota reset
- Request quota increase in Google Cloud Console
- Use a different API key

## Next Steps

- Add sentiment analysis using transformers or TextBlob
- Implement real-time sentiment dashboard
- Add data persistence (database, file storage)
- Create multiple consumers for parallel processing
- Add error handling and retry logic
