import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class YouTubeFetcher:
    def __init__(self):
        from googleapiclient.discovery import build
        from config import YOUTUBE_API_KEY, FILE_PATHS

        self.api_key = YOUTUBE_API_KEY
        self.file_paths = FILE_PATHS
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    def fetch_comments(self, video_url):
        """Fetch comments for a given YouTube video"""
        try:
            # Extract video ID
            video_id = video_url.split("v=")[-1].split("&")[0]
            logger.info(f"Fetching comments from: {video_id}")

            comments = []
            next_page_token = None
            while True:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "comment_id": item["id"],
                        "author": snippet.get("authorDisplayName"),
                        "text": snippet.get("textDisplay"),
                        "like_count": snippet.get("likeCount"),
                        "published_at": snippet.get("publishedAt")
                    })

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            df = pd.DataFrame(comments)
            logger.info(f"Fetched {len(df)} comments from video {video_id}")
            return df

        except Exception as e:
            logger.error(f"Error fetching comments: {e}")
            return None

    def safe_read_csv(self, path):
        """Safely read CSV file (returns empty DataFrame if missing or empty)"""
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return pd.DataFrame()
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def save_comments_to_csv(self, video_url):
        """Fetch comments and save/merge to CSV"""
        try:
            new_df = self.fetch_comments(video_url)
            if new_df is None or len(new_df) == 0:
                logger.warning("No new comments fetched.")
                return None

            path = self.file_paths["raw_comments"]
            existing_df = self.safe_read_csv(path)

            # Merge and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["comment_id"], inplace=True)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            combined_df.to_csv(path, index=False)

            logger.info(f"Saved {len(new_df)} new comments (total {len(combined_df)}) to {path}")
            return new_df

        except Exception as e:
            logger.error(f"Error in save_comments_to_csv: {e}")
            return None
