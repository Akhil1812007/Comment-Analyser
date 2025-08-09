from fastapi import FastAPI,Depends, Query, HTTPException
import pandas as pd
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os


nltk.download('vader_lexicon')

API_KEY = os.getenv("YOUTUBE_API_KEY")  # set this in environment variable

app = FastAPI()

# 🔹 Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# ------------------------
# 1️⃣ Search for videos
# ------------------------
def search_videos(query, max_results=5):
    request = youtube.search().list(
        q=query, part="snippet", type="video",
        maxResults=max_results, order="relevance"
    )
    response = request.execute()
    return [{"video_id": i["id"]["videoId"], "title": i["snippet"]["title"]}
            for i in response.get("items", [])]

def get_video_comments(video_id, max_comments=50):
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            pageToken=next_page_token, textFormat="plainText"
        )
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

def fetch_and_analyze(query, videos_limit=5, comments_per_video=50):
    sia = SentimentIntensityAnalyzer()
    videos = search_videos(query, max_results=videos_limit)
    results = []
    for video in videos:
        comments = get_video_comments(video["video_id"], max_comments=comments_per_video)
        if not comments:
            continue
        scores = [sia.polarity_scores(c)["compound"] for c in comments]
        avg_score = sum(scores) / len(scores)
        results.append({
            "video_id": video["video_id"],
            "title": video["title"],
            "avg_sentiment": avg_score,
            "positive_percent": sum(s > 0.05 for s in scores) / len(scores) * 100,
            "negative_percent": sum(s < -0.05 for s in scores) / len(scores) * 100,
            "total_comments": len(comments)
        })
    return sorted(results, key=lambda x: x["avg_sentiment"], reverse=True)


API_SECRET = os.getenv("SERVICE_API_SECRET", "changeme123")

def verify_api_key(api_key: str = Query(...)):
    if api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Key")
# ------------------------
# 4️⃣ Run the Script
# ------------------------
@app.get("/rank_videos")
def rank_videos(
    topic: str,
    videos_limit: int = 5,
    comments_per_video: int = 50,
    _: None = Depends(verify_api_key)
):
    ranked = fetch_and_analyze(topic, videos_limit, comments_per_video)
    return {"topic": topic, "results": ranked}
