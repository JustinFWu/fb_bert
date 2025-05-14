import pandas as pd
from transformers import pipeline
from tqdm import tqdm

comments_df = pd.read_csv("data/clustered_posts_BERT.csv")

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result["label"]
    except:
        return "UNKNOWN"
    
tqdm.pandas(desc="Analyzing Sentiment")
comments_df["sentiment"] = comments_df["cleaned_text"].progress_apply(get_sentiment)

sentiment_counts = comments_df.groupby("cluster")["sentiment"].value_counts().unstack().fillna(0)
sentiment_percent = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)

print("\nSentiment counts by cluster: ")
print(sentiment_counts.astype(int))

print("Sentiment percentages by cluster: ")
print(sentiment_percent.round(2))

with open("sentiment infor/sentiment_cluster_stats_posts.txt", "w") as f:
    f.write("Sentiment counts by cluster:\n")
    f.write(sentiment_counts.astype(int).to_string())
    f.write("\n\nSentiment percentages by cluster:\n")
    f.write(sentiment_percent.round(2).to_string())

comments_df.to_csv("data/posts_with_sentiment_analysis.csv", index=False)