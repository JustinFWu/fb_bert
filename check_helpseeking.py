import pandas as pd
import re

HELP_SEEKING_PHRASES = [
    "how do i", "how can i", "how to", "can anyone", "can someone", "does anyone know",
    "what should i", "any advice", "any suggestions", "any tips", "is there a way",
    "i need help", "has anyone tried", "i'm struggling with", "what do you use for",
    "any idea how", "could someone explain", "what's the best way to", "i'm stuck on",
    "any help appreciated", "need advice", "question about", "looking for help", "is it possible to"
]

HELPFUL_PHRASES = [
    "i recommend", "i suggest", "try", "worked for me", "you can use", "i had similar issue",
    "you should", "what worked for me", "here's what i do", "my approach is", "check out",
    "i usually", "one way is", "i found that", "you might want to", "this tool helps",
    "this might help", "hope this helps", "you could try", "you might try"
]


# Load input data
posts_df = pd.read_csv("data/posts_with_sentiment_analysis.csv")
comments_df = pd.read_csv("data/comments_with_sentiment_analysis.csv")

# If comments_df has a column linking to the original post, e.g., "post_text"
# Rename it to "post_id" for consistency
comments_df.rename(columns={"text": "post_id"}, inplace=True)  # Adjust if your column name is different

# Flag help-seeking posts
def is_help_seeking(text):
    text = str(text).lower()
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in HELP_SEEKING_PHRASES)

posts_df["help_seeking"] = posts_df["cleaned_text"].apply(is_help_seeking)

# Sentiment stats grouped by post
grouped = comments_df.groupby("post_id")["sentiment"].value_counts().unstack().fillna(0)
grouped["total_comments"] = grouped.sum(axis=1)
grouped["positive_ratio"] = grouped["POSITIVE"] / grouped["total_comments"]

# Detect helpful comments
def is_helpful_comment(text):
    text = str(text).lower()
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in HELPFUL_PHRASES)

comments_df["is_helpful"] = comments_df["cleaned_text"].apply(is_helpful_comment)

# Support score per post
helpful_comments = comments_df.groupby("post_id")["is_helpful"].sum()
total_comments = comments_df.groupby("post_id")["cleaned_text"].count()
support_score = (helpful_comments / total_comments).fillna(0)

# Merge support score back to posts (based on post text match)
posts_df = posts_df.set_index("text")
posts_df["support_score"] = support_score
posts_df.reset_index(inplace=True)

# Also add support score to grouped comment stats
grouped["helpful_comments"] = helpful_comments
grouped["support_score"] = support_score

# Print results
print("=== Top Help-Seeking Posts with High Support Scores ===")
print(posts_df[posts_df["help_seeking"]].sort_values("support_score", ascending=False)[["text", "support_score"]].head(10))

print("\n=== Help-Seeking Posts with Low Support Scores ===")
print(posts_df[posts_df["help_seeking"]].sort_values("support_score", ascending=True)[["text", "support_score"]].head(10))

# Save enriched data
posts_df.to_csv("data/posts_with_support_score.csv", index=False)
grouped.to_csv("data/post_comment_support_stats.csv")
