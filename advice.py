import re
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Cluster Files
# -----------------------------

# Read cluster files
with open('hdb-clusters/comments_categorized_clusters_BERT.txt', 'r', encoding='utf-8') as f:
    comments_data = f.read()

with open('hdb-clusters/posts_categorized_clusters_BERT.txt', 'r', encoding='utf-8') as f:
    posts_data = f.read()

# -----------------------------
# 2. Extract ADVICE Clusters & Keywords
# -----------------------------

def extract_advice_keywords(text_data):
    advice_clusters = re.findall(r'Cluster \d+: ADVICE\nKeywords: (.+)', text_data)
    return [kw.strip() for cluster in advice_clusters for kw in cluster.split(',')]

# Extract keywords from ADVICE clusters in comments & posts
comments_advice_keywords = extract_advice_keywords(comments_data)
posts_advice_keywords = extract_advice_keywords(posts_data)

# Combine all ADVICE keywords
all_advice_keywords = comments_advice_keywords + posts_advice_keywords

# -----------------------------
# 3. Count Keyword Frequencies
# -----------------------------

keyword_counts = Counter(all_advice_keywords)
top_keywords = keyword_counts.most_common(30)

# Convert to DataFrame for report output
keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
print("\nTop Resource Needs from ADVICE Clusters:\n")
print(keyword_df)

# -----------------------------
# 4. Visualize Word Cloud
# -----------------------------

wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(keyword_counts)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Resource Needs (ADVICE Clusters)')
plt.show()

# -----------------------------
# 5. Visualize Bar Chart of Top Keywords
# -----------------------------

plt.figure(figsize=(12, 6))
plt.barh(keyword_df['Keyword'][::-1], keyword_df['Frequency'][::-1], color='steelblue')
plt.xlabel('Frequency')
plt.title('Top Resource Needs (ADVICE Clusters)')
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Save Table for Report
# -----------------------------

keyword_df.to_csv('data/advice_resource_needs_summary.csv', index=False)
print("\nSummary saved as 'advice_resource_needs_summary.csv'.")
