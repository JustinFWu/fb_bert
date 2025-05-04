import torch
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    patterns_to_remove = [
        r"warm welcome.*?joined us.*?please introduce.*get know.*",
        r"warm welcome.*?joined us",
        r"please introduce.*get know.*",
        r"hi\s+\w+",
        r"hello everyone", r"hey everyone",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)

    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)      # collapse whitespace
    return text.strip()

def remove_near_literal_duplicates(df, embeddings_tensor, column='cleaned_text'):
    seen = set()
    keep_rows = []
    keep_indices = []

    for idx, row in df.iterrows():
        norm = normalize_text(row[column])
        if norm not in seen:
            seen.add(norm)
            keep_rows.append(row)
            keep_indices.append(idx)

    new_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    new_embeddings = embeddings_tensor[keep_indices]
    return new_df, new_embeddings

embeddings = torch.load("embeddings/post_embeddings.pt")
dataframe = pd.read_csv("data/cleaned_posts.csv")
dataframe, embeddings = remove_near_literal_duplicates(dataframe, embeddings, column='cleaned_text')

print("Reducing dimensions with UMAP.....")
umap_model = umap.UMAP(n_components=10, random_state=42, init="random", metric='cosine')
reduced_embeddings = umap_model.fit_transform(embeddings.numpy())


# === HDBSCAN Clustering ===
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
cluster_labels = clusterer.fit_predict(reduced_embeddings)

dataframe['cluster'] = cluster_labels

# === TF-IDF Keyword Extraction ===
def get_top_keywords(df, text_column='cleaned_text', label_column='cluster', top_n=10):
    cluster_keywords = {}
    for cluster in sorted(df[label_column].unique()):
        if cluster == -1:
            continue
        texts = df[df[label_column] == cluster][text_column].dropna().tolist()
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        X = vectorizer.fit_transform(texts)
        indices = X.sum(axis=0).A1.argsort()[::-1][:top_n]
        keywords = [vectorizer.get_feature_names_out()[i] for i in indices]
        cluster_keywords[cluster] = keywords
    return cluster_keywords

print("\nExtracting keywords per cluster...")
keywords_per_cluster = get_top_keywords(dataframe)

# for cluster, keywords in keywords_per_cluster.items():
#     print(f"\n🔍 Cluster {cluster} Top Keywords: {', '.join(keywords)}")

def is_low_quality_cluster(texts, min_avg_len=20, uniqueness_threshold=0.5):
    if not texts:
        return True
    avg_len = sum(len(t) for t in texts) / len(texts)
    unique_ratio = len(set(texts)) / len(texts)
    return avg_len < min_avg_len or unique_ratio < uniqueness_threshold

# === Save Keywords + Sample Posts to File ===
os.makedirs("hdb-clusters", exist_ok=True)
output_path = "hdb-clusters/posts_hdbscan_summary.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for cluster in sorted(dataframe['cluster'].unique()):
        if cluster == -1:
            continue

        sample_comments = dataframe[dataframe['cluster'] == cluster]['cleaned_text'].dropna().head(10).tolist()

        # 🧹 Skip low-quality clusters
        if is_low_quality_cluster(sample_comments):
            continue

        keywords = keywords_per_cluster.get(cluster, [])
        f.write(f"######## Cluster {cluster} ########\n\n")
        f.write("🔑 Top Keywords:\n")
        f.write(", ".join(keywords) + "\n\n")
        f.write("💬 Sample Posts:\n")
        for idx, comment in enumerate(sample_comments, 1):
            f.write(f"[{idx}] {comment.strip()}\n")
        f.write("\n\n")

print(f"✅ Cluster summaries written to {output_path}")
dataframe.to_csv("data/clustered_posts.csv", index=False)
print("📦 Saved clustered DataFrame to data/clustered_posts.csv")



# (Optional Plotting Block — Uncomment if needed)
# umap_2d = umap.UMAP(n_components=2, random_state=42, init="random", metric='cosine')
# plot_embeddings = umap_2d.fit_transform(embeddings.numpy())
# dataframe['x'] = plot_embeddings[:, 0]
# dataframe['y'] = plot_embeddings[:, 1]

# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=dataframe, x='x', y='y', hue='cluster', palette='tab10', legend='full')
# plt.title("UMAP + HDBSCAN clustering of posts")
# plt.legend(title='Cluster')
# plt.tight_layout()
# plt.savefig("plots/posts_hdbscan_clusters.png")
# plt.show()
