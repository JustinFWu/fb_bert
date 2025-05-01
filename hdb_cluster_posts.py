import torch
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

embeddings = torch.load("embeddings/post_embeddings.pt")
dataframe = pd.read_csv("data/cleaned_posts.csv")

print("Reducing dimensions with UMAP.....")
umap_model = umap.UMAP(n_components=5, random_state=42, init="random", metric='cosine')
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

# === Save Keywords + Sample Posts to File ===
os.makedirs("hdb-clusters", exist_ok=True)

output_path = "hdb-clusters/posts_hdbscan_summary.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for cluster in sorted(dataframe['cluster'].unique()):
        if cluster == -1:
            continue  # skip noise

        f.write(f"######## Cluster {cluster} ########\n\n")

        # Write Top Keywords
        keywords = keywords_per_cluster.get(cluster, [])
        f.write("🔑 Top Keywords:\n")
        f.write(", ".join(keywords) + "\n\n")

        # Write Top Posts (up to 10)
        f.write(" Sample Posts:\n")
        sample_posts = dataframe[dataframe['cluster'] == cluster]['cleaned_text'].dropna().head(5).tolist()
        for idx, post in enumerate(sample_posts, 1):
            f.write(f"[{idx}] {post.strip()}\n")

        f.write("\n\n")  # spacing between clusters

print(f"✅ Post cluster summaries written to {output_path}")


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
