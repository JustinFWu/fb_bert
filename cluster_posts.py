import torch
import pandas as pd
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os


embeddings  = torch.load("embeddings/post_embeddings.pt")
dataframe = pd.read_csv("data/cleaned_posts.csv")

print("Reducing dimensions with UMAP.....")
umap_model = umap.UMAP(n_components=2, random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings.numpy())

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_embeddings)

dataframe['cluster'] = cluster_labels
dataframe['x'] = reduced_embeddings[:, 0]
dataframe['y'] = reduced_embeddings[:, 1]

for i in range(num_clusters):
    print(f"\n #########Cluster {i}")
    sample = dataframe[dataframe['cluster'] == i]['cleaned_text'].head(5).tolist()
    for j, text in enumerate(sample):
        print(f"  [{j+1}] {str(text)[:200]}...")


# Create output folder if it doesn't exist
# os.makedirs("kmean-clusters", exist_ok=True)

# with open("kmean-clusters/clusters-5D-5.txt", "w", encoding="utf-8") as f:
#     for i in range(num_clusters):
#         cluster_posts = dataframe[dataframe['cluster'] == i]['cleaned_text'].head(5).tolist()

#         f.write(f"######## Cluster {i}\n\n")
#         for j, post in enumerate(cluster_posts):
#             f.write(f"[{j+1}] {str(post)}\n")  # make sure post is a string

#         f.write("\n\n")  # spacing between clusters

# print("✅ All clusters written to kmean-clusters/all_clusters.txt")

plt.figure(figsize=(10,6))
sns.scatterplot(data=dataframe, x='x', y='y', hue='cluster', palette='tab10')
plt.title("UMAP + KMeans clustering of posts")
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("plots/post_3cluster.png")
plt.show()

# for i in range(num_clusters):
#     print("1111")
#     print(f"\n🧠 Cluster {i}")
#     sample = dataframe[dataframe['cluster'] == i]['cleaned_text'].head(3).tolist()
#     for j, text in enumerate(sample):
#         print(f"  [{j+1}] {text[:200]}...")