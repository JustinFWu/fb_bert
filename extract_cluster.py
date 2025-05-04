import pandas as pd
import os


def extract_cluster_texts(csv_path, cluster_id, output_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'cluster' not in df.columns:
        raise ValueError("CSV must contain a 'cluster' column.")

    # Ensure cluster column is numeric
    df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce').astype('Int64')

    filtered_df = df[df['cluster'] == cluster_id]

    if filtered_df.empty:
        print(f" No entries found for cluster {cluster_id}.")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        filtered_df['cleaned_text'].to_csv(output_path, index=False, header=False)
        print(f" Saved {len(filtered_df)} comments to {output_path}")

if __name__ == "__main__":
    extract_cluster_texts("data/clustered_comments.csv", 179, "hdb-clusters/single_cluster.txt")