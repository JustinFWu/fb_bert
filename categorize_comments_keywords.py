import re

# Define keyword categories
CHALLENGE_KEYWORDS = {
    'error', 'fail', 'failure', 'problem', 'issue', 'bug', 'crash', 'broken',
    'ban', 'cheat', 'bias', 'unfair', 'hallucinate', 'hallucination',
    'confuse', 'confusion', 'concern', 'worried', 'worry', 'risk', 'danger',
    'policy', 'restriction', 'ethics', 'ethical', 'privacy', 'plagiarism',
    'illegal', 'inaccuracy', 'inaccurate', 'overload', 'uncertain',
    'unsure', 'difficulty', 'limitations', 'misuse', 'glitch', 'wrong',
    'violation', 'violate', 'trust', 'transparency', 'blackbox', 'fear'
}


ADVICE_KEYWORDS = {
    'recommend', 'recommendation', 'suggest', 'suggestion', 'tutorial',
    'tips', 'tip', 'advice', 'resource', 'resources', 'guide', 'guidance',
    'manual', 'how', 'how-to', 'help', 'support', 'tools', 'tool', 'walkthrough',
    'learn', 'learning', 'strategy', 'strategies', 'workflow', 'step', 'steps',
    'instruction', 'training', 'question', 'faq', 'example', 'template'
}


def categorize_keywords(keywords):
    keywords = set(k.lower() for k in keywords)
    challenges = len(keywords & CHALLENGE_KEYWORDS)
    advice = len(keywords & ADVICE_KEYWORDS)

    if challenges > 0 and advice == 0:
        return "challenges"
    elif advice > 0 and challenges == 0:
        return "advice"
    elif challenges > 0 and advice > 0:
        return "both"
    else:
        return "other"

# Input/output files
INPUT_FILE = "hdb-clusters/comments_hdbscan_summary.txt"
OUTPUT_FILE = "hdb-clusters/comments_categorized_clusters.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Split by clusters
clusters = re.split(r"#+ Cluster (\d+) #+", content)

# Even indexes are text separators, odd indexes are cluster numbers + content
cluster_entries = []
for i in range(1, len(clusters), 2):
    cluster_id = clusters[i].strip()
    cluster_text = clusters[i + 1]

    # Extract keywords
    keyword_match = re.search(r"Top Keywords:\n(.+?)\n", cluster_text)
    if keyword_match:
        keywords = [kw.strip() for kw in keyword_match.group(1).split(",")]
        label = categorize_keywords(keywords)
    else:
        keywords = []
        label = "other"

    cluster_entries.append((cluster_id, label, keywords))

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for cluster_id, label, keywords in cluster_entries:
        f.write(f"Cluster {cluster_id}: {label.upper()}\n")
        f.write(f"Keywords: {', '.join(keywords)}\n\n")

print(f"####### Categorized clusters written to {OUTPUT_FILE}")
