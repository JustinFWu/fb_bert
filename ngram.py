import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("data/clustered_comments_BERT.csv")

# Combine all posts into one text corpus
all_text = " ".join(df['cleaned_text'].dropna().astype(str).tolist())


# Function to get top N n-grams
def get_top_ngrams(text, ngram_range=(2,3), top_n=20):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform([text])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_n]

# Get top bigrams and trigrams
top_ngrams = get_top_ngrams(all_text, ngram_range=(2,3), top_n=20)

# Display results
print("Top N-grams (2-3 words):")
for phrase, freq in top_ngrams:
    print(f"{phrase}: {freq}")

# # Optional: Visualize top n-grams
# phrases, freqs = zip(*top_ngrams)
# plt.figure(figsize=(10,6))
# plt.barh(phrases[::-1], freqs[::-1])
# plt.title("Top N-grams in Teacher Posts (2-3 word phrases)")
# plt.xlabel("Frequency")
# plt.tight_layout()
# plt.show()

# -------------------------
# Extra: Help-Seeking Patterns Detection
# -------------------------

# Define common help-seeking phrases
help_patterns = [
    r"how do i", r"can anyone", r"any advice", r"what should i", 
    r"recommend", r"suggest", r"best way to", r"how to use", r"need help"
]

# Count help-seeking pattern occurrences
pattern_counts = {}
for pattern in help_patterns:
    matches = re.findall(pattern, all_text)
    pattern_counts[pattern] = len(matches)

# Display pattern counts
print("\nHelp-Seeking Patterns Found:")
for pattern, count in pattern_counts.items():
    print(f"'{pattern}': {count}")