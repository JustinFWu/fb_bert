import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    """ 
    We aim to clean text by lowercasing everything, removing irrelevant stuff such as urls,
    removing non-alphabetic characters, extra spaces, and removing stopwords.
    """

    if pd.isnull(text):
        return ""
    
    # turn some text into lowercase
    text = text.lower()

    # remove all urls
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    #remove all special characters as well
    text = re.sub(r"[^a-zA-Z\s]", '', text)

    # remove all extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # remove all the stopwords
    stop_words = set(stopwords.words('english'))
    text_tokens = text.split()
    filtered_text = [word for word in text_tokens if word not in stop_words]

    return ' '.join(filtered_text)

def load_and_clean_data(posts_path, comments_path):
    """
    Load the posts and comments and get them cleaned
    """
    posts_dataframe = pd.read_excel(posts_path)
    comments_dataframe = pd.read_excel(comments_path)

    post_col = posts_dataframe.columns[1]
    comment_col = comments_dataframe.columns[1]

    posts_dataframe['cleaned_text'] = posts_dataframe[post_col].apply(clean_text)
    comments_dataframe['cleaned_text'] = comments_dataframe[comment_col].apply(clean_text)

    return posts_dataframe, comments_dataframe

if __name__ == "__main__":
    posts_path = "facebook posts.xlsx"
    comments_path = "facebook posts.xlsx"

    posts_dataframe, comments_dataframe = load_and_clean_data(posts_path, comments_path)

    print(posts_dataframe.head())
    print(comments_dataframe.head())