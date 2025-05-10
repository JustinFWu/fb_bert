import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from tqdm import tqdm
from preprocessing import clean_text, load_and_clean_data

# gonna use some gpu acceleration just to speed up the embedding process, since the data is so much
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.to(device)
model.eval()

def get_roberta_embeddings(texts, batch_size=16):
    """ just to comvert what ever lists of texts passed in into embeddings"""
    embeddings = []

    # adding a little loading progress bar just to make life better
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding: "):
        batch_texts = texts[i:i+batch_size]

        encode = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**encode)

        cls_embeddings = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())

    return torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    posts_path = "data/facebook posts.xlsx"
    comments_path = "data/facebook comments.xlsx"

    posts_dataframe, comments_dataframe = load_and_clean_data(posts_path, comments_path)
    posts_dataframe.to_csv("data/cleaned_posts_BERT.csv", index=False)
    comments_dataframe.to_csv("data/cleaned_comments_BERT.csv", index=False)

    post_text = posts_dataframe['cleaned_text'].fillna("").tolist()
    comments_text = comments_dataframe['cleaned_text'].fillna("").tolist()

    post_embeddings = get_roberta_embeddings(post_text)
    comments_embeddings = get_roberta_embeddings(comments_text)

    torch.save(post_embeddings, "embeddings/post_embeddings_BERT.pt")
    torch.save(comments_embeddings, "embeddings/comments_embeddings_BERT.pt")

    print(" Posts Embedding shape:", post_embeddings.shape)
    print(" Comments Embedding shape:", comments_embeddings.shape)
