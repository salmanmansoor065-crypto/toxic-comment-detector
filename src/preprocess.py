"""
Data preprocessing for Toxic Comment Classification
"""
import pandas as pd
import re
import numpy as np

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"\n", " ", text)                     # remove newlines
    text = re.sub(r"[^a-z0-9\s']", " ", text)           # remove special chars
    text = re.sub(r"\s+", " ", text).strip()            # remove extra spaces
    return text

def load_and_clean(path):
    df = pd.read_csv(path)
    df["clean_text"] = df["comment_text"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_and_clean(r"C:\Users\Dell\Desktop\toxic_project\data\train.csv")
    print("Shape:", df.shape)
    print(df[["comment_text", "clean_text"] + LABEL_COLS].head(3))
    print("\nLabel distribution (positive counts):")
    print(df[LABEL_COLS].sum())