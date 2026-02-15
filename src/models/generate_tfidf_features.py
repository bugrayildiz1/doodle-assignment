import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_text_column(parquet_path, text_col):
    df = pd.read_parquet(parquet_path)
    return df[text_col].fillna("").astype(str)

def save_tfidf_features(split, text_col, tfidf_matrix, feature_names):
    out_path = f"feature_store/{split}_tfidf_{text_col}.parquet"
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{text_col}_{i}" for i in range(tfidf_matrix.shape[1])])
    tfidf_df.to_parquet(out_path)
    print(f"Saved {out_path}")

def main():
    splits = ["train", "val", "test"]
    text_col = "description"  # You can change to 'subject' or others as needed
    # Load all splits for fitting vectorizer on train and transforming all
    train_text = load_text_column("feature_store/train_features.parquet", text_col)
    val_text = load_text_column("feature_store/val_features.parquet", text_col)
    test_text = load_text_column("feature_store/test_features.parquet", text_col)

    # Fit TF-IDF on train, transform all
    tfidf = TfidfVectorizer(max_features=256, stop_words="english")
    X_train = tfidf.fit_transform(train_text)
    X_val = tfidf.transform(val_text)
    X_test = tfidf.transform(test_text)
    feature_names = tfidf.get_feature_names_out()

    save_tfidf_features("train", text_col, X_train.toarray(), feature_names)
    save_tfidf_features("val", text_col, X_val.toarray(), feature_names)
    save_tfidf_features("test", text_col, X_test.toarray(), feature_names)

    # Optionally save the vectorizer for inference
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf, f"models/tfidf_{text_col}_vectorizer.joblib")
    print(f"Saved models/tfidf_{text_col}_vectorizer.joblib")

if __name__ == "__main__":
    main()