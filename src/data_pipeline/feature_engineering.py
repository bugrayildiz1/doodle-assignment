import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Example feature engineering for support tickets
def engineer_features(df):
    # Encode categorical features
    cat_cols = [
        'priority', 'severity', 'channel', 'customer_tier', 'product', 'product_module',
        'agent_specialization', 'business_impact', 'language', 'region'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Text features: subject + description
    df['text'] = df['subject'].fillna('') + ' ' + df['description'].fillna('')
    tfidf = TfidfVectorizer(max_features=256)
    tfidf_matrix = tfidf.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # Example: binary features
    for col in ['contains_error_code', 'contains_stack_trace', 'known_issue', 'bug_report_filed', 'weekend_ticket', 'after_hours']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Fill missing values
    df = df.fillna(0)
    # Convert all object columns to string to avoid ArrowTypeError
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

import os

def process_split(split_name, in_dir="data_splits", out_dir="feature_store"):
    in_path = os.path.join(in_dir, f"{split_name}.json")
    out_path = os.path.join(out_dir, f"{split_name}_features.parquet")
    df = pd.read_json(in_path, lines=True)
    features = engineer_features(df)
    os.makedirs(out_dir, exist_ok=True)
    features.to_parquet(out_path)
    print(f"{split_name} features saved to {out_path}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
