import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the JSON file
data_path = "support_tickets.json"

# Load the data
def load_tickets(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def split_data(df, stratify_col="category"):
    # 70/15/15 split
    train, temp = train_test_split(df, test_size=0.3, stratify=df[stratify_col], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[stratify_col], random_state=42)
    return train, val, test

def save_splits(train, val, test, out_dir="data_splits"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    train.to_json(f"{out_dir}/train.json", orient="records", lines=True)
    val.to_json(f"{out_dir}/val.json", orient="records", lines=True)
    test.to_json(f"{out_dir}/test.json", orient="records", lines=True)

if __name__ == "__main__":
    df = load_tickets(data_path)
    print(f"Loaded {len(df)} tickets.")
    train, val, test = split_data(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    save_splits(train, val, test)
    print("Data splits saved to data_splits/ directory.")
