import joblib
import csv
import re
import pandas as pd

# Path to the saved TF-IDF vectorizer
VECTORIZER_PATH = "models/tfidf_description_vectorizer.joblib"
# Path to the XGBoost feature importance CSV
FEATURE_IMPORTANCE_CSV = "model_results/xgboost/feature_importance.csv"
# Output CSV path
OUTPUT_CSV = "model_results/xgboost/feature_importance_mapped.csv"

# Load the vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = vectorizer.get_feature_names_out()

def get_tfidf_word(feature):
    match = re.match(r"tfidf_(\d+)", feature)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            return feature_names[idx]
        else:
            return "[OUT OF RANGE]"
    return "[NOT TFIDF]"

rows = []
with open(FEATURE_IMPORTANCE_CSV, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        feature = row["feature"]
        importance = float(row["importance"])
        word = get_tfidf_word(feature) if feature.startswith("tfidf_") else feature
        rows.append({
            "feature": feature,
            "word_or_column": word,
            "importance": importance
        })

# Save to new CSV
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

# Print top 20
print(f"{'Feature':<20} {'Word/Column':<30} {'Importance'}")
print("-" * 70)
for row in rows[:20]:
    print(f"{row['feature']:<20} {row['word_or_column']:<30} {row['importance']}")
