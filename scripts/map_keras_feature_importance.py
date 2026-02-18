import joblib
import csv

# Path to the saved TF-IDF vectorizer
VECTORIZER_PATH = "models/tfidf_description_vectorizer.joblib"
# Path to the feature importance CSV
FEATURE_IMPORTANCE_CSV = "model_results/keras/feature_importance.csv"

# Load the vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = vectorizer.get_feature_names_out()

# Read feature importance CSV
with open(FEATURE_IMPORTANCE_CSV, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    print(f"{'Feature':<10} {'Word/Column':<30} {'Importance'}")
    print("-" * 60)
    for row in reader:
        feature = row["feature"]
        importance = float(row["importance"])
        if feature.startswith("f"):
            idx = int(feature[1:])
            # Defensive: check if index is in range
            if 0 <= idx < len(feature_names):
                word = feature_names[idx]
            else:
                word = "[OUT OF RANGE]"
        else:
            word = "[UNKNOWN]"
        print(f"{feature:<10} {word:<30} {importance}")
