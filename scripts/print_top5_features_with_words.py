import joblib
import csv
import pandas as pd

# Keras mapping
KERAS_CSV = "model_results/keras/feature_importance.csv"
VECTORIZER_PATH = "models/tfidf_description_vectorizer.joblib"

# XGBoost mapping
XGB_CSV = "model_results/xgboost/feature_importance.csv"
XGB_MAPPED = "model_results/xgboost/feature_importance_mapped.csv"

# Load vectorizer
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = vectorizer.get_feature_names_out()

# Keras: Map fXX to word
keras_rows = []
with open(KERAS_CSV, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        feature = row["feature"]
        importance = float(row["importance"])
        word = ""
        if feature.startswith("f"):
            idx = int(feature[1:])
            if 0 <= idx < len(feature_names):
                word = feature_names[idx]
            else:
                word = "[OUT OF RANGE]"
        keras_rows.append({
            "feature": feature,
            "word": word,
            "importance": importance
        })

keras_df = pd.DataFrame(keras_rows)
keras_df = keras_df.sort_values("importance", ascending=False).reset_index(drop=True)
keras_df["rank"] = keras_df.index + 1
keras_df = keras_df[["rank", "feature", "word", "importance"]]
keras_df = keras_df.head(5)

# XGBoost: Use mapped CSV
xgb_df = pd.read_csv(XGB_MAPPED)
xgb_df = xgb_df.rename(columns={"word_or_column": "word"})
xgb_df = xgb_df.sort_values("importance", ascending=False).reset_index(drop=True)
xgb_df["rank"] = xgb_df.index + 1
xgb_df = xgb_df[["rank", "feature", "word", "importance"]]
xgb_df = xgb_df.head(5)

# Print both tables
print("KERAS TOP 5 FEATURES:")
print(keras_df.to_string(index=False))
print("\nXGBOOST TOP 5 FEATURES:")
print(xgb_df.to_string(index=False))

# Save to CSVs for reference
keras_df.to_csv("model_results/keras/feature_importance_top5_with_words.csv", index=False)
xgb_df.to_csv("model_results/xgboost/feature_importance_top5_with_words.csv", index=False)
