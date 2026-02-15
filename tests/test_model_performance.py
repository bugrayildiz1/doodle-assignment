import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import f1_score
import joblib
import time

# Paths to features and models
XGB_MODEL_PATH = "../models/xgb_model.joblib"
KERAS_MODEL_PATH = "../models/keras_model"
TEST_FEATURES_PATH = "../feature_store/test_features.parquet"
LABEL_COL = "category"

# Load test data
features = pd.read_parquet(TEST_FEATURES_PATH)
X_test = features.drop([LABEL_COL, 'subcategory'], axis=1, errors='ignore')
y_test = features[LABEL_COL]

# XGBoost
start = time.time()
xgb_model = joblib.load(XGB_MODEL_PATH)
xgb_preds = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_preds, average='weighted')
xgb_latency = (time.time() - start) / len(X_test)
print(f"XGBoost Weighted F1: {xgb_f1:.4f}, Avg Latency: {xgb_latency*1000:.2f} ms/sample")

# Keras
start = time.time()
keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
keras_preds = np.argmax(keras_model.predict(X_test), axis=1)
# If label encoding was used, decode here
keras_f1 = f1_score(y_test, keras_preds, average='weighted')
keras_latency = (time.time() - start) / len(X_test)
print(f"Keras Weighted F1: {keras_f1:.4f}, Avg Latency: {keras_latency*1000:.2f} ms/sample")

# Compare
if xgb_f1 > 0.85:
    print("XGBoost meets F1 target.")
else:
    print("XGBoost does NOT meet F1 target.")
if keras_f1 > 0.85:
    print("Keras meets F1 target.")
else:
    print("Keras does NOT meet F1 target.")
