# Automated Hyperparameter Tuning for XGBoost and Keras

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load data (adjust paths as needed)
df = pd.read_parquet('feature_store/train_features.parquet')
X = df.select_dtypes(include=["int", "float", "bool"]).fillna(0).astype('float32')
y = df['category']
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- XGBoost Hyperparameter Search ---
xgb_model = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss')
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'min_child_weight': [1, 3, 5]
}
rs_xgb = RandomizedSearchCV(
    xgb_model, xgb_param_grid, n_iter=10, scoring=make_scorer(f1_score, average='weighted'),
    cv=3, verbose=2, n_jobs=-1, random_state=42
)
rs_xgb.fit(X, y_enc)
print("Best XGBoost params:", rs_xgb.best_params_)
print("Best XGBoost F1 score:", rs_xgb.best_score_)

# --- Keras Hyperparameter Search ---
def build_keras_model(optimizer='adam', dropout1=0.3, dropout2=0.2, units1=256, units2=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(units1, activation='relu'),
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(units2, activation='relu'),
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(len(np.unique(y_enc)), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

keras_model = KerasClassifier(build_fn=build_keras_model, epochs=10, batch_size=256, verbose=0)
keras_param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout1': [0.2, 0.3, 0.4],
    'dropout2': [0.1, 0.2, 0.3],
    'units1': [128, 256, 512],
    'units2': [64, 128, 256]
}
rs_keras = RandomizedSearchCV(
    keras_model, keras_param_grid, n_iter=8, scoring=make_scorer(f1_score, average='weighted'),
    cv=3, verbose=2, n_jobs=1, random_state=42
)
rs_keras.fit(X, y_enc)
print("Best Keras params:", rs_keras.best_params_)
print("Best Keras F1 score:", rs_keras.best_score_)
