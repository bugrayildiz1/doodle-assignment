import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import os
import joblib

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_features_labels(feature_path, label_col="category"):
    df = pd.read_parquet(feature_path)
    # Only keep numeric columns for XGBoost
    X = df.drop([label_col, 'subcategory'], axis=1, errors='ignore')
    X = X.select_dtypes(include=["int", "float", "bool"]).copy()
    y = df[label_col]
    return X, y

def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        max_depth=4,                # Lowered from default (6)
        min_child_weight=3,         # Increased from default (1)
        subsample=0.8,              # Add row subsampling
        colsample_bytree=0.8,       # Add feature subsampling
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        learning_rate=0.05,         # Lower learning rate
        n_estimators=200            # More trees to compensate for lower learning rate
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    return model

def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    f1 = f1_score(y, preds, average='weighted')
    print(f"{split_name} Weighted F1: {f1:.4f}")
    print(classification_report(y, preds))

    # Ensure model_results directory exists
    os.makedirs("model_results/xgboost", exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{split_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'model_results/xgboost/confusion_matrix_{split_name.lower()}.png')
    plt.close()

    return f1

def main():
    mlflow.set_experiment("ticket_categorization_xgboost")
    with mlflow.start_run():
        X_train, y_train_raw = load_features_labels("feature_store/train_features.parquet")
        X_val, y_val_raw = load_features_labels("feature_store/val_features.parquet")
        X_test, y_test_raw = load_features_labels("feature_store/test_features.parquet")

        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val = le.transform(y_val_raw)
        y_test = le.transform(y_test_raw)
        # Save encoder for later use
        os.makedirs("models", exist_ok=True)
        joblib.dump(le, "models/xgb_label_encoder.joblib")

        model = train_xgboost(X_train, y_train, X_val, y_val)
        mlflow.sklearn.log_model(model, "xgboost_model")

        f1_train = evaluate(model, X_train, y_train, "Train")
        f1_val = evaluate(model, X_val, y_val, "Validation")
        f1_test = evaluate(model, X_test, y_test, "Test")

        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_val", f1_val)
        mlflow.log_metric("f1_test", f1_test)

        # Plot and save feature importances
        import matplotlib.pyplot as plt
        xgb.plot_importance(model, max_num_features=20)
        plt.tight_layout()
        plt.savefig('model_results/xgboost/feature_importance.png')
        plt.close()


        # Save all feature importances to CSV (like Keras)
        import numpy as np
        import pandas as pd
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        fi_df = fi_df.sort_values('importance', ascending=False)
        fi_df.to_csv('model_results/xgboost/feature_importance.csv', index=False)

        # Print top 20 feature importances
        indices = np.argsort(importances)[::-1][:20]
        print("Top 20 Feature Importances:")
        for idx in indices:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        print("All feature importances saved to model_results/xgboost/feature_importance.csv")

        print("Model training and evaluation complete. Model logged to MLflow.")

if __name__ == "__main__":
    main()
