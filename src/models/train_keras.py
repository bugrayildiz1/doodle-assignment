import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import mlflow
import os
import joblib

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_features_labels(feature_path, label_col="category"):
    df = pd.read_parquet(feature_path)
    # Select only numeric columns for Keras
    X_num = df.select_dtypes(include=["int", "float", "bool"]).copy()
    X_num = X_num.fillna(0).astype('float32')
    y = df[label_col].values
    # Load matching TF-IDF features if available
    import os
    split = os.path.basename(feature_path).split('_')[0]  # train/val/test
    tfidf_path = f"feature_store/{split}_tfidf_description.parquet"
    if os.path.exists(tfidf_path):
        X_tfidf = pd.read_parquet(tfidf_path)
        X = np.concatenate([X_num.values, X_tfidf.values], axis=1)
    else:
        X = X_num.values
    return X, y

def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    mlflow.set_experiment("ticket_categorization_keras")
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
        joblib.dump(le, "models/keras_label_encoder.joblib")


        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        num_classes = len(np.unique(y_train))
        model = build_model(X_train.shape[1], num_classes)

        # Train model and store history
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=256,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)],
            class_weight=class_weight_dict,
            verbose=2
        )

        # Ensure model_results directory exists
        os.makedirs("model_results/keras", exist_ok=True)

        # Plot and save training/validation accuracy and loss curves
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(history.history['accuracy'], label='Train Acc')
        axs[0].plot(history.history['val_accuracy'], label='Val Acc')
        axs[0].set_title('Accuracy')
        axs[0].legend()
        axs[1].plot(history.history['loss'], label='Train Loss')
        axs[1].plot(history.history['val_loss'], label='Val Loss')
        axs[1].set_title('Loss')
        axs[1].legend()
        plt.tight_layout()
        plt.savefig('model_results/keras/training_curves.png')
        plt.close()

        mlflow.tensorflow.log_model(model, "keras_model")


        # Save confusion matrices for each split
        for split, X, y in zip(["Train", "Validation", "Test"], [X_train, X_val, X_test], [y_train, y_val, y_test]):
            preds = np.argmax(model.predict(X), axis=1)
            f1 = f1_score(y, preds, average='weighted')
            print(f"{split} Weighted F1: {f1:.4f}")
            print(classification_report(y, preds))
            mlflow.log_metric(f"f1_{split.lower()}", f1)

            # Confusion matrix
            cm = confusion_matrix(y, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{split} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'model_results/keras/confusion_matrix_{split.lower()}.png')
            plt.close()

        print("Keras model training and evaluation complete. Model logged to MLflow.")
        
        # Permutation importance for Keras model
        from sklearn.inspection import permutation_importance
        from sklearn.base import BaseEstimator, ClassifierMixin
        print("Computing permutation importance for Keras model on test set...")
        class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, model):
                self.model = model
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.argmax(self.model.predict(X), axis=1)
        keras_estimator = KerasClassifierWrapper(model)
        result = permutation_importance(
            estimator=keras_estimator,
            X=X_test,
            y=y_test,
            scoring=lambda est, X, y: f1_score(y, est.predict(X), average='weighted'),
            n_repeats=2,
            random_state=42
        )
        importances = result.importances_mean
        feature_names = [f'f{i}' for i in range(X_test.shape[1])]
        indices = np.argsort(importances)[::-1][:20]
        print("Top 20 Feature Importances (Keras, permutation):")
        for idx in indices:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        # Save top 20 feature importances to CSV
        import pandas as pd
        fi_df = pd.DataFrame({
            'feature': [feature_names[idx] for idx in indices],
            'importance': [importances[idx] for idx in indices]
        })
        fi_df.to_csv('model_results/keras/feature_importance.csv', index=False)
        print("Top 20 feature importances saved to model_results/keras/feature_importance.csv")

if __name__ == "__main__":
    main()
