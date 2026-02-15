import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import mlflow
import argparse
import os

def compare_distributions(train_df, test_df, feature_cols):
    drift_results = {}
    for col in feature_cols:
        train_col = train_df[col].dropna()
        test_col = test_df[col].dropna()
        if len(train_col) > 0 and len(test_col) > 0:
            stat, p_value = ks_2samp(train_col, test_col)
            drift_results[col] = {'ks_stat': stat, 'p_value': p_value}
        else:
            drift_results[col] = {'ks_stat': np.nan, 'p_value': np.nan}
    return drift_results

def main():
    parser = argparse.ArgumentParser(description='Drift detection between train and new data.')
    parser.add_argument('--train', type=str, default='feature_store/train_features.parquet', help='Path to train features')
    parser.add_argument('--test', type=str, default='feature_store/test_features.parquet', help='Path to test/new features')
    parser.add_argument('--model_type', type=str, default='keras', help='Model type for MLflow tagging')
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    # Only compare numeric features present in both
    numeric_cols = list(set(train_df.select_dtypes(include=[np.number]).columns) & set(test_df.select_dtypes(include=[np.number]).columns))
    drift_results = compare_distributions(train_df, test_df, numeric_cols)

    mlflow.set_experiment('drift_detection')
    with mlflow.start_run():
        mlflow.set_tag('model_type', args.model_type)
        for col, res in drift_results.items():
            mlflow.log_metric(f'drift_{col}_ks_stat', res['ks_stat'] if res['ks_stat'] is not None else -1)
            mlflow.log_metric(f'drift_{col}_p_value', res['p_value'] if res['p_value'] is not None else -1)
        print('Drift detection complete. Results logged to MLflow.')
        print('Top drifted features:')
        sorted_cols = sorted(drift_results.items(), key=lambda x: x[1]['ks_stat'] if x[1]['ks_stat'] is not None else 0, reverse=True)
        for col, res in sorted_cols[:10]:
            print(f'{col}: KS={res["ks_stat"]:.4f}, p={res["p_value"]:.4g}')

if __name__ == '__main__':
    main()
