#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 18:05:47 2025

@author: michela
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # for IterativeImputere
from sklearn.impute import IterativeImputer # for IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# Model Imports
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from xgboost import XGBClassifier, XGBRegressor
#from lightgbm import LGBMClassifier, LGBMRegressor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")


class ModelSpecificFeatureSelector:
    """
    Performs model-based feature selection using SFS for multiple models and scoring metrics.
    The process identifies an optimal feature subset for each combination.
    """
    def __init__(self, data_file_path, stats_file_path, analysis_type, random_state=42, n_splits_cv=5):
        self.data_file_path = data_file_path
        self.stats_file_path = stats_file_path  
        self.analysis_type = analysis_type
        self.random_state = random_state
        self.n_splits_cv = n_splits_cv

        # Set target based on analysis type
        if 'classification' in self.analysis_type:
            self.target_column = 'target_hypoglycemia'
        else: # regression
            self.target_column = 'postprandial_glucose_min'



    def _get_models_and_scorers(self):
        """Returns dictionaries of models and scoring functions based on the analysis type."""
        models = {}
        scorers = {}

        if 'classification' in self.analysis_type:
            # Models configured for imbalanced data
            models = {
                "LogisticRegression": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "SVC": SVC(random_state=self.random_state, class_weight='balanced'),
                "RandomForest": RandomForestClassifier(random_state=self.random_state, class_weight='balanced_subsample', n_jobs=-1),
                "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric='logloss', n_jobs=-1), # scale_pos_weight is set dynamically
                "LightGBM": LGBMClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1, verbose=-1)
            }
            # Scoring metrics for classification
            scorers = {
                'roc_auc': 'roc_auc_ovr_weighted' if 'multi_class' in self.analysis_type else 'roc_auc',
                'recall': 'recall_weighted',
                'precision': 'precision_weighted',
                'f1_score': 'f1_weighted'
            }
        elif self.analysis_type == 'regression':
            models = {
                "Ridge": Ridge(random_state=self.random_state),
                "Lasso": Lasso(random_state=self.random_state, max_iter=5000),
                "SVR": SVR(),
                "RandomForest": RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                "XGBoost": XGBRegressor(random_state=self.random_state, n_jobs=-1),
                "LightGBM": LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1)
            }
            # Scoring metric for regression
            scorers = {'r2': 'r2'}
            
        return models, scorers



    def _drop_correlated_features(self, df, threshold=0.8):
        """Drops features with a correlation higher than the threshold."""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # np.ones(corr_matrix.shape) ---> creates a matrix of ones with the same shape
                                                                                         # np.triu(..., k=1) ---> keeps only the upper triangle of that matrix; k=1 to skip the diagonal itself
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print(f"Found and removed {len(to_drop)} features with correlation > {threshold}.")
        return df.drop(columns=to_drop)



    def _load_and_prepare_data(self):
        """
        Loads data and uses the pre-sorted list of features from statistical analysis
        to prepare the data for SFS.
        """
        # 1. Load the full feature dataset and the statistical results
        df_full_data = pd.read_csv(self.data_file_path)
        try:
            df_stats = pd.read_csv(self.stats_file_path)
        except FileNotFoundError:
            print(f"ERROR: Statistics file not found at '{self.stats_file_path}'. Please run the data analysis step first.")
            return pd.DataFrame(), pd.Series(), pd.Series()

        # 2. Get the pre-sorted list of valid predictive features
        ordered_feature_list = df_stats['Feature'].tolist()

        # Create a working copy of the dataframe to filter
        df = df_full_data.copy()

        # 3. Apply standard project filters to the rows
        print("--- Applying standard project filters... ---")
        filters = [
            {'column': 'hypo_to_check_flag', 'operator': '!=', 'value': 0},
            {'column': 'postprandial_duration_minutes', 'operator': '>', 'value': 120},
            {'column': 'time_peak_hypo_event', 'operator': '>', 'value': -1},
            {'column': 'rate_of_increase', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_before_peak1', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_after_peak1', 'operator': '<', 'value': 0},
            {'column': 'has_heart_rate_data', 'operator': '==', 'value': 1},
            {'column': 'has_pulse_ox_data', 'operator': '==', 'value': 1},
            {'column': 'has_steps_data', 'operator': '==', 'value': 1},
        ]
        for condition in filters:
            col, op, val = condition['column'], condition['operator'], condition['value']
            if col in df.columns:
                if op == '>': df = df[df[col] > val]
                elif op == '<': df = df[df[col] < val]
                elif op == '==': df = df[df[col] == val]
                elif op == '!=': df = df[df[col] != val]

        # 4. Handle target-specific filtering on the rows
        if self.analysis_type == 'regression':
            df = df[df[self.target_column] > 0].copy() # drop rows with invalid glucose values
        
        initial_rows = len(df)
        df.dropna(subset=[self.target_column], inplace=True) # drop rows where the target is NaN
        if initial_rows > len(df):
            print(f"Removed {initial_rows - len(df)} rows with missing target values.")
        
        # Get the original full feature list BEFORE de-correlation
        # This will be used to filter out leaky features from the stats list
        full_feature_list_from_df = df.columns.tolist()

        # 5. Create the initial feature matrix (X) using the sorted list
        valid_ordered_features = [f for f in ordered_feature_list if f in df.columns]

        # Define all leaky features and explicitly remove them 
        leaky_features_to_remove = [
            'activity_proportion_after_peak',
            'pulse_ox_mean', 'pulse_ox_std', 'pulse_ox_min', 'pulse_ox_max', 'pulse_ox_range'
        ]
        
        # Filter the ordered list to remove the leaky features
        cleaned_ordered_features = [f for f in valid_ordered_features if f not in leaky_features_to_remove]
        
        # Create the initial matrix from the cleaned list
        X_initial = df[cleaned_ordered_features]

        # 6. Drop highly correlated features, respecting the order of p-value importance
        X_decorrelated = self._drop_correlated_features(X_initial, threshold=0.8)
        
        # 7. Calculate and print NaN percentage for each candidate feature
        nan_percentages = X_decorrelated.isnull().sum() * 100 / len(X_decorrelated)
        nan_df = nan_percentages.reset_index()
        nan_df.columns = ['Feature', 'NaN %']
        nan_df = nan_df.sort_values(by='NaN %', ascending=False)

        print("\n" + "="*80)
        print(" " * 15 + "CANDIDATE FEATURES (POST-CORRELATION FILTER)")
        print(" " * 21 + f"Total Features: {len(nan_df)}")
        print("="*80)
        print(nan_df.to_string(index=False))

        # 8. Filter out features with more than 30% NaNs
        threshold = 30.0
        features_to_drop = nan_percentages[nan_percentages > threshold].index.tolist()
        X_final = X_decorrelated.drop(columns=features_to_drop)
        
        print(f"\n>>> Dropping {len(features_to_drop)} features with > {threshold}% missing values.")
        if features_to_drop:
            print("Dropped features:", ", ".join(features_to_drop))

        # 9. Print the final list of features going into SFS
        final_nan_percentages = X_final.isnull().sum() * 100 / len(X_final)
        final_nan_df = final_nan_percentages.reset_index()
        final_nan_df.columns = ['Feature', 'NaN %']
        final_nan_df = final_nan_df.sort_values(by='NaN %', ascending=False)

        print("\n" + "="*80)
        print(" " * 20 + "FINAL FEATURES FOR SFS (POST-NAN FILTER)")
        print(" " * 21 + f"Total Features: {len(final_nan_df)}")
        print("="*80)
        print(final_nan_df.to_string(index=False))
        print("="*80 + "\n")

        # 10. Get final X, y, and groups
        X = X_final
        y = df.loc[X.index, self.target_column]
        groups = df.loc[X.index, 'subj_id']

        # 11. Adjust target for classification type
        if self.analysis_type == 'binary_classification':
            y = (y > 0).astype(int)

        return X, y, groups



    def run_sfs_for_all_models(self, scorer_name, scorer_metric, output_path):
        """
        Main method to run the SFS process for all models for a SINGLE scoring metric.
        """
        print(f"\n{'='*25}\n Running SFS for Analysis: '{self.analysis_type.upper()}' | Metric: '{scorer_name.upper()}'\n{'='*25}")
        
        X, y, groups = self._load_and_prepare_data()
        models, _ = self._get_models_and_scorers() # We only need the models here

        final_feature_sets = {}

        for model_name, model in models.items():
            
            print(f"\n--- Running SFS for Model: '{model_name}' ---")

            # Dynamically set scale_pos_weight for XGBoost in binary classification
            if model_name == 'XGBoost' and self.analysis_type == 'binary_classification':
                count_neg, count_pos = y.value_counts().get(0, 0), y.value_counts().get(1, 1)
                if count_pos > 0:
                    model.scale_pos_weight = count_neg / count_pos
                    print(f"Set XGBoost scale_pos_weight to: {model.scale_pos_weight:.2f}")

            pipeline = Pipeline([
                # ('imputer', IterativeImputer(random_state=self.random_state)), # IterativeImputer to replace the NaNs with values predicted using all the other features as predictors and by training a Regression Model (i.e.: Ridge Regression)
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            cv = GroupKFold(n_splits=self.n_splits_cv)
            sfs = SFS(
                estimator=pipeline, 
                k_features=(1, X.shape[1]),
                forward=True, 
                floating=False,
                verbose=0, # verbose = 2 ---> to track the process, but too many prints it gets confusing
                scoring=scorer_metric, 
                cv=cv, 
                n_jobs=-1
            )

            sfs.fit(X, y, groups=groups)

            best_features = list(sfs.k_feature_names_) # extract the names of the best-selected features from the sfs object
            best_score = sfs.k_score_ # retrieve the best evaluation score achieved by the SFS process
            
            # The key is just the model name, as the file will be named by the metric
            final_feature_sets[model_name] = best_features

            print("\n" + "="*80)
            print(f"Results for '{model_name}' Optimized for '{scorer_name.upper()}'")
            print("="*80)
            print(f"Selected {len(best_features)} features with a score of {best_score:.4f}")
            print("Selected Features:")
            for i, feature in enumerate(best_features, 1):
                print(f"  {i:2d}. {feature}")
            print("="*80 + "\n")

            # Plotting
            fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
            plt.title(f'SFS Performance ({model_name})\nOptimizing for {scorer_name.upper()}')
            plt.ylabel(scorer_name.upper())
            plt.grid(True)
            plt.show()

        print(f"\n--- Saving feature sets optimized for '{scorer_name.upper()}' to: {output_path} ---")
        with open(output_path, 'w') as f:
            # Here, we save a simple dictionary: { "ModelName": ["feature1", ...], ... }
            # This is still compatible with model_trainer if you just want to run all models for one metric.
            # To make it fully compatible with the old structure, we can adjust the keys. 
            compatible_feature_sets = {}
            for model_name, features in final_feature_sets.items():
                key = f"{model_name}_Optimized_for_{scorer_name}"
                compatible_feature_sets[key] = features

            json.dump(compatible_feature_sets, f, indent=4)
        print("Save complete.")