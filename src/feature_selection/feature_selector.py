#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 20:05:35 2025

@author: michela
"""

import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



class FeatureSelector:
    """
    Performs model-based feature selection using Sequential Forward Selection (SFS)
    on a given set of initial feature lists.
    """
    def __init__(self, data_file_path, initial_features_path, analysis_type='classification', random_state=42, n_splits_cv=5):
        self.data_file_path = data_file_path
        self.initial_features_path = initial_features_path
        self.analysis_type = analysis_type
        self.random_state = random_state
        self.n_splits_cv = n_splits_cv

        if self.analysis_type == 'classification':
            self.target_column = 'target_hypoglycemia'
        else: # regression
            self.target_column = 'postprandial_glucose_min'



    def _load_and_filter_data(self):
        """Loads and applies standard project filters to the data."""
        df = pd.read_csv(self.data_file_path)

        # Filter for regression to drop rows with invalid glucose values
        if self.analysis_type == 'regression':
            initial_rows = len(df)
            df = df[df[self.target_column] > 0].copy()
            print(f"Removed {initial_rows - len(df)} row(s) with negative regression target values.")
        
        # Apply original filters to the dataset
        print("--- Applying standard project filters... ---")
        all_filters = [
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
        for condition in all_filters:
            col, op, val = condition['column'], condition['operator'], condition['value']
            if op == '>': df = df[df[col] > val]
            elif op == '<': df = df[df[col] < val]
            elif op == '==': df = df[df[col] == val]
            elif op == '!=': df = df[df[col] != val]
        
        return df



    def select_features(self, scoring_metric, output_path, plot_results=True):
        """
        Runs the SFS process for a specific scoring metric and saves the results.
        """
        base_df = self._load_and_filter_data()
        
        try:
            with open(self.initial_features_path, 'r') as f:
                initial_feature_sets = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Initial features file not found at {self.initial_features_path}")
            return

        final_feature_sets = {}

        for set_name, feature_list in initial_feature_sets.items():
            print(f"\n{'='*25}\n Running SFS for '{set_name}' (Optimizing for {scoring_metric.upper()})\n{'='*25}")
            
            analysis_df = base_df[feature_list + [self.target_column, 'subj_id']].dropna() # drop rows with NaNs
            
            y = analysis_df[self.target_column]
            if self.analysis_type == 'classification':
                # For both roc_auc and recall, we need a binary target
                y = (analysis_df[self.target_column] > 0).astype(int) # convert the target into a binary one

            X = analysis_df[feature_list]
            groups = analysis_df['subj_id']
            
            model = None
            if self.analysis_type == 'classification':
                # Calculate scale_pos_weight for class imbalance
                count_neg = (y == 0).sum()
                count_pos = (y == 1).sum()
                scale_pos_weight = count_neg / count_pos if count_pos > 0 else 1
                print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

                model = XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight
                )
            else: # regression
                model = XGBRegressor(
                    random_state=self.random_state,
                    n_jobs=-1
                )

            if len(X) < 50:
                print(f"Skipping '{set_name}' due to insufficient data ({len(X)} samples).")
                continue

            pipeline = Pipeline([(
                'scaler', StandardScaler()), 
                ('model', model)
            ])
            cv = GroupKFold(n_splits=self.n_splits_cv)
            sfs = SFS(
                estimator=pipeline, 
                k_features=(1, len(feature_list)), # the range of subset sizes
                forward=True, 
                floating=False, # once a feature is added, it stays; no remotion is allowed
                verbose=0, # verbose = 2 for a more detailed console output
                scoring=scoring_metric, 
                cv=cv, 
                n_jobs=-1
            )

            sfs.fit(X, y, groups=groups) # actual selection of the features subsets

            results_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T # .T ---> transpose
            best_k_idx = results_df['avg_score'].idxmax()
            best_features = list(results_df.loc[best_k_idx, 'feature_names'])
            
            final_feature_sets[set_name] = best_features
            
            print("\n" + "="*80)
            title_str = f"Final Selected Features for '{set_name}' Set"
            subtitle_str = f"Analysis: {self.analysis_type.capitalize()} | Optimized for: {scoring_metric.upper()}"
            print(f"{title_str.center(80)}")
            print(f"{subtitle_str.center(80)}")
            print("="*80)
            print(f"Total features selected: {len(best_features)} (with a score of {results_df.loc[best_k_idx, 'avg_score']:.4f})")
            print("-" * 80)
            for i, feature in enumerate(best_features, 1):
                print(f"{i:2d}. {feature}")
            print("="*80 + "\n")
            
            if plot_results:
                fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev') # kind='std_dev' ---> plots mean performance with a shaded area for ±1 standard deviation
                title = f'SFS Performance ({set_name} set) - {self.analysis_type.capitalize()}\nOptimizing for {scoring_metric.upper()}'
                plt.title(title)
                plt.ylabel(scoring_metric.upper())
                plt.grid(True)
                plt.show()

        print(f"\n--- Saving final selected feature sets to: {output_path} ---")
        with open(output_path, 'w') as f:
            json.dump(final_feature_sets, f, indent=4)
        print("Save complete.")
        
        
        
        
        
        