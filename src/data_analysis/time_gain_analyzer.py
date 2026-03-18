#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:32:46 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
import random
import json
import warnings

# Imports for ML Models and Evaluation
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#from imblearn.over_sampling import SMOTE
#from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

class TimeGainAnalyzer:
    """
    A class to handle training and evaluation of models, segmented by prediction time gain.
    This class mirrors the structure and logic of ModelTrainer for consistency.
    """
    def __init__(self, data_file_path: str, features_file_path: str, analysis_type: str,
                 classification_target: str, regression_target: str,
                 test_size: float = 0.2, random_state: int = 42, n_splits: int = 5,
                 match_models_to_sets: bool = False, use_model_specific_sets: bool = False):
        """Initializes the TimeGainAnalyzer."""
        self.data_file_path = data_file_path
        self.features_file_path = features_file_path
        self.analysis_type = analysis_type
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        
        self.classification_target = classification_target
        self.regression_target = regression_target

        if 'classification' in self.analysis_type:
            self.target_column = classification_target
        else:
            raise ValueError("analysis_type must be 'classification'")

        self.df = self._load_data()
        self.df = self._apply_pre_split_filters(self.df)
        self.feature_sets = self._load_feature_sets()
        self.match_models_to_sets = match_models_to_sets
        self.use_model_specific_sets = use_model_specific_sets

    def _load_data(self):
        df = pd.read_csv(self.data_file_path)
        for col in ['meal_time_start', 'meal_time_end', 'postprandial_hypo_time', 'postprandial_deep_hypo_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def _load_feature_sets(self):
        try:
            with open(self.features_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Features file not found for Time Gain Analysis at '{self.features_file_path}'.")
            return {}

    def _apply_pre_split_filters(self, df):
        """Applies the standard project filters before data splitting."""
        initial_rows = len(df)
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
        print(f"Filtered dataset from {initial_rows} to {len(df)} rows.")
        return df

    def _split_data_by_subject(self):
        """Splits data into training and test sets by subject."""
        meal_counts = self.df['subj_id'].value_counts()
        subjects = meal_counts.index.tolist()
        random.Random(self.random_state).shuffle(subjects)
        target_test_rows = int(len(self.df) * self.test_size)
        test_subjects = []
        current_test_rows = 0
        for subject in subjects:
            subject_rows = meal_counts[subject]
            if abs(target_test_rows - (current_test_rows + subject_rows)) < abs(target_test_rows - current_test_rows):
                 test_subjects.append(subject)
                 current_test_rows += subject_rows
        train_subjects = [s for s in self.df['subj_id'].unique() if s not in test_subjects]
        self.train_df = self.df[self.df['subj_id'].isin(train_subjects)].copy()
        self.test_df = self.df[self.df['subj_id'].isin(test_subjects)].copy()

    def _get_models(self):
        """Returns a dictionary of models based on analysis type."""
        models = {}
        if 'classification' in self.analysis_type:
            models["LogisticRegression"] = LogisticRegression(random_state=self.random_state, max_iter=10000, class_weight='balanced')
            models["SVC"] = SVC(random_state=self.random_state, class_weight='balanced', probability=True)
            models["RandomForest"] = RandomForestClassifier(random_state=self.random_state, class_weight='balanced_subsample', n_jobs=-1)
            models["LightGBM"] = LGBMClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1, verbose=-1)
            if self.analysis_type == 'binary_classification':
                imbalance_ratio = 5.9 # From model_trainer
                models["XGBoost"] = XGBClassifier(random_state=self.random_state, eval_metric='logloss', n_jobs=-1, scale_pos_weight=imbalance_ratio)
            else: # multi-class
                models["XGBoost"] = XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', n_jobs=-1)
        return models
    
    def _evaluate_segmented_performance(self, y_true, y_pred, y_pred_proba):
        """Prints performance metrics for a specific time gain segment."""
        if y_true.empty:
            print("  No samples in this segment.")
            return

        # Dynamically set labels and target names based on analysis type
        if 'multi_class' in self.analysis_type:
            labels = [0, 1, 2]
            target_names = ['No Hypo', 'Hypo', 'Deep Hypo']
        else: # binary
            labels = [0, 1]
            target_names = ['No Hypo', 'Hypo']

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("  Confusion Matrix:")
        if 'multi_class' in self.analysis_type:
            print(f"                     Pred_No_Hypo   Pred_Hypo   Pred_Deep_Hypo")
            for i, label in enumerate(target_names):
                print(f"  Actual_{label:<12}  {cm[i]}")
        else: # binary
            print(f"           Predicted 0   Predicted 1")
            print(f"  Actual 0 [[{cm[0][0]:^11d} {cm[0][1]:^11d}]] (No Hypo)")
            print(f"  Actual 1 [[{cm[1][0]:^11d} {cm[1][1]:^11d}]] (Hypo)")
        print("-" * 50)
        
        print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
        
        if len(np.unique(y_true)) > 1 and y_pred_proba is not None:
            try:
                if self.analysis_type == 'binary_classification':
                    # For binary, use the probability of the positive class (column 1)
                    roc_auc = roc_auc_score(y_true, y_pred_proba[1])
                    print(f"  AUC-ROC: {roc_auc:.4f}")
                else: # multi-class
                    # For multi-class, use the full probability matrix
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    print(f"  AUC-ROC (OvR, Weighted): {roc_auc:.4f}")
            except Exception as e:
                print(f"  Could not calculate AUC-ROC: {e}")


    def run_analysis(self, use_smote=False):
        """Main orchestration method mirroring ModelTrainer."""
        if self.df.empty or not self.feature_sets:
            print("Cannot start Time Gain analysis. Data or feature sets are missing.")
            return

        print("\n--- Starting Time Gain Performance Analysis Pipeline ---")
        self._split_data_by_subject()
        
        models_to_train = self._get_models()

        for feature_set_name, feature_list in self.feature_sets.items():
            print(f"\n{'='*25} Feature Set: '{feature_set_name}' ({len(feature_list)} features) {'='*25}")
            
            # --- Data preparation block identical to model_trainer.py ---
            X_train_initial = self.train_df[feature_list]
            y_train_initial = self.train_df[self.target_column]
            if self.analysis_type == 'binary_classification':
                y_train_initial = y_train_initial.apply(lambda x: 1 if x > 0 else 0)
            
            train_full = pd.concat([X_train_initial, y_train_initial, self.train_df['subj_id']], axis=1)
            if not self.use_model_specific_sets:
                train_full.dropna(inplace=True)
            
            X_train = train_full[feature_list]
            y_train = train_full[y_train_initial.name]
            
            # Prepare test set, including all necessary columns for time_gain calculation
            test_full_with_times = self.test_df.copy()
            X_test_initial = test_full_with_times[feature_list]
            y_test_initial = test_full_with_times[self.target_column]
            if self.analysis_type == 'binary_classification':
                y_test_initial = y_test_initial.apply(lambda x: 1 if x > 0 else 0)

            # Combine to drop NaNs, including all required columns
            test_full_combined = pd.concat([
                X_test_initial, y_test_initial, 
                test_full_with_times['subj_id'], 
                test_full_with_times['meal_time_start'], 
                test_full_with_times['time_to_first_glucose_peak_after_meal'],
                test_full_with_times['postprandial_hypo_time'],
                test_full_with_times['postprandial_deep_hypo_time']
            ], axis=1)
            
            if not self.use_model_specific_sets:
                test_full_combined.dropna(inplace=True)

            X_test = test_full_combined[feature_list]
            y_test = test_full_combined[y_test_initial.name]

            if X_train.empty or X_test.empty:
                print("Not enough data after dropping NaNs. Skipping feature set.")
                continue

            for model_name, model in models_to_train.items():
                if self.match_models_to_sets and model_name not in feature_set_name:
                    continue

                print(f"\n>>> Analyzing Model: {model_name}...")

                pipeline_steps = []
                if self.use_model_specific_sets:
                    pipeline_steps.append(('imputer', SimpleImputer(strategy='median')))
                pipeline_steps.append(('scaler', StandardScaler()))
                if use_smote and 'classification' in self.analysis_type:
                    pipeline_steps.append(('sampler', SMOTE(random_state=self.random_state)))
                pipeline_steps.append(('model', model))
                
                pipeline = ImbPipeline(pipeline_steps)
                pipeline.fit(X_train, y_train)

                y_pred_series = pd.Series(pipeline.predict(X_test), index=X_test.index)
                y_pred_proba_df = pd.DataFrame(pipeline.predict_proba(X_test), index=X_test.index)
                
                # --- CORRECTED Time Gain Calculation and Segmentation ---
                eval_df = test_full_combined.copy()
                eval_df['time_gain'] = np.nan
                
                true_hypo_mask = eval_df[y_test.name] > 0
                
                # Calculate time gain ONLY for true hypo events
                if true_hypo_mask.any():
                    prediction_time = eval_df.loc[true_hypo_mask, 'meal_time_start'] + pd.to_timedelta(eval_df.loc[true_hypo_mask, 'time_to_first_glucose_peak_after_meal'] + 15, unit='m')
                    
                    if 'multi_class' in self.analysis_type:
                         event_time = pd.to_datetime(np.where(
                            eval_df.loc[true_hypo_mask, y_test.name] == 2,
                            eval_df.loc[true_hypo_mask, 'postprandial_deep_hypo_time'],
                            eval_df.loc[true_hypo_mask, 'postprandial_hypo_time']
                        ))
                    else: # binary
                        event_time = eval_df.loc[true_hypo_mask, 'postprandial_hypo_time']

                    time_gain_values = (event_time - prediction_time).dt.total_seconds() / 60
                    eval_df.loc[true_hypo_mask, 'time_gain'] = time_gain_values

                # Define masks for three EXCLUSIVE segments
                no_hypo_indices = eval_df.index[eval_df[y_test.name] == 0]
                short_gain_indices = eval_df.index[(eval_df[y_test.name] > 0) & (eval_df['time_gain'] <= 15)]
                long_gain_indices = eval_df.index[(eval_df[y_test.name] > 0) & (eval_df['time_gain'] > 15)]
                
                # Verify that the sum equals the total test set size
                assert len(no_hypo_indices) + len(short_gain_indices) + len(long_gain_indices) == len(X_test), "Segmentation error: sample counts do not match!"

                # Evaluate and print results for each segment
                print(f"\n--- Performance for 'No Hypo' Events ({len(no_hypo_indices)} samples) ---")
                self._evaluate_segmented_performance(
                    y_test.loc[no_hypo_indices], 
                    y_pred_series.loc[no_hypo_indices], 
                    y_pred_proba_df.loc[no_hypo_indices]
                )

                print(f"\n--- Performance for Hypo Events with Time Gain <= 15 minutes ({len(short_gain_indices)} samples) ---")
                self._evaluate_segmented_performance(
                    y_test.loc[short_gain_indices], 
                    y_pred_series.loc[short_gain_indices], 
                    y_pred_proba_df.loc[short_gain_indices]
                )

                print(f"\n--- Performance for Hypo Events with Time Gain > 15 minutes ({len(long_gain_indices)} samples) ---")
                self._evaluate_segmented_performance(
                    y_test.loc[long_gain_indices], 
                    y_pred_series.loc[long_gain_indices],
                    y_pred_proba_df.loc[long_gain_indices]
                )