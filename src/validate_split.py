#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 22:03:54 2025

@author: michela
"""

import pandas as pd
import json
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DB_DIR = "../postprandial_db"
DATA_FILE = "all_postprandial_data_with_features.csv"
DATA_FILE_PATH = os.path.join(DB_DIR, DATA_FILE)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Select Analysis Type ---
ANALYSIS_TYPE = 'classification'  # Options: 'regression' or 'classification'

# Dynamically set the target and features file based on analysis type
if ANALYSIS_TYPE == 'classification':
    TARGET_COLUMN = 'target_hypoglycemia'
    FEATURES_FILE = "sfs_selected_features_classification_roc_auc.json"
    #FEATURES_FILE = "classification_selected_feature_sets.json"
else: # regression
    TARGET_COLUMN = 'postprandial_glucose_min' 
    FEATURES_FILE = "regression_selected_feature_sets.json" 

FEATURES_FILE_PATH = os.path.join(DB_DIR, FEATURES_FILE)


def plot_feature_comparison(train_data, test_data, feature_name, is_target=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    title = f'Target Distribution Comparison: {feature_name}' if is_target else f'Feature Distribution Comparison: {feature_name}'
    fig.suptitle(title, fontsize=16)

    # Plot for Training Set
    sns.histplot(
        train_data[feature_name], 
        kde=True, 
        ax=axes[0], 
        bins=30, 
        color='royalblue',
        stat="density", # Normalize by density
        common_norm=False # Normalize each plot independently
    )
    axes[0].set_title(f'Training Set (n={len(train_data)})')
    axes[0].set_ylabel("Density") # <--- CHANGE: Update Y-axis label

    # Plot for Test Set
    sns.histplot(
        test_data[feature_name], 
        kde=True, 
        ax=axes[1], 
        bins=30, 
        color='seagreen',
        stat="density", # Normalize by density
        common_norm=False # Normalize each plot independently
    )
    axes[1].set_title(f'Test Set (n={len(test_data)})')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

# 1. Load, Filter, and Split Data ONCE 
df = pd.read_csv(DATA_FILE_PATH)

if ANALYSIS_TYPE == 'regression':
    initial_rows = len(df)
    df = df[df[TARGET_COLUMN] > 0].copy()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Removed {rows_dropped} row(s) with negative target values.")

print("--- Applying Pre-Split Filters ---")
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

# Subject-based split
meal_counts = df['subj_id'].value_counts()
subjects = meal_counts.index.tolist()
random.Random(RANDOM_STATE).shuffle(subjects)
target_test_rows = int(len(df) * TEST_SIZE)
test_subjects = []
current_test_rows = 0
for subject in subjects:
    subject_rows = meal_counts[subject]
    if abs(target_test_rows - (current_test_rows + subject_rows)) < abs(target_test_rows - current_test_rows):
         test_subjects.append(subject)
         current_test_rows += subject_rows
train_subjects = [s for s in df['subj_id'].unique() if s not in test_subjects]
train_df_base = df[df['subj_id'].isin(train_subjects)].copy()
test_df_base = df[df['subj_id'].isin(test_subjects)].copy()

print(f"Base split complete: {len(train_df_base)} training rows, {len(test_df_base)} test rows BEFORE NaN drop.")

# 2. Loop Through Feature Sets to Perform Analysis 
try:
    with open(FEATURES_FILE_PATH, 'r') as f:
        feature_sets = json.load(f)

    for feature_set_name, feature_list in feature_sets.items():
        print(f"\n{'='*25}\nAnalyzing Feature Set: '{feature_set_name}'\n{'='*25}")

        # a. Create copies and select columns for this specific feature set
        train_df_set = train_df_base[feature_list + [TARGET_COLUMN]].copy()
        test_df_set = test_df_base[feature_list + [TARGET_COLUMN]].copy()

        # b. Drop NaNs for THIS FEATURE SET
        initial_train = len(train_df_set)
        initial_test = len(test_df_set)
        train_df_set.dropna(inplace=True)
        test_df_set.dropna(inplace=True)
        print(f"Dropped {initial_train - len(train_df_set)} rows from training set.")
        print(f"Dropped {initial_test - len(test_df_set)} rows from test set.")
        
        if train_df_set.empty or test_df_set.empty:
            print("Not enough data after dropping NaNs. Skipping analysis for this set.")
            continue

        # c. Analyze Target Distribution for THIS FEATURE SET
        print(f"\n--- Target Distribution for '{feature_set_name}' ---")
        if ANALYSIS_TYPE == 'classification':
            train_dist = train_df_set[TARGET_COLUMN].value_counts(normalize=True).sort_index()
            test_dist = test_df_set[TARGET_COLUMN].value_counts(normalize=True).sort_index()
            comparison_df = pd.DataFrame({
                'Training Set %': train_dist * 100,
                'Test Set %': test_dist * 100
            })
            print(comparison_df.round(2))
        else: # regression
            train_stats = train_df_set[TARGET_COLUMN].describe()
            test_stats = test_df_set[TARGET_COLUMN].describe()
            comparison_df = pd.DataFrame({'Training Set': train_stats, 'Test Set': test_stats})
            print(comparison_df.round(2))
            plot_feature_comparison(train_df_set, test_df_set, TARGET_COLUMN, is_target=True)

        # d. Plot Feature Distributions for THIS FEATURE SET
        print(f"\n--- Plotting Feature Distributions for '{feature_set_name}' ---")
        for feature in feature_list:
            plot_feature_comparison(train_df_set, test_df_set, feature)

except FileNotFoundError:
    print(f"ERROR: Features file not found at '{FEATURES_FILE_PATH}'.")
except Exception as e:
    print(f"An error occurred: {e}")