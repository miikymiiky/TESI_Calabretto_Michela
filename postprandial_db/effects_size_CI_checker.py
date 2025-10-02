#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:19:16 2025

@author: michela
"""

import pandas as pd

try:
    # --- Step 1: Load your dataset ---
    df = pd.read_csv("all_postprandial_data_with_features.csv")
    print("Successfully loaded the dataset.")
except FileNotFoundError:
    print("Error: Could not find 'all_postprandial_data_with_features.csv'.")
    print("Please make sure this script is in the same folder as your data file.")
    exit()

target_column = 'target_hypoglycemia'

# --- Step 2: Define a list of features to check ---
features_to_check = [
    'mean_glucose_24h_pre_meal', # An ANOVA feature
    'hr_mean_15min_post_peak',     # An ANOVA feature
    'time_to_first_glucose_peak_after_meal', # A Kruskal-Wallis feature
    'rate_of_increase'             # An ANOVA feature
]

print("\n--- Running Diagnostic Check ---")
print("Checking how many groups remain for each feature AFTER applying .dropna()\n")

for feature in features_to_check:
    if feature not in df.columns:
        print(f"Feature: '{feature}'\n  - STATUS: Not found. Skipping.\n" + "-"*30)
        continue

    # --- Step 3: Replicate the exact data preparation from your script ---
    temp_df = df[[feature, target_column]].dropna()

    # --- Step 4: Count the number of unique groups that remain ---
    num_groups_remaining = temp_df[target_column].nunique()

    print(f"Feature: '{feature}'")
    print(f"  - Number of unique target groups REMAINING after .dropna(): {num_groups_remaining}")

    if num_groups_remaining < 3:
        print("  - DIAGNOSIS: A CI is not calculated because fewer than 3 groups have complete data for this feature.")
    else:
        print("  - DIAGNOSIS: 3 groups remain. The CI calculation should be possible.")
    print("-" * 30)