#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:41:01 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


# Adjust the path to go up one level to the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils4processing import interpolate_saturation_events


def visualize_interpolated_events(
    features_db_path: str,
    data_dir: str,
    glucose_filename_pattern: str,
    max_plots: int = 50  # Add a parameter to limit the plots
):
    """
    Finds and plots all meal events where saturation interpolation was applied.

    Args:
        features_db_path (str): Full path to the '...with_features.csv' file.
        data_dir (str): Path to the main data directory.
        glucose_filename_pattern (str): The filename pattern for processed glucose files.
        max_plots (int): The maximum number of plots to display.
    """
    try:
        df = pd.read_csv(features_db_path)
        for col in ['meal_time_start', 'meal_time_end']:
            df[col] = pd.to_datetime(df[col])
        print(f"Successfully loaded feature database from: {features_db_path}")
    except FileNotFoundError:
        print(f"ERROR: Feature database not found at '{features_db_path}'.")
        return

    interpolated_df = df[df['was_interpolated'] == 1].copy()
    total_found = len(interpolated_df)

    if total_found == 0:
        print("ℹNo interpolated events found in the database. Nothing to visualize.")
        return

    print(f"Found {total_found} interpolated meal events to visualize.")

    # Interactive prompt to handle a large number of plots
    plots_to_show = total_found
    if total_found > max_plots:
        user_input = input(f"Display all {total_found} plots? (This may take a while). "
                           f"Enter 'yes' to proceed or anything else to show only the first {max_plots}: ").lower()
        if user_input != 'yes':
            plots_to_show = max_plots
            print(f"Showing the first {max_plots} plots...")
    
    # Loop through each interpolated event and generate a plot
    for index, row in interpolated_df.head(plots_to_show).iterrows():
        subj_id = row['subj_id']
        meal_start = row['meal_time_start']
        meal_end = row['meal_time_end']

        glucose_file_path = os.path.join(data_dir, subj_id, glucose_filename_pattern.format(subj_id))
        try:
            glucose_df = pd.read_csv(glucose_file_path, index_col='Time', parse_dates=True)
        except FileNotFoundError:
            print(f"⚠️ Warning: Could not find glucose file for subject {subj_id}. Skipping.")
            continue

        plot_start = meal_start - pd.Timedelta(hours=1)
        plot_end = meal_end + pd.Timedelta(hours=1)
        
        glucose_series_raw = glucose_df['glucose'].loc[plot_start:plot_end]
        glucose_series_filtered = glucose_df['glucose_filtered'].loc[plot_start:plot_end]

        glucose_series_interpolated, was_changed = interpolate_saturation_events(
            glucose_series_to_interpolate=glucose_series_filtered,
            glucose_series_for_detection=glucose_series_raw
        )
       
        changed_points_mask = (glucose_series_filtered != glucose_series_interpolated)
        interpolated_segment = glucose_series_interpolated[changed_points_mask]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(glucose_series_filtered.index, glucose_series_filtered, 
                color='dodgerblue', marker='o', markersize=4, linestyle='-', 
                label='Original Filtered Glucose')

        ax.plot(glucose_series_interpolated.index, glucose_series_interpolated, 
                color='red', linestyle='--', linewidth=2,
                label='Interpolated Glucose')
        
        if not interpolated_segment.empty:
            ax.scatter(interpolated_segment.index, interpolated_segment.values,
                       color='red', s=80, zorder=5, label='Interpolated Points')

        ax.axvline(meal_start, color='green', linestyle='--', label=f'Meal Start ({meal_start.strftime("%H:%M")})')
        ax.axvline(meal_end, color='purple', linestyle=':', label=f'Meal End ({meal_end.strftime("%H:%M")})')

        ax.set_title(f"Interpolation Inspection\nSubject: {subj_id} | Meal Start: {meal_start.strftime('%Y-%m-%d %H:%M')}", fontsize=16)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Glucose (mg/dL)", fontsize=12)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, '..')
    
    DATABASE_DIR = os.path.join(project_root, "postprandial_db")
    DATA_DIR = os.path.join(project_root, "data")
    FEATURES_FILENAME = "all_postprandial_data_with_features.csv"
    GLUCOSE_FILENAME_PATTERN = "{}tma_processed_new.csv"

    full_db_path = os.path.join(DATABASE_DIR, FEATURES_FILENAME)

    # Run the Visualization
    visualize_interpolated_events(
        features_db_path=full_db_path,
        data_dir=DATA_DIR,
        glucose_filename_pattern=GLUCOSE_FILENAME_PATTERN,
        max_plots=50  # this default limit can be changed
    )