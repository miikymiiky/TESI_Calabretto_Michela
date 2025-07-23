# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:54:56 2025

@author: prend
"""

# main.py
from matplotlib import pyplot as plt
import os
from src.data_preparation.dataset_preparer import DatasetPreparer
from src.feature_extraction.postprandial_feature_extractor import PostprandialFeatureExtractor
from src.data_visualization.data_visualizer import DataVisualizer
from src.data_analysis.feature_analyzer import FeatureAnalyzer

plt.close('all')

def run_pipeline(run_preparation=True, run_feature_extraction=True,run_visualization=True, 
                 visualize_preparation_steps=False, run_data_analysis = False):
    
    """
    Pipeline di elaborazione dei dati.

    Args:
        run_preparation (bool): Se True, preparazione del dataset.
        run_feature_extraction (bool): Se True, estrazione delle feature.
        run_visualization (bool): Se True, visualizzatore dati.
        visualize_preparation_steps (bool): Se True, mostra i plot di debug per ogni soggetto.
    """

    # --- Parametri Comuni ---
    # Da migliorare: i parametri comuni dovrebbero essere passati in modo flessibile tipo:
    # caricarli da un file di configurazione (es. config.ini, config.yaml)
    COMMON_PARAMS = {
        "data_dir": "data",
        "data_dir_db": "postprandial_db",
    }

    # --- Parametri Specifici per DatasetPreparer ---
    PREPARER_PARAMS = {
        **COMMON_PARAMS,
        "sampling_rate_seconds": 300,  # 5 minuti in secondi
        "cutoff_frequency_hz": 2 * 1.67e-4,  # Frequenza di taglio del filtro (da ottimizzare)
        "filter_order": 1,  # Ordine del filtro Butterworth (da ottimizzare)
        "derivative_threshold": 0.1,  # Soglia derivata prima (da ottimizzare)
        "second_derivative_threshold": 0.01, # Soglia derivata seconda (da ottimizzare)
        "search_window_before_meal": 65,  # 60 minuti prima del pasto identificato (da ottimizzare)
        "search_window_after_meal": 30, # 30 minuti dopo il pasto
        "postprandial_duration_hours": 4, # max durata periodo postpradiale
        "wearable_variable_names": ['BATTERY_STATUS','HEART_RATE', 'HEART_RATE_VARIABILITY', 
                                   'PULSE_OX', 'RESPIRATION', 'STEPS', 'ZERO_CROSSING', 
                                   'ZERO_CROSSING_ENERGYTOTAL','ZERO_CROSSING_TIMEELAPSED'],
        "output_filename": 'all_postprandial_data.csv' # Nome del file output per la preparazione
    }

    # --- Parametri Specifici per PostprandialFeatureExtractor ---
    EXTRACTOR_PARAMS = {
        **COMMON_PARAMS,
        "postprandial_data_file": PREPARER_PARAMS["output_filename"], # Input è l'output del preparer
        "hypo_threshold": 70,  # mg/dL
        "deep_hypo_threshold": 54, # mg/dL
        "min_consecutive_samples_hypo": 3, # 3 campioni a 5 minuti = 15 minuti
        "peak_search_min_distance_samples": 6, # 6 campioni a 5 minuti = 30 minuti (distanza minima tra due picchi consecutivi)
        "peak_search_prominence": 15, # mg/dL (per evitare di considerare il rumore)
        "output_filename_with_features": 'all_postprandial_data_with_features.csv' # Output dell'estrattore
    }

    # --- Parametri Specifici per DataVisualizer ---
    VISUALIZER_PARAMS = {
        **COMMON_PARAMS,
        "postprandial_data_file": EXTRACTOR_PARAMS["output_filename_with_features"] # Input è l'output dell'estrattore
    }
    
    # --- Parametri Specifici per DataVisualizer ---
    ANALIZER_PARAMS = {
        "data_file_path":os.path.join(COMMON_PARAMS["data_dir_db"],EXTRACTOR_PARAMS["output_filename_with_features"]), 
        "target_column":'target_hypoglycemia',
    }

    if run_preparation:
        print("\n--- Fase 1: Preparazione del Dataset ---")
        preparer = DatasetPreparer(**PREPARER_PARAMS) # creates an instance of the class DatasetPreparer using the parameter defined in the dictionary
        prepared_df = preparer.prepare_dataset( # calls the method .prepare_dataset() on the DatasetPreparer object
            visualize_each_subject=visualize_preparation_steps
        )
        if prepared_df.empty:
            print("La preparazione del dataset non ha prodotto dati. Interruzione pipeline.")
            return

    if run_feature_extraction:
        print("\n--- Fase 2: Estrazione delle Feature Post-prandiali ---")
        extractor = PostprandialFeatureExtractor(**EXTRACTOR_PARAMS)
        features_df = extractor.extract_features()
        
        if features_df.empty:
            print("L'estrazione delle feature non ha prodotto dati. Interruzione pipeline.")
            return

    if run_visualization:
        print("\n--- Fase 3: Visualizzazione dei Dati ---")
        visualizer = DataVisualizer(**VISUALIZER_PARAMS)
        visualizer.run_interactive_visualizer()
    
    if run_data_analysis:
        print("\n--- Fase 3: Analisi Esplorativa delle Feature ---")
        analyzer = FeatureAnalyzer(**ANALIZER_PARAMS)
        
        #1. Filtrs data for the analysis
        filter_conditions = [
            {'column': 'hypo_to_check_flag', 'operator': '!=', 'value': 0},
            {'column': 'postprandial_duration_minutes', 'operator': '>', 'value': 120},
            {'column': 'time_peak_hypo_event', 'operator': '>', 'value': -1}, # -1 è il flag che consente di selezionare anche i pasti che non vanno in ipo 
            {'column': 'rate_of_increase', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_before_peak1', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_after_peak1', 'operator': '<', 'value': 0},
            {'column': 'has_heart_rate_data', 'operator': '==', 'value': 1}, # ADDED FILTER TO KEEP ONLY SUBJECTS WITH HEART_RATE DATA
            {'column': 'has_pulse_ox_data', 'operator': '==', 'value': 1}, # Filters for the other wearable data
            {'column': 'has_steps_data', 'operator': '==', 'value': 1},
            ]
        analyzer.filter_data(filter_conditions)
        # debug
        #a = analyzer.df;
        #b = a.where((a['target_hypoglycemia'] == 2))
        #c = b.where(b['glucose_derivative_15min_after_peak1'] > 0)
        
        # 2. Show basic info and missing values
        analyzer.display_basic_info()
        
        # 3. Plotcorrelations divided by logic groups
        print("\n--- Generating Grouped Correlation Heatmaps ---")
        
        # Generates a heatmap with all the features
        analyzer.plot_correlations(
            output_path='correlation_heatmap_full.png'
        )
        
        # Original heatmap
        all_analyzable_features = analyzer._get_features_for_analysis()
        
        # Define the list of features to exclude
        postprandial_exclusions = [
            'postprandial_mean_glucose',
            'postprandial_std_glucose',
            'postprandial_range',
            'postprandial_tir_target',
            'postprandial_tir_l1_hyper',
            'postprandial_tir_l2_hyper',
            'postprandial_tir_l1_hypo',
            'postprandial_tir_l2_hypo',
        ]
        
        # Create the feature group 
        original_features_group = [
            f for f in all_analyzable_features
            if 'hr_' not in f and 'cv_' not in f  and 'pulse_ox_' not in f and f not in postprandial_exclusions
        ]
       
        analyzer.plot_focused_correlation(
            original_features_group,
            title='Original Glucose-Based Features Hetmap',
            output_path='correlation_heatmap_original_features.png',
            annotate_cells=False
        )


        # Group 1: Postprandial Response Summary
        postprandial_group = [
            'postprandial_mean_glucose', 'postprandial_std_glucose', 'rate_of_increase',
            'first_glucose_peak_value_after_meal', 'time_to_first_glucose_peak_after_meal', 'postprandial_tir_l1_hypo',
            'hr_mean', 'hr_std', 'hr_slope', 'hr_variability'
        ]
        analyzer.plot_focused_correlation(
            postprandial_group,
            title='Focused Heatmap: Postprandial Response (Glucose vs. HR)',
            output_path='correlation_heatmap_postprandial.png',
            annotate_cells=True
        )

        # Group 2: Pre-Meal Glucose State (1h, 6h, 24h)
        pre_meal_glucose_group = [
            'mean_glucose_1h_pre_meal', 'cv_glucose_1h_pre_meal', 'num_positive_peaks_1h_pre_meal',
            'mean_glucose_6h_pre_meal', 'cv_glucose_6h_pre_meal', 'num_positive_peaks_6h_pre_meal',
            'mean_glucose_24h_pre_meal', 'cv_glucose_24h_pre_meal', 'num_positive_peaks_24h_pre_meal'
        ]
        analyzer.plot_focused_correlation(
            pre_meal_glucose_group,
            title='Focused Heatmap: Pre-Meal Glucose State',
            output_path='correlation_heatmap_pre_meal_glucose.png',
            annotate_cells=True
        )
        
        # Group 3: Pre-Meal Heart Rate State (1h, 6h, 24h)
        pre_meal_hr_group = [
            'hr_mean_1h_pre_meal', 'hr_std_1h_pre_meal', 'hr_slope_1h_pre_meal', 'hr_variability_1h_pre_meal',
            'hr_mean_6h_pre_meal', 'hr_std_6h_pre_meal', 'hr_slope_6h_pre_meal', 'hr_variability_6h_pre_meal',
            'hr_mean_24h_pre_meal', 'hr_std_24h_pre_meal', 'hr_slope_24h_pre_meal', 'hr_variability_24h_pre_meal'
        ]
        analyzer.plot_focused_correlation(
            pre_meal_hr_group,
            title='Focused Heatmap: Pre-Meal Heart Rate State',
            output_path='correlation_heatmap_pre_meal_hr.png',
            annotate_cells=True
        )
        
        # Group 4: Pulse Ox Summary
        pulse_ox_group = [
            'pulse_ox_mean', 'pulse_ox_std', 'pulse_ox_range',
            'pulse_ox_mean_1h_pre_meal', 'pulse_ox_std_1h_pre_meal',
            'pulse_ox_mean_6h_pre_meal', 'pulse_ox_std_6h_pre_meal'
        ]
        analyzer.plot_focused_correlation(
            pulse_ox_group,
            title='Focused Heatmap: Pulse Ox Summary',
            output_path='correlation_heatmap_pulse_ox.png'
        )
        
        # Group 5: Pre-Meal Pulse Ox State
        pre_meal_pulse_ox_group = [
            'pulse_ox_mean_1h_pre_meal', 'pulse_ox_std_1h_pre_meal', 'pulse_ox_range_1h_pre_meal',
            'pulse_ox_mean_3h_pre_meal', 'pulse_ox_std_3h_pre_meal', 'pulse_ox_range_3h_pre_meal',
            'pulse_ox_mean_6h_pre_meal', 'pulse_ox_std_6h_pre_meal', 'pulse_ox_range_6h_pre_meal',
            'pulse_ox_mean_12h_pre_meal', 'pulse_ox_std_12h_pre_meal', 'pulse_ox_range_12h_pre_meal',
            'pulse_ox_mean_24h_pre_meal', 'pulse_ox_std_24h_pre_meal', 'pulse_ox_range_24h_pre_meal'
        ]
        analyzer.plot_focused_correlation(
            pre_meal_pulse_ox_group,
            title='Focused Heatmap: Pre-Meal Pulse Ox State',
            output_path='correlation_heatmap_pre_meal_pulse_ox.png'
        )
        
        # Group 6: Pre-Meal Physiological State (1-Hour Window)
        pre_meal_state_1h_group = [
            'mean_glucose_1h_pre_meal', 'std_glucose_1h_pre_meal',
            'hr_mean_1h_pre_meal', 'hr_variability_1h_pre_meal',
            'pulse_ox_mean_1h_pre_meal', 'pulse_ox_std_1h_pre_meal'
        ]
        analyzer.plot_focused_correlation(
            pre_meal_state_1h_group,
            title='Focused Heatmap: Pre-Meal Physiological State (1-Hour Window)',
            output_path='correlation_heatmap_pre_meal_1h_state.png'
        )

        # Group 7: Predictive Features (Pre- vs. Post-Meal) 
        predictive_group = [
            'mean_glucose_1h_pre_meal', 'hr_variability_1h_pre_meal', 'pulse_ox_mean_1h_pre_meal', # Pre-meal state
            'postprandial_tir_l1_hypo', 'first_glucose_peak_value_after_meal', 'postprandial_mean_glucose', 'hr_slope' # Post-meal outcome
        ]
        analyzer.plot_focused_correlation(
            predictive_group,
            title='Focused Heatmap: Predictive Features (Pre-Meal vs. Post-Meal Outcome)',
            output_path='correlation_heatmap_predictive.png'
        )
        
        # Group 8: Core Metrics Summary 
        core_metrics_group = [
            'meal_hour',
            'mean_glucose_1h_pre_meal', 'std_glucose_1h_pre_meal', 'hr_variability_1h_pre_meal', 'pulse_ox_mean_1h_pre_meal',
            'rate_of_increase', 'first_glucose_peak_value_after_meal',
            'postprandial_mean_glucose', 
            'postprandial_tir_l1_hypo', 'postprandial_tir_l2_hypo',
            'hr_slope', 'hr_variability', 'pulse_ox_mean', 'pulse_ox_range',
            'daily_step_target_reached',
            'hr_prop_time_active_1h_pre_meal',
        ]
        analyzer.plot_focused_correlation(
            core_metrics_group,
            title='Core Metrics Correlation Heatmap',
            output_path='correlation_heatmap_core_metrics.png'
        )
        
        # Group 9: New Hypoglycemia Predictors group
        hypoglycemia_predictors_group = [
            'daily_step_target_reached',
            'hr_prop_time_active_1h_pre_meal',
            'mean_glucose_1h_pre_meal',
            'std_glucose_1h_pre_meal',
            'hr_variability_1h_pre_meal',
            'postprandial_tir_l1_hypo',
            'postprandial_tir_l2_hypo',
            'postprandial_hypo_duration_minutes'
        ]
        analyzer.plot_focused_correlation(
            hypoglycemia_predictors_group,
            title='Focused Heatmap: Hypoglycemia Predictors',
            output_path='correlation_heatmap_hypoglycemia_predictors.png'
        )
        
        # Group 10: Activity & Pre-Meal State
        activity_pre_meal_group = [
            'daily_step_target_reached', 'hr_prop_time_active_1h_pre_meal', 'mean_glucose_1h_pre_meal',
            'std_glucose_1h_pre_meal', 'hr_variability_1h_pre_meal', 'pulse_ox_mean_1h_pre_meal'
        ]
        analyzer.plot_focused_correlation(
            activity_pre_meal_group,
            title='Focused Heatmap: Activity & Pre-Meal State',
            output_path='correlation_heatmap_activity_pre_meal.png'
        )


            
        # 4. Plot features distribution
        analyzer.plot_feature_distributions(num_cols_per_plot=5) # number of columns
        
        # 5. Plot features distribution by target
        analyzer.plot_distributions_by_target(num_cols_per_plot=3)
        
if __name__ == "__main__":
   
    run_pipeline(
        run_preparation = False, # Esegui la preparazione dei dati
        visualize_preparation_steps = False, # Imposta a True per vedere i plot durante la preparazione (può essere molto verboso)
        run_feature_extraction = False, # Esegui l'estrazione delle feature
        run_visualization = False, # Avvia il visualizzatore interattivo
        run_data_analysis = True, # Per il momento fa solo boxplot delle tre classi
    )