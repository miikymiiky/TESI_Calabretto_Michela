# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:54:56 2025

@author: prend
"""

# main.py
from matplotlib import pyplot as plt
import os

# This forces all child processes to ignore UserWarnings, fixing the issue with n_jobs=-1
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

from src.data_preparation.dataset_preparer import DatasetPreparer
from src.feature_extraction.postprandial_feature_extractor import PostprandialFeatureExtractor
from src.data_visualization.data_visualizer import DataVisualizer
from src.data_analysis.feature_analyzer import FeatureAnalyzer
from src.data_analysis.performance_analyzer import PerformanceAnalyzer 
from src.data_analysis.time_gain_analyzer import TimeGainAnalyzer
from src.feature_selection.feature_selector import FeatureSelector
from src.feature_selection.model_specific_feature_selector import ModelSpecificFeatureSelector
from src.training.model_trainer import ModelTrainer
from src.training.hybrid_model_trainer import HybridModelTrainer
from src.training.two_step_model_trainer import TwoStepModelTrainer
from src.utils.utils4processing import drop_correlated_features 
import pandas as pd
import json


plt.close('all')


def run_pipeline(run_preparation=True, run_feature_extraction=True,run_visualization=True, 
                 visualize_preparation_steps=False, 
                 run_data_analysis = False, 
                 run_feature_selection=False,
                 run_model_specific_sfs=False,
                 display_sfs_results=False,
                 run_model_training=False,
                 run_performance_analysis=False,
                 run_time_gain_analysis=False,
                 RELOAD_STATS_FROM_CSV=True, # Flag for the effect size and CIs computation ---> True: loads the saved values from the csv files; False: runs the entire extrapolation process (bootstraping, ecc...) 
                 USE_SMOTE_SAMPLING=True): # Flag for the conditional SMOTE step in the models training
    
    """
    Pipeline di elaborazione dei dati.

    Args:
        run_preparation (bool): Se True, preparazione del dataset.
        run_feature_extraction (bool): Se True, estrazione delle feature.
        run_visualization (bool): Se True, visualizzatore dati.
        visualize_preparation_steps (bool): Se True, mostra i plot di debug per ogni soggetto.
        run_data_analysis (bool): Se True, esegue l'analisi statistica.
        RELOAD_STATS_FROM_CSV (bool): Se True, carica i risultati statistici da CSV invece di ricalcolarli.
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
    # Switch to easily change between analysis modes
    ANALYSIS_MODE = 'classification'  # Options: 'classification' or 'regression' or 'hybrid' or 'two_step' ('hybrid' and 'two_step' are only for the run_model_training_pipeline)

    ANALIZER_PARAMS = {
        "data_file_path":os.path.join(COMMON_PARAMS["data_dir_db"],EXTRACTOR_PARAMS["output_filename_with_features"]), 
        "data_dir_db": COMMON_PARAMS["data_dir_db"],
        "analysis_type": ANALYSIS_MODE,
        "classification_target": 'target_hypoglycemia',
        "regression_target": 'postprandial_glucose_min',
    }
    
    
    # --- Parametri Specifici per ModelTrainer ---
    # Switch for classification type
    CLASSIFICATION_TYPE = 'binary' # Options: 'binary' or 'multi_class

    
    # --- CHOOSE THE FEATURE SELECTION STRATEGY ---
    # 'STATISTICAL': Uses the sets from feature_analyzer.py (Pre-Meal, Mixed, etc.).
    # 'MODEL_SPECIFIC': Uses the sets from model_specific_feature_selector.py. 
    # 'GENERAL_SFS': Uses the single best set from feature_selector.py.
    FEATURE_SET_STRATEGY = 'MODEL_SPECIFIC'  # 'STATISTICAL' or 'MODEL_SPECIFIC' or 'GENERAL_SFS'
    
    # This is only used for the 'MODEL_SPECIFIC' strategy
    OPTIMIZATION_METRIC = 'f1_score' # 'roc_auc' or 'f1_score'

    # Determine the features filename and trainer flags based on the chosen strategy
    if FEATURE_SET_STRATEGY == 'MODEL_SPECIFIC':
        print(">>> Using MODEL-SPECIFIC feature sets.")
        match_models_flag = True
        if ANALYSIS_MODE == 'regression':
            trainer_analysis_type = 'regression'
            features_filename = "sfs_per_model_regression_r2.json"
        else: # Classification
            trainer_analysis_type = f'{CLASSIFICATION_TYPE}_classification'
            features_filename = f"sfs_per_model_{trainer_analysis_type}_{OPTIMIZATION_METRIC}.json"

    elif FEATURE_SET_STRATEGY == 'GENERAL_SFS':
        print(">>> Using GENERAL SFS (XGBoost-Optimized) feature sets.")
        match_models_flag = False  # All models will use the same file
        if ANALYSIS_MODE == 'regression':
            trainer_analysis_type = 'regression'
            features_filename = "sfs_selected_features_regression_r2.json"
        else: # Classification
            trainer_analysis_type = f'{CLASSIFICATION_TYPE}_classification'
            # Only use the one set optimized for roc_auc
            features_filename = "sfs_selected_features_classification_roc_auc.json"
            
    else: # Default to 'STATISTICAL'
        print(">>> Using STATISTICAL (Analyzer-generated) feature sets.")
        match_models_flag = False
        if ANALYSIS_MODE == 'regression':
            trainer_analysis_type = 'regression'
            features_filename = "regression_selected_feature_sets.json"
        else: # Covers classification, hybrid, and two_step
            trainer_analysis_type = f'{CLASSIFICATION_TYPE}_classification'
            features_filename = "classification_selected_feature_sets.json"



    TRAINER_PARAMS = {
        "data_file_path": os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
        "features_file_path": os.path.join(COMMON_PARAMS["data_dir_db"], features_filename),
        "analysis_type": trainer_analysis_type, 
        "classification_target": 'target_hypoglycemia',
        "regression_target": 'postprandial_glucose_min',
        "test_size": 0.20,
        "random_state": 42,
        "match_models_to_sets": match_models_flag,
        "use_model_specific_sets": FEATURE_SET_STRATEGY == 'MODEL_SPECIFIC'
    }
    
    
    # --- Parametri Specifici per Feature Selection and Display Feature Selection ---
    classification_stats_file = os.path.join(COMMON_PARAMS["data_dir_db"], "classification_omnibus_tests.csv")
    regression_stats_file = os.path.join(COMMON_PARAMS["data_dir_db"], "regression_feature_significance.csv")

    # These filter lists are needed by both the analysis run and the display run
    non_wearable_filters = [
        {'column': 'hypo_to_check_flag', 'operator': '!=', 'value': 0},
        {'column': 'postprandial_duration_minutes', 'operator': '>', 'value': 120},
        {'column': 'time_peak_hypo_event', 'operator': '>', 'value': -1},
        {'column': 'rate_of_increase', 'operator': '>', 'value': 0},
        {'column': 'glucose_derivative_15min_before_peak1', 'operator': '>', 'value': 0},
        {'column': 'glucose_derivative_15min_after_peak1', 'operator': '<', 'value': 0},
    ]
    wearable_filters = [
        {'column': 'has_heart_rate_data', 'operator': '==', 'value': 1},
        {'column': 'has_pulse_ox_data', 'operator': '==', 'value': 1},
        {'column': 'has_steps_data', 'operator': '==', 'value': 1},
    ]



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
        print(f"\n--- Fase 4: Analisi Esplorativa delle Feature ({ANALYSIS_MODE.capitalize()}) ---")
        analyzer = FeatureAnalyzer(**ANALIZER_PARAMS)
        
        
        # Switch to control the inclusion of meals with sensor saturation
        EXCLUDE_SATURATION_EVENTS = False

        if analyzer.df is not None and 'is_saturated_at_39' in analyzer.df.columns:
            total_meals = len(analyzer.df)
            saturation_event_count = analyzer.df['is_saturated_at_39'].sum()
            print(f"\nInitial meal events loaded for analysis: {total_meals}")
            print(f"Among these, {saturation_event_count} events include sensor saturation (glucose at 39 mg/dL).")

            if EXCLUDE_SATURATION_EVENTS:
                print(">>> EXCLUDING saturation events from the analysis.")
                analyzer.filter_data([{'column': 'is_saturated_at_39', 'operator': '==', 'value': 0}])
                print(f"Meal events remaining for analysis: {len(analyzer.df)}")
            else:
                print(">>> INCLUDING saturation events in the analysis.")
            
        
        
        print("\n--- Target Distribution (Before Any Filtering) ---")
        if analyzer.df is not None:
            print(analyzer.df['target_hypoglycemia'].value_counts(dropna=False))
        
        # 1. Apply filters IN STAGES
        # Apply the non-wearable filters
        print("\nApplying non-wearable filters...")
        analyzer.filter_data(non_wearable_filters)
        
        # Print the distribution *before* applying wearable filters
        print("\n--- Target Distribution (After Non-Wearable Filters) ---")
        if analyzer.df is not None:
            print(analyzer.df['target_hypoglycemia'].value_counts(dropna=False))
        
        # Apply the wearable filters
        print("\nApplying wearable filters...")
        analyzer.filter_data(wearable_filters)

        # Print the final distribution *after* applying wearable filters
        print("\n--- Target Distribution (After Wearable Filters) ---")
        if analyzer.df is not None:
            print(analyzer.df['target_hypoglycemia'].value_counts(dropna=False))
        
        

        # Print the final number of subjects
        if analyzer.df is not None and not analyzer.df.empty:
            final_subject_count = analyzer.df['subj_id'].nunique()
            print(f"\n--- Number of Subjects Remaining After All Filters ---")
            print(f"Total unique subjects for the analysis: {final_subject_count}")
            
 
        
        # 2. Show basic info and missing values
        analyzer.display_basic_info()
        
        
        # BLOCK to handle loading vs. calculating of effect sizes and CIs
        if RELOAD_STATS_FROM_CSV:
            print("\n>>> RELOADING pre-computed statistics from CSV files. Skipping calculations. <<<")
            if ANALYSIS_MODE == 'classification':
                try:
                    significant_features = pd.read_csv('classification_omnibus_tests.csv')
                    posthoc_results = pd.read_csv('classification_posthoc_tests.csv')
                    features_analyzed = significant_features['Feature'].tolist() 
                    print("Successfully loaded classification omnibus and post-hoc results.")
                except FileNotFoundError:
                    print("\nERROR: Could not find result CSVs. Please run with RELOAD_STATS_FROM_CSV=False first.")
                    return
            elif ANALYSIS_MODE == 'regression':
                try:
                    regression_results = pd.read_csv('regression_feature_significance.csv')
                    features_analyzed = regression_results['Feature'].tolist()
                    print("Successfully loaded regression results.")
                except FileNotFoundError:
                    print("\nERROR: Could not find result CSV. Please run with RELOAD_STATS_FROM_CSV=False first.")
                    return
        else:
            print("\n>>> RUNNING all statistical calculations. This may take some time... <<<")
            if ANALYSIS_MODE == 'classification':
                significant_features, features_analyzed = analyzer.run_classification_tests()
                if significant_features is not None and not significant_features.empty:
                    posthoc_results = analyzer.run_posthoc_tests(significant_features)
                else:
                    posthoc_results = pd.DataFrame() 
            elif ANALYSIS_MODE == 'regression':
                regression_results, features_analyzed = analyzer.display_regression_correlations(top_n=25)
       
        
        # This part of the script now uses the data that was either loaded or calculated above
        # Run analysis based on the selected mode
        if ANALYSIS_MODE == 'classification':
            print("\n--- Running Classification Analysis ---")
            # Box Plots
            #analyzer.plot_distributions_by_target(num_cols_per_plot=3)
            
            
            # Number of Features Analyzed
            if features_analyzed:
                print(f"\n--- Number of Features Analyzed for Prediction ---")
                print(f"Total predictive features analyzed: {len(features_analyzed)}")

            

            # If there are significant features, run the post-hoc tests
            if significant_features is not None and not significant_features.empty:
                
                print("\n--- Top 25 Most Significant Features for Classification (with Effect Size CIs) ---")
                print(significant_features.head(25))
                print("\n--- All Significant Features for Classification (Sorted by P-Value) ---")
                with pd.option_context('display.max_rows', None):
                    print(significant_features)
                
                print("\n--- All Significant Pairwise Comparisons (Ordered by Omnibus Test Significance) ---")
                with pd.option_context('display.max_rows', None):
                    print(posthoc_results)
            
                # --- Heatmap 1: Top 25 Most Significant Features ---
                print("\n--- Generating Correlation Heatmap for Top 25 Classification Features ---")
                top_class_features = significant_features['Feature'].head(25).tolist()
                analyzer.plot_focused_correlation(
                    feature_list=top_class_features,
                    title='Correlation of Top 25 Classification Features',
                    figsize=(20, 18),
                    annotate_cells=True,
                    output_path='heatmap_top25_classification.png'
                )

                # --- Heatmap 2: Top 35 Less-Correlated Features ---
                print("\n--- Generating Correlation Heatmap for Top 25 Less-Correlated Features ---")
                
                # Step 1: Get the wide-format pairwise results using our new helper
                pairwise_results_df = analyzer.get_pairwise_comparison_results(significant_features)

                if pairwise_results_df is not None and not pairwise_results_df.empty:
                    # Step 2: Find features where ALL pairwise p-values are < 0.05
                    p_value_cols = [col for col in pairwise_results_df.columns if 'P-Value' in col]
                    strict_df = pairwise_results_df.dropna(subset=p_value_cols)
                    strict_df = strict_df[(strict_df[p_value_cols] < 0.05).all(axis=1)]
                   
                    fully_significant_list = strict_df['Feature'].tolist()
                    print(f"Found {len(fully_significant_list)} features that are significant across all pairwise comparisons.")

                    if fully_significant_list:
                        # Step 3: Filter the original significant features list to maintain the correct order and stats
                        strict_significant_features = significant_features[significant_features['Feature'].isin(fully_significant_list)]

                        # Step 4: Select uncorrelated features from this STRICT list
                        uncorrelated_features = analyzer.select_uncorrelated_features(
                            strict_significant_features, 
                            top_n=25,
                            threshold=0.8
                        )
                       
                        # Step 5: Plot the new heatmap
                        if uncorrelated_features and len(uncorrelated_features) > 1:
                            analyzer.plot_focused_correlation(
                                feature_list=uncorrelated_features,
                                title='Correlation of Top 25 Strictly Significant & Less-Correlated Features',
                                figsize=(24, 22),
                                annotate_cells=True,
                                output_path='heatmap_top25_uncorrelated_strict.png'
                            )
                        
                        
                        # Summary Table from Statistical Tests with the most significant and less correlated features
                        print("\n\n" + "="*80)
                        print(" " * 15 + "STATISTICAL SUMMARY FOR UNCORRELATED HEATMAP FEATURES")
                        print("="*80)

                        # 1. Filter and print the omnibus test results (ANOVA/Kruskal-Wallis)
                        final_omnibus_summary = significant_features[significant_features['Feature'].isin(uncorrelated_features)]
                        print("\n--- Omnibus Test Results (ANOVA / Kruskal-Wallis) ---")
                        columns_to_show = [
                            'Feature', 
                            'Test', 
                            'P-Value', 
                            'Effect_Size_Type', 
                            'Effect_Size', 
                            'Effect_Size_CI95%'
                        ]
                        print(final_omnibus_summary[columns_to_show])

                        # 2. Filter and print the pairwise post-hoc results
                        if posthoc_results is not None and not posthoc_results.empty:
                            final_posthoc_summary = posthoc_results[posthoc_results['Feature'].isin(uncorrelated_features)]
                            # For better readability, sort by the original feature significance order
                            final_posthoc_summary['Feature'] = pd.Categorical(final_posthoc_summary['Feature'], categories=uncorrelated_features, ordered=True)
                            print("\n--- Pairwise Post-Hoc Test Results (Games-Howell / Dunn) ---")
                            #print(final_posthoc_summary.sort_values('Feature'))
                            with pd.option_context('display.max_rows', None):
                                print(final_posthoc_summary.sort_values('Feature'))
                        
                        print("\n" + "="*80 + "\n")
                
       
        elif ANALYSIS_MODE == 'regression':
            print("\n--- Running Regression Analysis ---")
            # Plot feature vs. target scatter plots
            #analyzer.plot_distributions_by_target(num_cols_per_plot=3)
   

            # Number of Features Analyzed
            if features_analyzed:
               print(f"\n--- Number of Features Analyzed for Prediction ---")
               print(f"Total predictive features analyzed: {len(features_analyzed)}")
            
            
            
            # Generate heatmap for top regression features 
            if regression_results is not None and not regression_results.empty:
                
                print("\n--- Top 25 Most Significant Features for Regression ---")
                print(regression_results.head(25))
                print("\n--- All Significant Features for Regression (Sorted by P-Value) ---")
                with pd.option_context('display.max_rows', None):
                    print(regression_results)
                
                # --- Heatmap 1: Top 25 Most Significant Features ---
                print("\n--- Generating Correlation Heatmap for Top 25 Regression Features ---")
                top_reg_features = regression_results['Feature'].head(25).tolist()
                analyzer.plot_focused_correlation(
                    feature_list=top_reg_features,
                    title='Correlation of Top 25 Regression Features',
                    figsize=(20, 18),
                    annotate_cells=True,
                    output_path='heatmap_top25_regression.png'
                )

                # --- Heatmap 2: Top 35 Less-Correlated Features ---
                print("\n--- Generating Correlation Heatmap for Top 25 Less-Correlated Regression Features ---")
                uncorrelated_reg_features = analyzer.select_uncorrelated_features(
                    regression_results,
                    top_n=25,
                    threshold=0.8
                )
                if uncorrelated_reg_features and len(uncorrelated_reg_features) > 1:
                    analyzer.plot_focused_correlation(
                        feature_list=uncorrelated_reg_features,
                        title='Correlation of Top 25 Less-Correlated Regression Features (Threshold=0.8)',
                        figsize=(24, 22),
                        annotate_cells=True,
                        output_path='heatmap_top35_uncorrelated_regression.png'
                    )
                 
                
                    # Summary Table from Statistical Tests with the most significant and less correlated features
                    print("\n\n" + "="*80)
                    print(" " * 12 + "STATISTICAL SUMMARY FOR UNCORRELATED REGRESSION HEATMAP FEATURES")
                    print("="*80)

                    # 1. Filter and print the regression correlation results
                    final_regression_summary = regression_results[regression_results['Feature'].isin(uncorrelated_reg_features)]
                    print("\n--- Pearson Correlation Results ---")
                    columns_to_show = [
                        'Feature', 
                        'Effect_Size_Type', 
                        'Effect_Size', 
                        'P-Value', 
                        'Effect_Size_CI95%'
                    ]
 
                    # Convert the final DataFrame to a string and print it
                    summary_string = final_regression_summary[columns_to_show].to_string(index=False)
                    print(summary_string)
                    
                    print("\n" + "="*80 + "\n")
        
        
        # --- Final Feature Selection for Machine Learning Models ---
        
        # Use the results from your primary analysis (classification or regression)
        if ANALYSIS_MODE == 'classification':
            results_for_selection = significant_features
        else: # regression
            results_for_selection = regression_results

        if results_for_selection is not None and not results_for_selection.empty:
            
            # Helper function to print the summary tables 
            def print_summary_tables(feature_sets_dict, title):
                print("\n\n" + "="*80)
                print(f" " * 20 + f"{title}")
                print("="*80)
                
                for group_name, df_group in feature_sets_dict.items():
                    print(f"\n--- Group: {group_name} (Top {len(df_group)} Features) ---")
                    if df_group.empty:
                        print("No features met the selection criteria for this group.")
                    else:
                        summary_cols = [
                            'Feature', 'Test', 'P-Value', 'Effect_Size_Type', 
                            'Effect_Size', 'Effect_Size_CI95%'
                        ]
                        if 'Test' not in df_group.columns:
                            summary_cols.remove('Test')
                        
                        with pd.option_context('display.max_rows', None):
                            print(df_group[summary_cols])
                print("\n" + "="*80 + "\n")

            # 1. Print the Version WITH the effect size filter 
            print("\n>>> Generating feature sets WITH minimum effect size filter...")
            effect_size_thresholds = {
                "Epsilon-Squared": 0.01,
                "Partial Eta-Squared": 0.01,
                "Pearson's r": 0.1,
            }

            feature_sets_with_filter = analyzer.select_final_feature_sets(
                significant_features_df=results_for_selection,
                top_n=50,
                correlation_threshold=0.8,
                min_effect_sizes=effect_size_thresholds
            )
            print_summary_tables(feature_sets_with_filter, "SELECTED FEATURES (WITH EFFECT SIZE FILTER)")
            
            
            # Convert the DataFrames to a dictionary of lists and save to JSON 
            feature_lists_to_save = {
                name: df['Feature'].tolist() for name, df in feature_sets_with_filter.items()
            }
            
            features_filename = f"{ANALYSIS_MODE}_selected_feature_sets.json"
            output_path = os.path.join(COMMON_PARAMS["data_dir_db"], features_filename)
            try:
                with open(output_path, 'w') as f:
                    json.dump(feature_lists_to_save, f, indent=4)
                print(f"\nSuccessfully saved selected feature lists to: {output_path}")
            except Exception as e:
                print(f"\nError saving feature lists to JSON: {e}")


            # 2. Print the Version WITHOUT the effect size filter 
            print("\n>>> Generating feature sets WITHOUT minimum effect size filter...")
            
            feature_sets_without_filter = analyzer.select_final_feature_sets(
                significant_features_df=results_for_selection,
                top_n=50,
                correlation_threshold=0.8
            )
            print_summary_tables(feature_sets_without_filter, "SELECTED FEATURES (WITHOUT EFFECT SIZE FILTER)")
            
            print("\n" + "="*80 + "\n")
                    
   
        # Steps Plots
        print("\n--- Plotting Step Distributions by Meal Category ---")
        #analyzer.plot_postmeal_steps_by_meal_category()
        #analyzer.plot_steps_by_meal_category()
    
    
    
    
    if run_model_specific_sfs:
        print(f"\n--- Fase 5: Esecuzione della Selezione Feature per Modello e Metrica Specifici ---")
        
        # Define the metrics to optimize for each analysis type
        classification_scorers = {
            'roc_auc': 'roc_auc',
            # 'recall': 'recall',
            # 'precision': 'precision',
            'f1_score': 'f1'
        }
        
        multiclass_scorers = {
            'roc_auc': 'roc_auc_ovr_weighted',
            # 'recall': 'recall_weighted',
            # 'precision': 'precision_weighted',
            'f1_score': 'f1_weighted'
        }

        regression_scorers = {
            'r2': 'r2'
        }

        # --- BINARY CLASSIFICATION mode ---
        for scorer_name, scorer_metric in classification_scorers.items():
            binary_selector = ModelSpecificFeatureSelector(
                data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
                stats_file_path=classification_stats_file,
                analysis_type='binary_classification'
            )
            output_filename = f"sfs_per_model_binary_classification_{scorer_name}.json"
            binary_selector.run_sfs_for_all_models(
                scorer_name=scorer_name,
                scorer_metric=scorer_metric,
                output_path=os.path.join(COMMON_PARAMS["data_dir_db"], output_filename)
            )

        # --- MULTI-CLASS CLASSIFICATION mode ---
        for scorer_name, scorer_metric in multiclass_scorers.items():
            multiclass_selector = ModelSpecificFeatureSelector(
                data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
                stats_file_path=classification_stats_file,
                analysis_type='multi_class_classification'
            )
            output_filename = f"sfs_per_model_multi_class_classification_{scorer_name}.json"
            multiclass_selector.run_sfs_for_all_models(
                scorer_name=scorer_name,
                scorer_metric=scorer_metric,
                output_path=os.path.join(COMMON_PARAMS["data_dir_db"], output_filename)
            )
        
        # --- REGRESSION mode ---
        for scorer_name, scorer_metric in regression_scorers.items():
            regression_selector = ModelSpecificFeatureSelector(
                data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
                stats_file_path=regression_stats_file,
                analysis_type='regression'
            )
            output_filename = f"sfs_per_model_regression_{scorer_name}.json"
            regression_selector.run_sfs_for_all_models(
                scorer_name=scorer_name,
                scorer_metric=scorer_metric,
                output_path=os.path.join(COMMON_PARAMS["data_dir_db"], output_filename)
            )
    
    
    
    
    if display_sfs_results:
        print(f"\n--- Fase 8: Visualizzazione dei Risultati SFS Salvati ---")
        
        # --- Part A: Generate and Display the Candidate Feature Pools (with new NaN filtering) ---
        print("\n" + "#"*80)
        print("### DYNAMICALLY GENERATING CANDIDATE FEATURE POOLS (with NaN filtering) ###")
        print("#"*80)

        analysis_configs = [
            {'name': 'Classification', 'stats_file': classification_stats_file, 'target': 'target_hypoglycemia', 'is_regression': False},
            {'name': 'Regression', 'stats_file': regression_stats_file, 'target': 'postprandial_glucose_min', 'is_regression': True}
        ]

        for config in analysis_configs:
            print(f"\n--- Generating Candidate Pool for: {config['name']} Analysis ---")
            
            # 1. Load data
            df_full_data = pd.read_csv(os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]))
            try:
                df_stats = pd.read_csv(config['stats_file'])
            except FileNotFoundError:
                print(f"ERROR: Statistics file not found at '{config['stats_file']}'. Skipping pool generation.")
                continue
                
            ordered_feature_list = df_stats['Feature'].tolist()

            # 2. Apply row filters (using filter lists defined earlier in main.py)
            df = df_full_data.copy()
            for condition in non_wearable_filters + wearable_filters:
                col, op, val = condition['column'], condition['operator'], condition['value']
                if col in df.columns:
                    if op == '>': df = df[df[col] > val]
                    elif op == '<': df = df[df[col] < val]
                    elif op == '==': df = df[df[col] == val]
                    elif op == '!=': df = df[df[col] != val]

            # 3. Handle target-specific filtering
            if config['is_regression']:
                df = df[df[config['target']] > 0].copy()
            df.dropna(subset=[config['target']], inplace=True)

            # 4. Create initial feature matrix and remove leaky features
            valid_ordered_features = [f for f in ordered_feature_list if f in df.columns]
            leaky_features_to_remove = [
                'activity_proportion_after_peak',
                'pulse_ox_mean', 'pulse_ox_std', 'pulse_ox_min', 'pulse_ox_max', 'pulse_ox_range'
            ]
            cleaned_ordered_features = [f for f in valid_ordered_features if f not in leaky_features_to_remove]
            X_initial = df[cleaned_ordered_features]
            
            # 5. De-correlate the feature matrix
            X_decorrelated = drop_correlated_features(X_initial, threshold=0.8)
            
            # 6. Calculate and print NaN percentage for each candidate feature
            nan_percentages = X_decorrelated.isnull().sum() * 100 / len(X_decorrelated)
            nan_df = nan_percentages.reset_index()
            nan_df.columns = ['Feature', 'NaN %']
            nan_df = nan_df.sort_values(by='NaN %', ascending=False)
            
            print("\n" + "="*80)
            print(f"CANDIDATE FEATURES (POST-CORRELATION) for {config['name']}")
            print(f"Total Features: {len(nan_df)}")
            print("="*80)
            print(nan_df.to_string(index=False))

            # 7. Filter out features with more than 30% NaNs
            nan_threshold = 30.0
            features_to_drop = nan_percentages[nan_percentages > nan_threshold].index.tolist()
            X_final = X_decorrelated.drop(columns=features_to_drop)
            candidate_features = X_final.columns.tolist()

            print(f"\n>>> Dropping {len(features_to_drop)} features with > {nan_threshold}% missing values.")
            if features_to_drop:
                print("Dropped features:", ", ".join(features_to_drop))

            # 8. Print the final, refined feature pool
            print("\n" + "="*80)
            print(f"--- FINAL CANDIDATE POOL FOR SFS for: {config['name']} Analysis ---")
            print(f"Total Features in Pool: {len(candidate_features)}")
            print("="*80)
            for i, feature in enumerate(candidate_features, 1):
                print(f"  {i:3d}. {feature}")
            print("\n")


        # --- Part B: Display the final SFS results from saved files ---
        print("\n" + "#"*80)
        print("### FINAL SELECTED FEATURES (from saved SFS results) ###")
        print("#"*80)

        # List of all the result files you want to display
        sfs_result_files = [
            "sfs_per_model_binary_classification_roc_auc.json",
            "sfs_per_model_binary_classification_f1_score.json",
            "sfs_per_model_multi_class_classification_roc_auc.json",
            "sfs_per_model_multi_class_classification_f1_score.json",
            "sfs_per_model_regression_r2.json"
        ]

        for filename in sfs_result_files:
            file_path = os.path.join(COMMON_PARAMS["data_dir_db"], filename)
            try:
                with open(file_path, 'r') as f:
                    feature_sets = json.load(f)
                
                print("\n" + "#"*80)
                print(f"### Results from: {filename} ###")
                print("#"*80)

                for set_name, features in feature_sets.items():
                    print("\n" + "="*80)
                    print(f"--- Feature Set: '{set_name}' ---")
                    print(f"Total Features: {len(features)}")
                    print("="*80)
                    if not features:
                        print("  (No features selected for this set)")
                    else:
                        for i, feature in enumerate(features, 1):
                            print(f"  {i:2d}. {feature}")
                print("\n")

            except FileNotFoundError:
                print(f"\nWARNING: SFS result file not found at '{file_path}'. Skipping.")
            except Exception as e:
                print(f"\nERROR: Could not read or parse file '{file_path}'. Error: {e}")




    elif run_feature_selection:
        print(f"\n--- Fase 5: Esecuzione della Selezione Feature (Pre-meal, post-meal, e mixed sets) ---")
        
        # --- REGRESSION mode ---
        print("\n--- Avvio SFS per la modalità: REGRESSION ---")
        reg_selector = FeatureSelector(
            data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
            initial_features_path=os.path.join(COMMON_PARAMS["data_dir_db"], "regression_selected_feature_sets.json"),
            analysis_type='regression'
        )
        reg_selector.select_features(
            scoring_metric='r2',
            output_path=os.path.join(COMMON_PARAMS["data_dir_db"], "sfs_selected_features_regression_r2.json"),
            plot_results=True
        )

        # --- CLASSIFICATION mode ---
        print("\n--- Avvio SFS per la modalità: CLASSIFICATION ---")
        class_selector = FeatureSelector(
            data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
            initial_features_path=os.path.join(COMMON_PARAMS["data_dir_db"], "classification_selected_feature_sets.json"),
            analysis_type='classification'
        )
        # 1. Run for ROC-AUC
        class_selector.select_features(
            scoring_metric='roc_auc',
            output_path=os.path.join(COMMON_PARAMS["data_dir_db"], "sfs_selected_features_classification_roc_auc.json"),
            plot_results=True
        )
        # 2. Run for Recall
        class_selector.select_features(
            scoring_metric='recall',
            output_path=os.path.join(COMMON_PARAMS["data_dir_db"], "sfs_selected_features_classification_recall.json"),
            plot_results=True
        )

    
    
    
    if run_model_training:
        if ANALYSIS_MODE == 'hybrid':
            print(f"\n--- Fase 6: Training Modelli Ibridi (Classificazione + Regressione) ---")
            
            # Instantiate and run the pipeline. No need to define models here.
            hybrid_trainer = HybridModelTrainer(
                data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
                features_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], features_filename),
                classification_target='target_hypoglycemia',
                regression_target='postprandial_glucose_min',
                random_state=42
            )
            hybrid_trainer.run_pipeline(use_smote=USE_SMOTE_SAMPLING)
            #hybrid_trainer.save_results_to_csv("tuned_hybrid_performance_log.csv")
        
        elif ANALYSIS_MODE == 'two_step':
            print(f"\n--- Fase 6: Training Modelli a Due Fasi (Multi-Class) ---")

            # Use the classification features file
            features_filename = "classification_selected_feature_sets.json" 

            two_step_trainer = TwoStepModelTrainer(
                data_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
                features_file_path=os.path.join(COMMON_PARAMS["data_dir_db"], features_filename),
                classification_target='target_hypoglycemia',
                random_state=42
            )
            two_step_trainer.run(use_smote=USE_SMOTE_SAMPLING)
            #two_step_trainer.save_results_to_csv("tuned_two_step_performance_log.csv")


        else:
            print(f"\n--- Fase 6: Training dei Modelli ({trainer_analysis_type}) ---")
            trainer = ModelTrainer(**TRAINER_PARAMS)
            trainer.train(use_smote=USE_SMOTE_SAMPLING)
            # log_filename = f"{trainer_analysis_type}_performance_log.csv"
            # trainer.save_results_to_csv(log_filename)
    
    
    
    
    if run_performance_analysis:
        print(f"\n--- Fase 7: Analisi delle Performance dei Modelli ---")
        
        # Instantiate the analyzer with the path to the logs
        performance_analyzer = PerformanceAnalyzer(logs_directory=COMMON_PARAMS["data_dir_db"])
        
        if not performance_analyzer.master_df.empty:
            # Display the summary table of best models per feature set
            print("\n--- Summary of Best Performing Models ---")
            performance_analyzer.display_best_models(sort_by='roc_auc_ovr_weighted')

            # Plot key comparison metrics
            performance_analyzer.plot_metric_comparison(
                metric='recall_weighted',
                title='Model Comparison: Weighted Recall'
            )
            performance_analyzer.plot_metric_comparison(
                metric='roc_auc_ovr_weighted',
                title='Model Comparison: Weighted AUC-ROC (OvR)'
            )
            
            # For binary or hybrid models that produce these metrics
            performance_analyzer.plot_precision_recall_tradeoff()

            # Analyze which feature set was most effective
            performance_analyzer.analyze_feature_sets(metric='roc_auc_ovr_weighted')
            
            # Print all the model logs for each analysis type
            performance_analyzer.display_log_summary()
        else:
            print("Skipping performance analysis as no log data was found.")
        
        
        
        
    if run_time_gain_analysis:
        if 'classification' in TRAINER_PARAMS['analysis_type']:
            print(f"\n--- Fase 8: Analisi delle Performance per Time Gain ---")
            # Instantiate and run the new analyzer using the same shared parameters
            time_gain_analyzer = TimeGainAnalyzer(**TRAINER_PARAMS)
            time_gain_analyzer.run_analysis(use_smote=USE_SMOTE_SAMPLING)
        else:
            print(f"\n--- Fase 8: Analisi delle Performance per Time Gain ---")
            print("Skipping Time Gain Analysis: This analysis is only applicable for classification modes.")
            print("For regression, the equivalent 'Prediction Horizon' is calculated during the main model training.")






        
        
        
if __name__ == "__main__":
   
    run_pipeline(
        run_preparation = False, # Esegui la preparazione dei dati
        visualize_preparation_steps = False, # Imposta a True per vedere i plot durante la preparazione (può essere molto verboso)
        run_feature_extraction = False, # Esegui l'estrazione delle feature
        run_visualization = False, # Avvia il visualizzatore interattivo
        run_data_analysis = False, 
        run_feature_selection = False,
        run_model_specific_sfs = False,
        display_sfs_results = False,
        run_model_training = True,
        run_performance_analysis = False,
        run_time_gain_analysis = False,
        RELOAD_STATS_FROM_CSV = True # Set to True to load, False to recalculate the effect size and CIs
    )