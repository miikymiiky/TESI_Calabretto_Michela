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
from src.training.model_trainer import ModelTrainer
import pandas as pd
import json

plt.close('all')


def run_pipeline(run_preparation=True, run_feature_extraction=True,run_visualization=True, 
                 visualize_preparation_steps=False, run_data_analysis = False, run_model_training=True,
                 RELOAD_STATS_FROM_CSV=True): # Flag for the effect size and CIs computation ---> True: loads the saved values from the csv files; False: runs the entire extrapolation process (bootstraping, ecc...)  
    
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
    ANALYSIS_MODE = 'regression'  # Options: 'classification' or 'regression'

    ANALIZER_PARAMS = {
        "data_file_path":os.path.join(COMMON_PARAMS["data_dir_db"],EXTRACTOR_PARAMS["output_filename_with_features"]), 
        "analysis_type": ANALYSIS_MODE,
        "classification_target": 'target_hypoglycemia',
        "regression_target": 'postprandial_glucose_min',
    }
    
    # --- Parametri Specifici per ModelTrainer ---
    features_filename = f"{ANALYSIS_MODE}_selected_feature_sets.json"
    TRAINER_PARAMS = {
        "data_file_path": os.path.join(COMMON_PARAMS["data_dir_db"], EXTRACTOR_PARAMS["output_filename_with_features"]),
        "features_file_path": os.path.join(COMMON_PARAMS["data_dir_db"], features_filename),
        "analysis_type": ANALYSIS_MODE,
        "classification_target": 'target_hypoglycemia',
        "regression_target": 'postprandial_glucose_min',
        "test_size": 0.20, # 20% for the test set
        "random_state": 42 # For reproducible results
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
        
        # 1. Apply filters
        # SPLITTING FILTERS 
        # Define non-wearable (physiological) filters
        non_wearable_filters = [
            {'column': 'hypo_to_check_flag', 'operator': '!=', 'value': 0},
            {'column': 'postprandial_duration_minutes', 'operator': '>', 'value': 120},
            {'column': 'time_peak_hypo_event', 'operator': '>', 'value': -1}, # -1 è il flag che consente di selezionare anche i pasti che non vanno in ipo 
            {'column': 'rate_of_increase', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_before_peak1', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_after_peak1', 'operator': '<', 'value': 0},
        ]

        # Define wearable filters
        wearable_filters = [
            {'column': 'has_heart_rate_data', 'operator': '==', 'value': 1},
            {'column': 'has_pulse_ox_data', 'operator': '==', 'value': 1},
            {'column': 'has_steps_data', 'operator': '==', 'value': 1},
        ]
        
        # APPLY FILTERS IN STAGES
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
    
    
    
    
    if run_model_training:
        print("\n--- Fase 5: Training dei Modelli di Machine Learning ---")
        trainer = ModelTrainer(**TRAINER_PARAMS)

        
        # The train method will handle splitting and the model training
        trainer.train()
        
        log_filename = f"{ANALYSIS_MODE}_performance_log.csv"
        trainer.save_results_to_csv(log_filename)
        
        
        
        
if __name__ == "__main__":
   
    run_pipeline(
        run_preparation = False, # Esegui la preparazione dei dati
        visualize_preparation_steps = False, # Imposta a True per vedere i plot durante la preparazione (può essere molto verboso)
        run_feature_extraction = False, # Esegui l'estrazione delle feature
        run_visualization = False, # Avvia il visualizzatore interattivo
        run_data_analysis = False, 
        run_model_training = True,
        RELOAD_STATS_FROM_CSV = True # Set to True to load, False to recalculate the effect size and CIs
    )