# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:23:12 2025

@author: prend
"""

# src/feature_extraction/postprandial_feature_extractor.py

import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from src.utils.utils4processing import interpolate_saturation_events, apply_zero_phase_low_pass_filter


class PostprandialFeatureExtractor:
    """
    Classe per la definizione del target e l'estrazione di feature dai periodi 
    post-prandiali.
    """
    def __init__(self, data_dir="data", data_dir_db="postprandial_db",
                 postprandial_data_file='all_postprandial_data.csv',
                 glucose_filename_pattern="{}tma_processed_new.csv", #"{}_tma_processed.csv",
                 hypo_threshold=70, deep_hypo_threshold=54,
                 min_consecutive_samples_hypo=3, # 3 campioni a 5 minuti = 15 minuti
                 peak_search_min_distance_samples=6, # 6 campioni a 5 minuti = 30 minuti
                 peak_search_prominence=20, #mg/dL
                 output_filename_with_features = 'all_postprandial_data_with_features.csv',
                 pre_meal_intervals_hours=[1, 2, 3, 6, 12, 24], # Intervalli di tempo per considerare la storia passata della glicmeia
                 peak_prominence_pre_meal=10, # Prominenza per i picchi pre-pasto (può essere diversa)
                 sampling_interval_minutes=5, # Intervallo di campionamento CGM in minuti 
                 wearable_variables=['HEART_RATE', 'HEART_RATE_VARIABILITY', 'PULSE_OX', 'STEPS'],
                 daily_step_target=10000,
                 hr_activity_hrr_intensity=0.4, # 40% of Heart Rate Reserve for light/moderate activity
                 min_cadence_for_activity=20,
                 hr_activity_hrr_intensities={
                     'light': [0.3, 0.39],
                     'moderate': [0.4, 0.59],
                     'vigorous': [0.6, 0.89]
                 },
                 # Add filter parameters here 
                 sampling_rate_seconds=300, 
                 cutoff_frequency_hz=2*1.67e-4, 
                 filter_order=1
                 ): 
        """
        Inizializza l'estrattore di feature.

        Args:
            data_dir (str): Directory principale con i dati dei soggetti.
            data_dir_db (str): Directory del database post-prandiale.
            postprandial_data_file (str): Nome del file del database post-prandiale.
            glucose_filename_pattern (str): Pattern del nome file per i dati di glucosio processati.
            hypo_threshold (int): Soglia di glucosio per l'ipoglicemia.
            deep_hypo_threshold (int): Soglia di glucosio per l'ipoglicemia profonda.
            min_consecutive_samples_hypo (int): Minimo numero di campioni consecutivi sotto soglia per ipoglicemia.
            peak_search_min_distance_samples (int): Minima distanza tra i picchi in numero di campioni.
            peak_search_prominence (int): Prominenza minima per la rilevazione dei picchi.
            output_filename (str): Directory in cui salvare il database post-prandiale con le features. 
        """
        self.data_dir = data_dir
        self.data_dir_db = data_dir_db
        self.postprandial_data_file = postprandial_data_file
        self.glucose_filename_pattern = glucose_filename_pattern
        self.HYPO_THRESHOLD = hypo_threshold
        self.DEEP_HYPO_THRESHOLD = deep_hypo_threshold
        self.MIN_CONSECUTIVE_SAMPLES_HYPO = min_consecutive_samples_hypo
        self.PEAK_SEARCH_MIN_DISTANCE_SAMPLES = peak_search_min_distance_samples
        self.PEAK_SEARCH_PROMINENCE = peak_search_prominence
        self.output_filename = output_filename_with_features
        self.PRE_MEAL_INTERVALS_HOURS = pre_meal_intervals_hours
        self.PEAK_PROMINENCE_PRE_MEAL = peak_prominence_pre_meal
        self.SAMPLING_INTERVAL_MINUTES = sampling_interval_minutes
        self.WEARABLE_VARIABLES = wearable_variables
        self.DAILY_STEP_TARGET = daily_step_target
        self.HR_ACTIVITY_HRR_INTENSITY = hr_activity_hrr_intensity
        self.MIN_CADENCE_FOR_ACTIVITY = min_cadence_for_activity
        self.HR_ACTIVITY_HRR_INTENSITIES = hr_activity_hrr_intensities
        self.sampling_rate_seconds = sampling_rate_seconds
        self.cutoff_frequency_hz = cutoff_frequency_hz
        self.filter_order = filter_order


    def _load_glucose_df_for_subject(self, subject_id):
        """Carica il DataFrame del glucosio processato per un soggetto."""
        glucose_df_path = os.path.join(self.data_dir, subject_id, self.glucose_filename_pattern.format(subject_id))
        try:
            return pd.read_csv(glucose_df_path, index_col='Time', parse_dates=True)
        except FileNotFoundError:
            print(f"Errore: File glucosio processato non trovato per {subject_id} a {glucose_df_path}")
            return None
        except Exception as e:
            print(f"Errore caricamento glucosio processato per {subject_id}: {e}")
            return None
    
    
    def _load_all_wearable_data_for_subject(self, subject_id):
        """
        Loads all available wearable DataFrames for a subject.
        Returns a dictionary mapping variable names to DataFrames.
        """
        all_wearable_data = {}
        for var_name in self.WEARABLE_VARIABLES:
            file_path = os.path.join(self.data_dir, subject_id, f"{var_name}.csv")
        
            if os.path.exists(file_path):
                try:
                    # Manually assign header names to handle files without a header row.
                    df = pd.read_csv(file_path, header=None, names=['Time', var_name])
                
                    # Check if the dataframe is empty after loading
                    if df.empty:
                        continue # move to the next file
                    
                    # Convert the data column to numeric, coercing errors to NaN
                    df[var_name] = pd.to_numeric(df[var_name], errors='coerce')
                    
                    # Replace -1 with NaN 
                    df[var_name] = df[var_name].replace(-1, np.nan)
                    
                    # Convert the 'Time' column using the correct format
                    df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
                    df.set_index('Time', inplace=True)
                    
                    df.sort_index(inplace=True)
                
                    all_wearable_data[var_name] = df
                except Exception as e:
                    print(f"Warning: An unexpected error occurred loading {var_name}.csv for {subject_id}: {e}")
        
        return all_wearable_data


    def _detect_hypoglycemia(self, glucose_series, glucose_derivative_series, start_time, end_time):
        """
        rileva ipoglicemia e ipoglicemia profonda in un periodo, inclusa la durata.
        Restituisce (is_hypo, hypo_time, hypo_duration, is_deep_hypo, deep_hypo_time, deep_hypo_duration)
        """
        # TODO: rendere più forte la ricerca dell'ipo
        # cerca ipoglicemia dopo start_time E dopo la prima salita
        # così evito di labellare com ehypo o deep_hypo i pasti che inizio sotto 70 o 54
        period_glucose = glucose_series.loc[start_time+pd.Timedelta(minutes=30):end_time]
            
        is_hypo = 0
        hypo_time = np.nan
        hypo_duration = np.nan
        #hypo_duration = 0

        is_deep_hypo = 0
        deep_hypo_time = np.nan
        deep_hypo_duration = np.nan
        #deep_hypo_duration = 0

        # Rilevamento ipoglicemia
        hypo_segments = (period_glucose < self.HYPO_THRESHOLD).astype(int).groupby((period_glucose < self.HYPO_THRESHOLD).astype(int).diff().ne(0).cumsum()).filter(lambda x: len(x) >= self.MIN_CONSECUTIVE_SAMPLES_HYPO and x.iloc[0] == 1)
                      # .astype(int) ---> converts the True/False series into a 1/0 series
                      # .ne(0) = not equal ---> compares the result of diff() with 0 and returns True where the value changed (where the value is not 0)
                      # .cumsum() ---> each time True (which is 1 in summation) appears, the sum increments ---> this effectively creates a unique group ID for each consecutive block of identical values 
                      # .groupby() ---> groups the original integer Series based on the unique group IDs generated by cumsum()
                      
                      # .filter(lambda x: len(x) >= self.MIN_CONSECUTIVE_SAMPLES_HYPO and x.iloc[0] == 1):
                      # lambda x: ---> defines an anonymous function that takes a sub-Series x (representing a consecutive segment of 0s or 1s)
                      # x.iloc[0] == 1 ---> condition that ensures that the segment actually represents values below the threshold ( segment of 1s, not 0s)
        if not hypo_segments.empty:
            is_hypo = 1
            # Il primo momento in cui si verificano almeno MIN_CONSECUTIVE_SAMPLES_HYPO valori consecutivi
            first_hypo_segment_start_idx = hypo_segments.index[0]
            hypo_time = first_hypo_segment_start_idx
            hypo_duration = (hypo_segments.index[-1] - first_hypo_segment_start_idx).total_seconds() / 60 + 5 # + 5 min per l'ultimo campione

        # Rilevamento ipoglicemia profonda
        deep_hypo_segments = (period_glucose < self.DEEP_HYPO_THRESHOLD).astype(int).groupby((period_glucose < self.DEEP_HYPO_THRESHOLD).astype(int).diff().ne(0).cumsum()).filter(lambda x: len(x) >= self.MIN_CONSECUTIVE_SAMPLES_HYPO and x.iloc[0] == 1)
        if not deep_hypo_segments.empty:
            is_deep_hypo = 1
            # Il primo momento in cui si verificano almeno MIN_CONSECUTIVE_SAMPLES_HYPO valori consecutivi
            first_deep_hypo_segment_start_idx = deep_hypo_segments.index[0]
            deep_hypo_time = first_deep_hypo_segment_start_idx
            deep_hypo_duration = (deep_hypo_segments.index[-1] - first_deep_hypo_segment_start_idx).total_seconds() / 60 + 5 # + 5 min per l'ultimo campione

        return is_hypo, hypo_time, hypo_duration, is_deep_hypo, deep_hypo_time, deep_hypo_duration


    def _find_postprandial_peaks(self, glucose_series, meal_start_time, meal_end_time):
        """
        Trova il primo e il secondo picco glicemico post-prandiale.
        Restituisce (peak1_val, time_to_peak1, peak2_val, time_to_peak2)
        """
        
        # Filtra i dati di glucosio per il periodo post-prandiale
        postprandial_glucose = glucose_series.loc[meal_start_time:meal_end_time].copy()
        
        if postprandial_glucose.empty:
            return np.nan, np.nan, np.nan, np.nan

        # La funzione find_peaks vuole un array numpy senza DatetimeIndex
        glucose_values = postprandial_glucose.values
        
        # Considera solo valori non NaN per la ricerca dei picchi
        valid_indices = np.where(~np.isnan(glucose_values))[0] # [0] ---> we only care about the first (and only) dimension, so we select the array of indices
        if len(valid_indices) < 2: # Abbiamo bisogno di almeno 2 punti per trovare un picco
            return np.nan, np.nan, np.nan, np.nan

        peaks, properties = find_peaks(
            glucose_values[valid_indices],
            prominence=self.PEAK_SEARCH_PROMINENCE,
            distance=self.PEAK_SEARCH_MIN_DISTANCE_SAMPLES
        )
        
        # Mappa gli indici dei picchi all'array originale con NaN
        original_peak_indices = valid_indices[peaks]
        
        # Ordina i picchi per tempo
        sorted_peaks_info = []
        
        for i, peak_idx in enumerate(original_peak_indices):
            peak_time = postprandial_glucose.index[peak_idx]
            peak_value = glucose_values[peak_idx]
            time_from_meal_start = (peak_time - meal_start_time).total_seconds() / 60
            sorted_peaks_info.append((peak_time, peak_value, time_from_meal_start))
        
        sorted_peaks_info.sort(key=lambda x: x[0]) # Ordina per tempo

        peak1_val, time_to_peak1 = np.nan, np.nan
        peak2_val, time_to_peak2 = np.nan, np.nan

        if len(sorted_peaks_info) >= 1:
            peak1_val = sorted_peaks_info[0][1]
            time_to_peak1 = sorted_peaks_info[0][2]

        if len(sorted_peaks_info) >= 2:
            peak2_val = sorted_peaks_info[1][1]
            time_to_peak2 = sorted_peaks_info[1][2]

        # modifica: il picco principale è peak1_val il resto serve solo per analisi ma non per training modello
        if len(original_peak_indices) !=0:
            peak_values = glucose_values[original_peak_indices]
            peak_times = postprandial_glucose.index[original_peak_indices]
            max_peak_idx = peak_values.argmax()
        
            main_peak_time = peak_times[max_peak_idx]
            main_peak_value = peak_values[max_peak_idx]

            peak1_val = main_peak_value
            time_to_peak1 = (main_peak_time - meal_start_time).total_seconds()/60
        
        return peak1_val, time_to_peak1, peak2_val, time_to_peak2
    
    
    def _calculate_excursion_metrics(self, glucose_series, condition, min_samples_for_event=1):
        """
        Calculates the number and total duration of glycemic excursions.
        An excursion is a consecutive block of samples meeting a certain condition.

        Args:
            glucose_series (pd.Series): The glucose data for the window.
            condition (lambda): A function that returns True for values in the excursion range (e.g., lambda g: g > 180).
            min_samples_for_event (int): The minimum number of consecutive samples to be considered a valid event.

        Returns:
            tuple: A tuple containing (event_count, total_duration_minutes).
        """
        if glucose_series.empty or glucose_series.isnull().all():
            return 0, 0

        in_excursion = condition(glucose_series.dropna())
        if in_excursion.empty:
            return 0, 0
            
        excursion_groups = in_excursion.ne(in_excursion.shift()).cumsum() # ne(shift()) finds where the True/False state changes
                                                                          # cumsum() assigns a group number to each block
        true_excursion_blocks = excursion_groups[in_excursion] # filter only True blocks
        if true_excursion_blocks.empty:
            return 0, 0

        block_sizes = true_excursion_blocks.value_counts() # value_counts() is a pandas method that counts how many True samples are in each block
        valid_events = block_sizes[block_sizes >= min_samples_for_event]
        event_count = len(valid_events)
        total_duration_minutes = valid_events.sum() * self.SAMPLING_INTERVAL_MINUTES

        return event_count, total_duration_minutes
    
    
    def _calculate_mage(self, glucose_series: pd.Series) -> float: # -> float specifies that the function is expected to return a float value
        """
        Calculates the Mean Amplitude of Glycemic Excursions (MAGE).
        This implementation follows the automated algorithm for CGM signals. 
        """
        if glucose_series.empty or glucose_series.isnull().all():
            return np.nan

        # 1. Identify local extrema (turning points)
        glucose_values = glucose_series.dropna().values
        if len(glucose_values) < 3:
            return np.nan

        peaks, _ = find_peaks(glucose_values)
        troughs, _ = find_peaks(-glucose_values)
        turning_points_indices = np.sort(np.concatenate([peaks, troughs]))

        if len(turning_points_indices) < 2:
            return np.nan

        # 2. Calculate excursions between turning points
        excursions = np.abs(np.diff(glucose_values[turning_points_indices]))
        
        # 3. Filter for significant excursions (greater than the overall SD of the series)
        series_std = glucose_series.std()
        if pd.isna(series_std) or series_std == 0:
            return 0.0
            
        significant_excursions = excursions[excursions > series_std]

        if len(significant_excursions) == 0:
            return 0.0
            
        # 4. MAGE is the mean of these significant excursions 
        return np.mean(significant_excursions)
    
    
    def _calculate_short_time_features(self, glucose_series: pd.Series):
        """
        Calculates short-time features based on the Glucose Rate of Change (ROC). 
        """
        features = {'lastROC': np.nan, 'meanROC': np.nan}
        
        if len(glucose_series.dropna()) < 4: # Need at least 4 samples for a 15-min diff
            return features
            
        # ROC is the difference between the current sample and the one 15 mins prior (3 samples ago) divided by 15 minutes
        roc_series = (glucose_series - glucose_series.shift(3)) / 15.0
        
        if roc_series.dropna().empty:
            return features

        features['lastROC'] = roc_series.iloc[-1]
        features['meanROC'] = roc_series.mean()
        
        return features


    def _calculate_pre_meal_metrics(self, glucose_series_filtered, meal_start_time, first_glucose_timestamp):
        metrics = {}
        min_distance_samples_pre_meal = int(30 / self.SAMPLING_INTERVAL_MINUTES)
        if min_distance_samples_pre_meal < 1: min_distance_samples_pre_meal = 1
        
        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            keys_to_init = [
                f'mean_glucose_{interval_name}', f'std_glucose_{interval_name}', f'cv_glucose_{interval_name}', f'min_glucose_{interval_name}', f'max_glucose_{interval_name}', f'range_{interval_name}', 
                f'prop_above_130_{interval_name}', f'prop_80_130_{interval_name}', f'prop_50_80_{interval_name}', f'prop_below_50_{interval_name}', f'num_positive_peaks_{interval_name}', f'num_negative_peaks_{interval_name}',
                f'num_deep_hypo_events_{interval_name}', f'duration_deep_hypo_minutes_{interval_name}',
                
                f'num_hypo_events_{interval_name}', f'duration_hypo_minutes_{interval_name}',
                f'num_hyper_events_{interval_name}', f'duration_hyper_minutes_{interval_name}',
                f'num_total_hypo_events_{interval_name}', f'duration_total_hypo_minutes_{interval_name}',
                
                f'lbgi_{interval_name}', f'hbgi_{interval_name}'
            ]
            for key in keys_to_init:
                metrics[key] = np.nan

            interval_start_time = meal_start_time - pd.Timedelta(hours=interval_hours)
            interval_start_time = max(interval_start_time, first_glucose_timestamp)
            pre_meal_glucose = glucose_series_filtered.loc[interval_start_time:meal_start_time - pd.Timedelta(minutes=self.SAMPLING_INTERVAL_MINUTES)]
            
            if pre_meal_glucose.empty or pre_meal_glucose.isnull().all():
                continue

            metrics[f'mean_glucose_{interval_name}'] = pre_meal_glucose.mean()
            metrics[f'std_glucose_{interval_name}'] = pre_meal_glucose.std()
            metrics[f'min_glucose_{interval_name}'] = pre_meal_glucose.min()
            metrics[f'max_glucose_{interval_name}'] = pre_meal_glucose.max()
            metrics[f'range_{interval_name}'] = pre_meal_glucose.max() - pre_meal_glucose.min()
            if pre_meal_glucose.mean() > 0:
                metrics[f'cv_glucose_{interval_name}'] = (pre_meal_glucose.std() / pre_meal_glucose.mean()) * 100
            
            total_samples = len(pre_meal_glucose.dropna())
            if total_samples > 0:
                metrics[f'prop_above_130_{interval_name}'] = (pre_meal_glucose > 130).sum() / total_samples
                metrics[f'prop_80_130_{interval_name}'] = ((pre_meal_glucose >= 80) & (pre_meal_glucose <= 130)).sum() / total_samples
                metrics[f'prop_50_80_{interval_name}'] = ((pre_meal_glucose >= 50) & (pre_meal_glucose < 80)).sum() / total_samples
                metrics[f'prop_below_50_{interval_name}'] = (pre_meal_glucose < 50).sum() / total_samples
            
            positive_peaks, _ = find_peaks(pre_meal_glucose.dropna().values, prominence=self.PEAK_PROMINENCE_PRE_MEAL, distance=min_distance_samples_pre_meal)
            metrics[f'num_positive_peaks_{interval_name}'] = len(positive_peaks)
            negative_peaks, _ = find_peaks(-pre_meal_glucose.dropna().values, prominence=self.PEAK_PROMINENCE_PRE_MEAL, distance=min_distance_samples_pre_meal)
            metrics[f'num_negative_peaks_{interval_name}'] = len(negative_peaks)
            
            # This block calculates the new event-based features using the helper method
            # Define conditions for different event types
            deep_hypo_cond = lambda g: g < self.DEEP_HYPO_THRESHOLD  # < 54
            hypo_cond = lambda g: (g >= self.DEEP_HYPO_THRESHOLD) & (g < self.HYPO_THRESHOLD) # 54-70
            hyper_cond = lambda g: g > 130 # Threshold for hyperglycemia

            # Calculate metrics for each event type, requiring at least 3 samples (15 mins)
            num_deep, dur_deep = self._calculate_excursion_metrics(pre_meal_glucose, deep_hypo_cond, 3)
            num_hypo, dur_hypo = self._calculate_excursion_metrics(pre_meal_glucose, hypo_cond, 3)
            num_hyper, dur_hyper = self._calculate_excursion_metrics(pre_meal_glucose, hyper_cond, 3)
            
            metrics[f'num_deep_hypo_events_{interval_name}'] = num_deep
            metrics[f'duration_deep_hypo_minutes_{interval_name}'] = dur_deep
            metrics[f'num_hypo_events_{interval_name}'] = num_hypo
            metrics[f'duration_hypo_minutes_{interval_name}'] = dur_hypo
            metrics[f'num_hyper_events_{interval_name}'] = num_hyper
            metrics[f'duration_hyper_minutes_{interval_name}'] = dur_hyper
            
            # Calculate the "Total Hypo" features by summing the deep and regular hypo metrics
            metrics[f'num_total_hypo_events_{interval_name}'] = num_deep + num_hypo
            metrics[f'duration_total_hypo_minutes_{interval_name}'] = dur_deep + dur_hypo
            
            
            # LBGI, HBGI, BGRI and ADRR Calculation
            glucose_for_risk = pre_meal_glucose.dropna()
            if not glucose_for_risk.empty:
                # Keep only positive glucose values for the log
                glucose_for_risk = glucose_for_risk[glucose_for_risk > 0]
                if not glucose_for_risk.empty:
                    # Calculate f(g)
                    f_g = 1.509 * (np.log(glucose_for_risk)**1.084 - 5.381)
                    # Calculate r(g)
                    r_g = 10 * (f_g**2)
                    
                    # Calculate LBGI
                    low_risk_values = r_g[f_g < 0]
                    if not low_risk_values.empty:
                        metrics[f'lbgi_{interval_name}'] = low_risk_values.mean()
                        
                    # Calculate HBGI
                    high_risk_values = r_g[f_g > 0]
                    if not high_risk_values.empty:
                        metrics[f'hbgi_{interval_name}'] = high_risk_values.mean()
                    
                    # Calculate BGRI as LBGI + HBGI 
                    if pd.notna(metrics.get(f'lbgi_{interval_name}')) and pd.notna(metrics.get(f'hbgi_{interval_name}')):
                        metrics[f'bgri_{interval_name}'] = metrics[f'lbgi_{interval_name}'] + metrics[f'hbgi_{interval_name}']

                    # Calculate ADRR from the same risk values [cite: 311]
                    if not low_risk_values.empty and not high_risk_values.empty:
                        metrics[f'ADRR_{interval_name}'] = low_risk_values.max() + high_risk_values.max()
            
            # MAGE Calculation
            metrics[f'MAGE_{interval_name}'] = self._calculate_mage(pre_meal_glucose)

            # J-Index Calculation 
            if pre_meal_glucose.mean() > 0 and pre_meal_glucose.std() > 0:
                metrics[f'J_index_{interval_name}'] = 0.001 * (pre_meal_glucose.mean() + pre_meal_glucose.std())**2

            
        return metrics
    
    
    def _calculate_hr_features_for_window(self, heart_rate_df, start_time, end_time, personalized_hr_threshold):
        """
        Calculates descriptive statistics and dynamics for the heart rate
        in the postprandial window.
        """
        features = {
            'hr_mean': np.nan,
            'hr_std': np.nan,
            'hr_min': np.nan,
            'hr_max': np.nan,
            'hr_range': np.nan,
            'hr_mean_derivative': np.nan,
            'hr_variability': np.nan, # RMSSD =  Root Mean Square of Successive Differences 
        }

        if heart_rate_df is None or heart_rate_df.empty:
            return features
        
        # Select the data within the postprandial window
        mask = (heart_rate_df.index >= start_time) & (heart_rate_df.index <= end_time) # created a mask because the timestamps of the glucose are not perfectly aligned with the timestamps of the heart rate
        hr_window = heart_rate_df.loc[mask]['HEART_RATE'].dropna()

        if len(hr_window) < 2: # Need at least 2 points for stats
            return features
        
        # Basic statistics
        features['hr_mean'] = hr_window.mean()
        features['hr_std'] = hr_window.std()
        features['hr_min'] = hr_window.min()
        features['hr_max'] = hr_window.max()
        features['hr_range'] = features['hr_max'] - features['hr_min']
        
        # Variability (RMSSD)
        successive_diffs = np.diff(hr_window.values)
        if len(successive_diffs) > 0:
            features['hr_variability'] = np.sqrt(np.mean(successive_diffs ** 2))
            
       # HR Abs Mean Derivative
        time_diffs_minutes = hr_window.index.to_series().diff().dt.total_seconds() / 60.0
        hr_diffs = hr_window.diff()
        derivatives = hr_diffs.div(time_diffs_minutes).replace([np.inf, -np.inf], np.nan).dropna()
        if not derivatives.empty:
            features['hr_mean_derivative'] = derivatives.abs().mean()

        return features
    
    
    def _calculate_pre_meal_hr_metrics(self, heart_rate_df, meal_start_time, first_hr_timestamp, personalized_hr_threshold):
        """
        Calculates heart rate metrics for the specified pre-meal intervals.
        """
        metrics = {}
        if heart_rate_df is None or heart_rate_df.empty:
            return metrics

        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            interval_start = meal_start_time - pd.Timedelta(hours=interval_hours)
            interval_end = meal_start_time - pd.Timedelta(minutes=self.SAMPLING_INTERVAL_MINUTES)
            
            interval_start = max(interval_start, first_hr_timestamp)

            keys = [f'hr_mean_{interval_name}', f'hr_std_{interval_name}', f'hr_min_{interval_name}',
                    f'hr_max_{interval_name}', f'hr_range_{interval_name}', f'hr_mean_derivative_{interval_name}',
                    f'hr_variability_{interval_name}', 
            ]
            
            for key in keys:
                metrics[key] = np.nan

            mask = (heart_rate_df.index >= interval_start) & (heart_rate_df.index <= interval_end)
            hr_pre_meal_window = heart_rate_df.loc[mask]['HEART_RATE'].dropna()

            if len(hr_pre_meal_window) < 2:
                continue

            metrics[f'hr_mean_{interval_name}'] = hr_pre_meal_window.mean()
            metrics[f'hr_std_{interval_name}'] = hr_pre_meal_window.std()
            metrics[f'hr_min_{interval_name}'] = hr_pre_meal_window.min()
            metrics[f'hr_max_{interval_name}'] = hr_pre_meal_window.max()
            metrics[f'hr_range_{interval_name}'] = metrics[f'hr_max_{interval_name}'] - metrics[f'hr_min_{interval_name}']

            # HR Variability
            successive_diffs = np.diff(hr_pre_meal_window.values)
            if len(successive_diffs) > 0:
                metrics[f'hr_variability_{interval_name}'] = np.sqrt(np.mean(successive_diffs ** 2))
            
            # Abs Mean Derivative Calculation 
            time_diffs_minutes = hr_pre_meal_window.index.to_series().diff().dt.total_seconds() / 60.0
            hr_diffs = hr_pre_meal_window.diff()
            derivatives = hr_diffs.div(time_diffs_minutes).replace([np.inf, -np.inf], np.nan).dropna()
            if not derivatives.empty:
                metrics[f'hr_mean_derivative_{interval_name}'] = derivatives.abs().mean()

        return metrics
    
    
    
    def _extract_pulse_ox_features(self, pulse_ox_df, start_time, end_time):
        """
        Calculates descriptive statistics for Pulse Ox in the postprandial window.
        """
        features = {'pulse_ox_mean': np.nan, 'pulse_ox_std': np.nan, 'pulse_ox_min': np.nan, 'pulse_ox_max': np.nan, 'pulse_ox_range': np.nan}
        if pulse_ox_df is None or pulse_ox_df.empty:
            return features
        
        mask = (pulse_ox_df.index >= start_time) & (pulse_ox_df.index <= end_time)
        pulse_ox_window = pulse_ox_df.loc[mask]['PULSE_OX'].dropna()

        if len(pulse_ox_window) < 2:
            return features

        features['pulse_ox_mean'] = pulse_ox_window.mean()
        features['pulse_ox_std'] = pulse_ox_window.std()
        features['pulse_ox_min'] = pulse_ox_window.min()
        features['pulse_ox_max'] = pulse_ox_window.max()
        features['pulse_ox_range'] = features['pulse_ox_max'] - features['pulse_ox_min']
        
        return features
    

    def _calculate_pre_meal_pulse_ox_metrics(self, pulse_ox_df, meal_start_time, first_pulse_ox_timestamp):
        """
        Calculates Pulse Ox metrics for the specified pre-meal intervals.
        """
        metrics = {}
        if pulse_ox_df is None or pulse_ox_df.empty:
            return metrics

        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            keys = [f'pulse_ox_mean_{interval_name}', f'pulse_ox_std_{interval_name}', f'pulse_ox_min_{interval_name}', f'pulse_ox_max_{interval_name}', f'pulse_ox_range_{interval_name}']
            for key in keys:
                metrics[key] = np.nan

            interval_start = meal_start_time - pd.Timedelta(hours=interval_hours)
            interval_end = meal_start_time - pd.Timedelta(minutes=self.SAMPLING_INTERVAL_MINUTES)
            interval_start = max(interval_start, first_pulse_ox_timestamp)

            mask = (pulse_ox_df.index >= interval_start) & (pulse_ox_df.index <= interval_end)
            pulse_ox_pre_meal_window = pulse_ox_df.loc[mask]['PULSE_OX'].dropna()

            if len(pulse_ox_pre_meal_window) < 2:
                continue

            metrics[f'pulse_ox_mean_{interval_name}'] = pulse_ox_pre_meal_window.mean()
            metrics[f'pulse_ox_std_{interval_name}'] = pulse_ox_pre_meal_window.std()
            metrics[f'pulse_ox_min_{interval_name}'] = pulse_ox_pre_meal_window.min()
            metrics[f'pulse_ox_max_{interval_name}'] = pulse_ox_pre_meal_window.max()
            metrics[f'pulse_ox_range_{interval_name}'] = metrics[f'pulse_ox_max_{interval_name}'] - metrics[f'pulse_ox_min_{interval_name}']
        
        return metrics
    
    
    # def _calculate_daily_steps(self, steps_df):
    #     """
    #     Calculates the total sum of steps for each day.
    #     """
    #     if steps_df is None or steps_df.empty:
    #         return None
    #     # Resample by calendar day ('D') and sum the steps
    #     daily_steps = steps_df['STEPS'].resample('D').max()
    #     return daily_steps
    
    
    def _calculate_physical_activity_for_window(self, hr_df, steps_df, start_time, end_time, hr_rest, hrr):
        """
        Detects physical activity and classifies its intensity based on combined HRR and step cadence.
        """
        # Initialize results with NaN for metrics that are undefined when there is no activity
        activity_metrics = {
            'activity_proportion': np.nan,
            'light_proportion': np.nan,
            'moderate_proportion': np.nan,
            'vigorous_proportion': np.nan,
            'mean_cadence_when_active': np.nan,
            'mean_cadence_light': np.nan,     
            'mean_cadence_moderate': np.nan,  
            'mean_cadence_vigorous': np.nan 
        }

        if hr_df is None or steps_df is None or hr_df.empty or steps_df.empty:
            return activity_metrics

        # Slice data for the specified window
        # Create a boolean mask to select all rows within the time range
        hr_mask = (hr_df.index >= start_time) & (hr_df.index <= end_time)
        hr_window = hr_df.loc[hr_mask]['HEART_RATE'].dropna()
        
        steps_mask = (steps_df.index >= start_time) & (steps_df.index <= end_time)
        steps_window = steps_df.loc[steps_mask]['STEPS'].dropna()

        if hr_window.empty or steps_window.empty or len(hr_window) < 2 or len(steps_window) < 2:
            return activity_metrics

        # Resample to 1-minute frequency for alignment
        hr_resampled = hr_window.resample('1min').ffill() # resample('1min') ---> resamples with a 1 minute frequence; if there's no original data in the one minute time slot, it puts NaN
                                                          # ffill() ---> forward fill: if there's a NaN value after the resampling, it uses the last aviable value
        steps_resampled = steps_window.resample('1min').ffill()
        
        # Calculate cadence (steps per minute)
        cadence = steps_resampled.diff()
        cadence[cadence < 0] = 0 # Prevent negative steps from counter resets (when the time window crosses midnight)

        # Merge the two series on their common timestamp index
        combined_df = pd.DataFrame({'HR': hr_resampled, 'Cadence': cadence}).dropna() # creates a new perfectly aligned DataFrame with Heart Rate and Cadence columns
                                                                                      # .dropna() to remove the first row where cadence is NaN
        if combined_df.empty:
            return activity_metrics
        
        # Define HR thresholds for each intensity level using Heart Rate Reserve
        thresholds = {
            'light': (hrr * self.HR_ACTIVITY_HRR_INTENSITIES['light'][0]) + hr_rest,
            'moderate': (hrr * self.HR_ACTIVITY_HRR_INTENSITIES['moderate'][0]) + hr_rest,
            'vigorous': (hrr * self.HR_ACTIVITY_HRR_INTENSITIES['vigorous'][0]) + hr_rest
        }
        
        # Create a mask to identify any movement
        is_moving_mask = combined_df['Cadence'] > self.MIN_CADENCE_FOR_ACTIVITY

        # Create masks for each intensity level, requiring movement
        is_light_mask = (combined_df['HR'] >= thresholds['light']) & (combined_df['HR'] < thresholds['moderate']) & is_moving_mask
        is_moderate_mask = (combined_df['HR'] >= thresholds['moderate']) & (combined_df['HR'] < thresholds['vigorous']) & is_moving_mask
        is_vigorous_mask = (combined_df['HR'] >= thresholds['vigorous']) & is_moving_mask
        
        # Combine all levels to find any active period
        is_active_mask = is_light_mask | is_moderate_mask | is_vigorous_mask
        
        if is_active_mask.any():
            # Calculate proportion for each intensity level and the total
            activity_metrics['light_proportion'] = is_light_mask.mean() * 100
            activity_metrics['moderate_proportion'] = is_moderate_mask.mean() * 100
            activity_metrics['vigorous_proportion'] = is_vigorous_mask.mean() * 100
            activity_metrics['activity_proportion'] = is_active_mask.mean() * 100
            
            # Calculate overall mean cadence (only during active periods)
            active_periods = combined_df[is_active_mask]
            activity_metrics['mean_cadence_when_active'] = active_periods['Cadence'].mean()
            
            # Calculate mean cadence for each intensity level 
            if is_light_mask.any():
                activity_metrics['mean_cadence_light'] = combined_df[is_light_mask]['Cadence'].mean()
            if is_moderate_mask.any():
                activity_metrics['mean_cadence_moderate'] = combined_df[is_moderate_mask]['Cadence'].mean()
            if is_vigorous_mask.any():
                activity_metrics['mean_cadence_vigorous'] = combined_df[is_vigorous_mask]['Cadence'].mean()

        return activity_metrics


    def _extract_pre_meal_activity_features(self, hr_df, steps_df, meal_start_time, hr_rest, hrr):
        """
        A wrapper function to calculate activity features for all pre-meal intervals
        by calling the generic window calculation method.
        """
        metrics = {}
        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            interval_start = meal_start_time - pd.Timedelta(hours=interval_hours)
            interval_end = meal_start_time

            # Call the generic function for each pre-meal window
            activity_metrics = self._calculate_physical_activity_for_window(
                hr_df, steps_df, interval_start, interval_end, hr_rest, hrr
            )
            
            # Add the results to the metrics dictionary with the correct prefix
            for key, value in activity_metrics.items():
                metrics[f'{key}_{interval_name}'] = value
        
        return metrics
 
    
    def extract_features(self):
        """
        Carica il database post-prandiale, estrae le feature e salva il DataFrame aggiornato.

        Args:
           
        Returns:
            pd.DataFrame: Il DataFrame aggiornato con le feature estratte.
        """
        input_file_path = os.path.join(self.data_dir_db, self.postprandial_data_file)
        try:
            all_postprandial_df = pd.read_csv(input_file_path)
            all_postprandial_df['meal_time_start'] = pd.to_datetime(all_postprandial_df['meal_time_start'])
            all_postprandial_df['meal_time_end'] = pd.to_datetime(all_postprandial_df['meal_time_end'])
            print(f"Caricato {self.postprandial_data_file} per l'estrazione delle feature.")
        except FileNotFoundError:
            print(f"Errore: File {self.postprandial_data_file} non trovato a {input_file_path}. Eseguire prima DatasetPreparer.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Errore durante il caricamento di {self.postprandial_data_file}: {e}")
            return pd.DataFrame()

        # Inserire nome nuove feature qui:
        base_columns = [
            # GLUCOSE basic features
            'postprandial_hypo', 'postprandial_hypo_time', 'postprandial_hypo_duration_minutes', 'postprandial_deep_hypo', 'postprandial_deep_hypo_time', 'postprandial_deep_hypo_duration_minutes',
            'first_glucose_peak_value_after_meal', 'time_to_first_glucose_peak_after_meal', 'second_glucose_peak_value_after_meal', 'time_to_second_glucose_peak_after_meal',
            'meal_hour', 'meal_day_of_week', 'glucose_at_meal',
            'glucose_at_first_peak', 'rate_of_increase', 'glucose_derivative_15min_before_peak1', 'glucose_derivative_15min_after_peak1', 'glucose_value_15min_after_peak1',
            'glucose_acceleration_15min_before_peak1', 'glucose_acceleration_15min_after_peak1',
            'initial_drop_magnitude',
            'mean_descent_rate_15min_post_peak', 'max_descent_rate_15min_post_peak',       
            'auc_glucose_rise',
            'time_since_previous_meal_minutes', 'time_peak_hypo_event', 'target_hypoglycemia', 'hypo_to_check_flag',
            
            'meal_category', 

            # NEW GLUCOSE features
            'postprandial_mean_glucose', 'postprandial_std_glucose', 'postprandial_range',
            
            # New columns for postprandial Time-In-Range (TIR)
            'postprandial_tir_target', 'postprandial_tir_l1_hyper', 'postprandial_tir_l2_hyper', 'postprandial_tir_l1_hypo', 'postprandial_tir_l2_hypo',
            
            # REGRESSION NEW COLUMNS
            'is_saturated_at_39', # Flag for sensor saturation events
            'postprandial_glucose_min', # New target for regression analysis, calculated from the interpolated series
            'was_interpolated', # Interpolation flag (to show if the original data was modified)
            'postprandial_glucose_min_original', # Column to preserve the original minimum value read by the sensor
            
            # NEW SHORT-TIME FEATURES
            'lastROC',
            'meanROC',
            
            # HEART_RATE features
            'hr_mean_15min_post_peak', 'hr_std_15min_post_peak', 'hr_min_15min_post_peak', 'hr_max_15min_post_peak', 'hr_range_15min_post_peak', 'hr_mean_derivative_15min_post_peak', 'hr_variability_15min_post_peak', 
            'hr_mean_later_post_peak', 'hr_std_later_post_peak', 'hr_min_later_post_peak', 'hr_max_later_post_peak', 'hr_range_later_post_peak', 'hr_mean_derivative_later_post_peak', 'hr_variability_later_post_peak', 
            
            # PULSE_OX features
            'pulse_ox_mean', 'pulse_ox_std', 'pulse_ox_min', 'pulse_ox_max', 'pulse_ox_range',
            
            # STEPS features 
            'steps_1h_pre_meal', 'steps_2h_pre_meal', 'steps_3h_pre_meal', 'steps_4h_pre_meal',
            'steps_during_rise',
            
            # PHYSICAL ACTIVITY features
            'activity_proportion_during_rise', 'mean_cadence_when_active_during_rise',
            'light_proportion_during_rise', 'moderate_proportion_during_rise', 'vigorous_proportion_during_rise',
            'mean_cadence_light_during_rise', 'mean_cadence_moderate_during_rise', 'mean_cadence_vigorous_during_rise',
            'activity_proportion_until_nadir', 'mean_cadence_when_active_until_nadir',
            'light_proportion_until_nadir', 'moderate_proportion_until_nadir', 'vigorous_proportion_until_nadir', 
            'mean_cadence_light_until_nadir', 'mean_cadence_moderate_until_nadir', 'mean_cadence_vigorous_until_nadir',
            
            # NEW ACTIVITY BOUT & TIMING FEATURES
            'number_of_activity_bouts_during_rise',
            'mean_bout_duration_during_rise',
            'max_bout_duration_during_rise',
            'time_from_meal_to_first_activity',
            'activity_proportion_before_peak',
            'activity_proportion_after_peak',

            # NEW INTERACTION FEATURES
            'activity_x_pre_meal_glucose',
            'activity_x_rate_of_increase',
        ]
        
        pre_meal_glucose_cols = []
        pre_meal_hr_cols = []
        pre_meal_pulse_ox_cols = []
        pre_meal_activity_cols = []
        # Aggiungi le colonne per le metriche pre-pasto dinamicamente
        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            pre_meal_glucose_cols.extend([
                f'mean_glucose_{interval_name}', f'std_glucose_{interval_name}', f'cv_glucose_{interval_name}', f'min_glucose_{interval_name}', f'max_glucose_{interval_name}', f'range_{interval_name}',
                f'prop_above_130_{interval_name}', f'prop_80_130_{interval_name}', f'prop_50_80_{interval_name}', f'prop_below_50_{interval_name}',
                f'num_positive_peaks_{interval_name}', f'num_negative_peaks_{interval_name}', 
                
                f'num_deep_hypo_events_{interval_name}', f'duration_deep_hypo_minutes_{interval_name}',
                f'num_hypo_events_{interval_name}', f'duration_hypo_minutes_{interval_name}',
                f'num_hyper_events_{interval_name}', f'duration_hyper_minutes_{interval_name}',
                f'num_total_hypo_events_{interval_name}', f'duration_total_hypo_minutes_{interval_name}',
                
                f'MAGE_{interval_name}',
                f'J_index_{interval_name}',
                f'lbgi_{interval_name}', f'hbgi_{interval_name}',
                f'bgri_{interval_name}',
                f'ADRR_{interval_name}',

            ])
            pre_meal_hr_cols.extend([ 
                f'hr_mean_{interval_name}', f'hr_std_{interval_name}', f'hr_min_{interval_name}',  f'hr_max_{interval_name}',  f'hr_range_{interval_name}', 
                #f'hr_slope_{interval_name}', 
                f'hr_mean_derivative_{interval_name}', f'hr_variability_{interval_name}', 
                #f'hr_prop_time_active_{interval_name}' 
            ])
            pre_meal_pulse_ox_cols.extend([
                f'pulse_ox_mean_{interval_name}', f'pulse_ox_std_{interval_name}', f'pulse_ox_min_{interval_name}', f'pulse_ox_max_{interval_name}', f'pulse_ox_range_{interval_name}'
            ])
            pre_meal_activity_cols.extend([
                f'activity_proportion_{interval_name}', f'mean_cadence_when_active_{interval_name}',
                f'light_proportion_{interval_name}', f'moderate_proportion_{interval_name}', f'vigorous_proportion_{interval_name}',
                f'mean_cadence_light_{interval_name}', f'mean_cadence_moderate_{interval_name}', f'mean_cadence_vigorous_{interval_name}'
            ])

        new_columns = base_columns + pre_meal_glucose_cols + pre_meal_hr_cols + pre_meal_pulse_ox_cols + pre_meal_activity_cols
        
        # Find which columns are truly new and add them all at once
        existing_columns = all_postprandial_df.columns
        columns_to_add = [col for col in new_columns if col not in existing_columns]
        
        if columns_to_add:
            # This is the efficient way to add many new columns, preventing fragmentation
            all_postprandial_df = all_postprandial_df.reindex(columns=existing_columns.tolist() + columns_to_add)
        
        
        # Explicitly set the dtype for the meal_category column to 'object' to allow strings
        if 'meal_category' in all_postprandial_df.columns:
            all_postprandial_df['meal_category'] = all_postprandial_df['meal_category'].astype('object')


        subjects_glucose_data = {} # initialize the glucos data
        subjects_wearable_data = {} # initialize the wearable data
        subjects_daily_steps_data = {} # initialize the daily steps
        subjects_hr_thresholds = {} # initialize the subjects' threshold

        old_subj = ''
        
        # Explicitly initialize the flag columns to 0 before the loop
        # This prevents any possibility of old values or NaNs causing a mismatch
        all_postprandial_df['is_saturated_at_39'] = 0
        all_postprandial_df['was_interpolated'] = 0


        print("--> Estrazione feature CGM + pasto <--")
        
        for index, row in all_postprandial_df.iterrows():
            
            subj_id = row['subj_id']
            meal_start = row['meal_time_start']
            meal_end = row['meal_time_end']

            # Carica dati glucosio del soggetto
            if subj_id not in subjects_glucose_data:
                glucose_df_subj = self._load_glucose_df_for_subject(subj_id)

                if glucose_df_subj is None or 'glucose_filtered' not in glucose_df_subj.columns:
                    print(f"Salto pasto per {subj_id} a {meal_start}: dati glucosio filtrati non disponibili.")
                    continue
               
                # Interpolate raw data, then filter (do this once per subject)
                # Step 1: Interpolate the raw glucose signal
                glucose_interpolated_raw, _ = interpolate_saturation_events(
                    glucose_series_to_interpolate=glucose_df_subj['glucose'],
                    glucose_series_for_detection=glucose_df_subj['glucose']
                )
                
                # Step 2: Apply the low-pass filter to the newly interpolated raw signal
                glucose_final_values = apply_zero_phase_low_pass_filter(
                    glucose_interpolated_raw.values,
                    self.sampling_rate_seconds,
                    self.cutoff_frequency_hz,
                    self.filter_order
                )
                
                # Step 3: Add this definitive series to the subject's DataFrame for use in the loop
                glucose_df_subj['glucose_final_for_features'] = pd.Series(glucose_final_values, index=glucose_df_subj.index)
                
                # FIX for NaN gaps 
                # If the new series has NaNs where the old one had valid data (due to failed interpolation),
                # patch the gaps using the original filtered data as a fallback
                glucose_df_subj['glucose_final_for_features'] = glucose_df_subj['glucose_final_for_features'].fillna(glucose_df_subj['glucose_filtered'])
                
                
                # Save the updated DataFrame with the final glucose series 
                # This is saved back to the individual subject's folder, adding to the file that was created by the DatasetPreparer (_tma_processed_new) the new column glucose_final_for_features
                output_path = os.path.join(self.data_dir, subj_id, self.glucose_filename_pattern.format(subj_id))
                glucose_df_subj.to_csv(output_path)
                
 
                subjects_glucose_data[subj_id] = glucose_df_subj

            glucose_df_subj = subjects_glucose_data[subj_id]

            if 'glucose_filtered' not in glucose_df_subj.columns or 'glucose_final_for_features' not in glucose_df_subj.columns:
                print(f"Salto pasto per {subj_id} a {meal_start}: colonne glucosio necessarie non disponibili.")
                continue

            # considero le tre serie di interesse: CGM, CGM filtrato e derivata del CGM
            glucose_series = glucose_df_subj['glucose'] # Unfiltered, original data
            glucose_series_filtered = glucose_df_subj['glucose_filtered'] # Filtered data
            
            
            # This is the new, definitive series for ALL feature calculations
            glucose_series_for_features = glucose_df_subj['glucose_final_for_features']

            
            glucose_derivative_series = glucose_df_subj['glucose_derivative']
            glucose_second_derivative_series = glucose_df_subj['glucose_second_derivative']
            
            # Ottieni il primo timestamp disponibile per i dati di glucosio del soggetto
            first_glucose_timestamp = glucose_series_for_features.index.min()
            
            if pd.isna(first_glucose_timestamp):
                 print(f"Salto pasto per {subj_id} a {meal_start}: Dati glucosio vuoti.")
                 continue


            #-----------------------------------------------------------------
            # 1. Target: Rilevamento Ipoglicemia
            #-----------------------------------------------------------------
            is_hypo, hypo_time, hypo_duration, is_deep_hypo, deep_hypo_time, deep_hypo_duration = \
                self._detect_hypoglycemia(glucose_series_for_features, glucose_derivative_series, meal_start, meal_end)
            
            all_postprandial_df.at[index, 'postprandial_hypo'] = is_hypo
            all_postprandial_df.at[index, 'postprandial_hypo_time'] = hypo_time
            all_postprandial_df.at[index, 'postprandial_hypo_duration_minutes'] = hypo_duration
            all_postprandial_df.at[index, 'postprandial_deep_hypo'] = is_deep_hypo
            all_postprandial_df.at[index, 'postprandial_deep_hypo_time'] = deep_hypo_time
            all_postprandial_df.at[index, 'postprandial_deep_hypo_duration_minutes'] = deep_hypo_duration

            # Encoding del target
            target_val = 0 # Default: nessun evento
            if is_hypo: 
                target_val = 1
            if is_deep_hypo: 
                target_val = 2
                
            
            all_postprandial_df.at[index, 'target_hypoglycemia'] = target_val
            
            #-----------------------------------------------------------------
            # 2. Ricerca Picchi
            #-----------------------------------------------------------------
            peak1_val, time_to_peak1, peak2_val, time_to_peak2 = \
                self._find_postprandial_peaks(glucose_series_for_features, meal_start, meal_end)
            
            all_postprandial_df.at[index, 'first_glucose_peak_value_after_meal'] = peak1_val
            all_postprandial_df.at[index, 'time_to_first_glucose_peak_after_meal'] = time_to_peak1
            all_postprandial_df.at[index, 'second_glucose_peak_value_after_meal'] = peak2_val
            all_postprandial_df.at[index, 'time_to_second_glucose_peak_after_meal'] = time_to_peak2
            
            # Rimuovo secondo picco se avviene dopo hypo o deep_hypo 
            hypo_time_relative = (hypo_time - meal_start).total_seconds() / 60 if pd.notna(hypo_time) else np.nan
            deep_hypo_time_relative = (deep_hypo_time - meal_start).total_seconds() / 60 if pd.notna(deep_hypo_time) else np.nan

            if is_hypo and pd.notna(time_to_peak2):
                if time_to_peak2 > hypo_time_relative:
                    all_postprandial_df.at[index, 'second_glucose_peak_value_after_meal'] = np.nan
                    all_postprandial_df.at[index, 'time_to_second_glucose_peak_after_meal'] = np.nan
            
            if is_deep_hypo and pd.notna(time_to_peak2):
                if time_to_peak2 > deep_hypo_time_relative:
                    all_postprandial_df.at[index, 'second_glucose_peak_value_after_meal'] = np.nan
                    all_postprandial_df.at[index, 'time_to_second_glucose_peak_after_meal'] = np.nan

            #------------------------------------------------------------------
            # Calcolo distanza picco-evento e identificazione di casi particolari
            #------------------------------------------------------------------
            # Questo servirà per evitare di prendere ipo (o deep) che si 
            # verificano entro i 15 min dal picco.
            # attenzione qui perchè devo calcolare la distanza considerando 
            # picco+15 min in quanto estraggo delle feature anche dai 15 min 
            # successivi al picco per calcolare la derivata (vedi 9-10-11)
            
            if target_val == 1: # the meal event was classified as regular hypoglycemia
                time_event = hypo_time
                
            if target_val == 2: # the meal event was classified as deep hypoglycemia
                time_event = deep_hypo_time
            
            if target_val != 0:
                
                # se hypo_to_check_flag = 1 va controllato che non sia artefatto da compressione
                if np.isnan(time_to_peak1): # if time_to_peak1 is NaN, it means no clear post-prandial glucose peak was identified for this meal
                    
                    all_postprandial_df.at[index, 'hypo_to_check_flag'] = 1 
                    
                    # plt.figure(), 
                    # plt.plot(glucose_df_subj['glucose'].loc[meal_start:meal_end], '-o')
                    # plt.plot(glucose_df_subj['glucose_filtered'].loc[meal_start:meal_end], '-o')
                    # plt.suptitle(f"Soggetto: {subj_id}, Inizio Pasto: {meal_start.strftime('%Y-%m-%d %H:%M')}", fontsize=16)
                    
                    # questo serve come flag per filtrare questi eventi
                    all_postprandial_df.at[index, 'time_peak_hypo_event'] =  -1 # this distinct value indicates that hypoglycemia occurred but its relationship to a peak cannot be determined due to the absence of a detectable peak
                else:
                    # The peak_time variable in this line is not representing the exact moment of the peak. Instead, it's defining a "critical cutoff point" that is 15 minutes after the actual peak
                    peak_time = meal_start + pd.Timedelta(minutes=time_to_peak1) + pd.Timedelta(minutes=15) # + pd.Timedelta(minutes=15) ---> adds an additional 15 minutes to the peak time because features are extracted for 15 minutes after the peak, so, the "critical window" for determining if hypoglycemia is too close to the peak extends 15 minutes beyond the peak itself
                    time_peak_hypo = (time_event - peak_time).total_seconds() / 60
                    # se zero, non usarlo per trainare il modello, probabilmente è hypo di rebound
                    if time_peak_hypo == 0:
                        all_postprandial_df.at[index, 'hypo_to_check_flag'] = 1 
                    else:
                        all_postprandial_df.at[index, 'time_peak_hypo_event'] = time_peak_hypo

            if target_val == 0:
                all_postprandial_df.at[index, 'time_peak_hypo_event'] =  0
                
            #------------------------------------------------------------------
            # 3. Ora del pasto (0-23) 
            #------------------------------------------------------------------
            all_postprandial_df.at[index, 'meal_hour'] = meal_start.hour
           
            #------------------------------------------------------------------ 
            # 4. Giorno della settimana (0=lunedì, 6=domenica)
            #------------------------------------------------------------------
            all_postprandial_df.at[index, 'meal_day_of_week'] = meal_start.dayofweek
            
            #------------------------------------------------------------------
            # 5. Glucosio al pasto
            #------------------------------------------------------------------
            # Use the interpolated series for the starting glucose value
            glucose_at_meal_start = glucose_series_for_features.get(meal_start, np.nan)
            #glucose_at_meal_start = glucose_series_filtered.get(meal_start, np.nan)
            all_postprandial_df.at[index, 'glucose_at_meal'] = glucose_at_meal_start
            
            #------------------------------------------------------------------
            # 6. Glucosio al picco ( peak1_val)
            #------------------------------------------------------------------
            all_postprandial_df.at[index, 'glucose_at_first_peak'] = peak1_val
           
            #------------------------------------------------------------------
            # 7. Rate of Increase
            #------------------------------------------------------------------
            rate_of_rise = np.nan
            if pd.notna(peak1_val) and pd.notna(glucose_at_meal_start) and pd.notna(time_to_peak1):
                if time_to_peak1 > 0: # Evita divisione per zero
                    rate_of_rise = (peak1_val - glucose_at_meal_start) / time_to_peak1
                elif peak1_val == glucose_at_meal_start: # Se il picco è subito all'inizio e non cambia, rate = 0
                    rate_of_rise = 0
            all_postprandial_df.at[index, 'rate_of_increase'] = rate_of_rise
            
            #------------------------------------------------------------------
            # 8-9-10. derivata 15 min pre e post picco + glucosio 15 min dopo il picco
            #------------------------------------------------------------------
            deriv_15min_before_peak1 = np.nan
            deriv_15min_after_peak1 = np.nan
            glucose_15min_after_peak1 = np.nan
            accel_15min_before_peak1 = np.nan
            accel_15min_after_peak1 = np.nan
            initial_drop_magnitude = np.nan
            mean_descent_rate = np.nan
            max_descent_rate = np.nan
            auc_rise = np.nan
            
            steps_during_rise = np.nan

            if pd.notna(time_to_peak1):
                
                # Calcola il timestamp del picco
                peak1_timestamp = meal_start + pd.Timedelta(minutes=time_to_peak1)
                
                # Timestamp 15 minuti prima del picco
                time_15min_before_peak1 = peak1_timestamp - pd.Timedelta(minutes=15)
                
                # Assicurati che non si vada prima dell'inizio del pasto
                time_15min_before_peak1 = max(time_15min_before_peak1, meal_start)
                
                # Timestamp 15 minuti dopo il picco
                time_15min_after_peak1 = peak1_timestamp + pd.Timedelta(minutes=15)
                
                # Assicurati che non si vada oltre la fine del periodo post-prandiale
                time_15min_after_peak1 = min(time_15min_after_peak1, meal_end)

                # Derivata 15 minuti prima del picco
                
                # fill di piccoli gaps
                glucose_derivative_series_filled = glucose_derivative_series.interpolate(method='linear')
                
                deriv_15min_before_peak1 = glucose_derivative_series_filled.asof(time_15min_before_peak1) # 'asof' prende il valore più recente <= timestamp
                
                # Derivata 15 minuti dopo il picco
                deriv_15min_after_peak1 = glucose_derivative_series_filled.asof(time_15min_after_peak1) # 'asof' prende il valore più recente <= timestamp
                
                # Glucosio 15 minuti dopo il picco
                glucose_15min_after_peak1 = glucose_series_for_features.asof(time_15min_after_peak1) # 'asof' prende il valore più recente <= timestamp
                
                
                # Accelerazione 15 minuti prima e dopo il picco
                glucose_second_derivative_series_filled = glucose_second_derivative_series.interpolate(method='linear')
                accel_15min_before_peak1 = glucose_second_derivative_series_filled.asof(time_15min_before_peak1)
                accel_15min_after_peak1 = glucose_second_derivative_series_filled.asof(time_15min_after_peak1)
                
                
                # Initial Drop Magnitude
                if pd.notna(peak1_val) and pd.notna(glucose_15min_after_peak1):
                    initial_drop_magnitude = peak1_val - glucose_15min_after_peak1
                
                
                # Mean and Max Descent Rate in 15-min post-peak window
                descent_window_derivatives = glucose_derivative_series_filled.loc[peak1_timestamp:time_15min_after_peak1]
                if not descent_window_derivatives.empty:
                    mean_descent_rate = descent_window_derivatives.mean()
                    max_descent_rate = descent_window_derivatives.min() # .min() will give the most negative rate
                
                
                # AUC of Glucose Rise (Meal Start to Peak)
                # Use the interpolated series for a more accurate AUC
                rise_window_glucose = glucose_series_for_features.loc[meal_start:peak1_timestamp].dropna()
                #rise_window_glucose = glucose_series_filtered.loc[meal_start:peak1_timestamp].dropna()
                if len(rise_window_glucose) >= 2:
                    y_values = rise_window_glucose.values
                    # Convert time index to minutes from start for x-axis
                    x_values = (rise_window_glucose.index - meal_start).total_seconds() / 60
                    auc_rise = np.trapz(y=y_values, x=x_values)
                
                
                # Steps during rise window (from meal start to 15 mins after peak)
                if row.get('has_steps_data') == 1:
                    steps_df_subj = subjects_wearable_data.get(subj_id, {}).get('STEPS')
                    if steps_df_subj is not None and not steps_df_subj.empty:
                        
                        mask = (steps_df_subj.index >= meal_start) & (steps_df_subj.index <= time_15min_after_peak1)
                        rise_window_steps = steps_df_subj.loc[mask]['STEPS'].dropna()
                        
                        if len(rise_window_steps) >= 2:
                            # Calculate the difference between each consecutive point
                            diffs = rise_window_steps.diff()
                            # Sum only the positive differences, ignoring decreases from resets
                            steps_during_rise = diffs[diffs > 0].sum()
                            # # Calculate steps taken by subtracting min from max, for cumulative counters
                            # steps_during_rise = rise_window_steps.max() - rise_window_steps.min()
                        elif len(rise_window_steps) == 1:
                            steps_during_rise = 0 # If only one sample, no change in steps


            all_postprandial_df.at[index, 'glucose_derivative_15min_before_peak1'] = deriv_15min_before_peak1
            all_postprandial_df.at[index, 'glucose_derivative_15min_after_peak1'] = deriv_15min_after_peak1
            all_postprandial_df.at[index, 'glucose_value_15min_after_peak1'] = glucose_15min_after_peak1
            all_postprandial_df.at[index, 'glucose_acceleration_15min_before_peak1'] = accel_15min_before_peak1
            all_postprandial_df.at[index, 'glucose_acceleration_15min_after_peak1'] = accel_15min_after_peak1
            all_postprandial_df.at[index, 'initial_drop_magnitude'] = initial_drop_magnitude
            all_postprandial_df.at[index, 'mean_descent_rate_15min_post_peak'] = mean_descent_rate
            all_postprandial_df.at[index, 'max_descent_rate_15min_post_peak'] = max_descent_rate
            all_postprandial_df.at[index, 'auc_glucose_rise'] = auc_rise
            
            all_postprandial_df.at[index, 'steps_during_rise'] = steps_during_rise

            #------------------------------------------------------------------
            # 11. tempo da ultimo pasto
            #------------------------------------------------------------------
            
            
            if index > 0:    
                if old_subj == subj_id:
                    all_postprandial_df.at[index, 'time_since_previous_meal_minutes'] = (all_postprandial_df.at[index,'meal_time_start']-all_postprandial_df.at[index-1,'meal_time_start']).total_seconds() / 60
                else:
                    # primo pasto di un nuovo soggetto
                    all_postprandial_df.at[index, 'time_since_previous_meal_minutes'] = np.nan        
            else:
                # primo pasto del primo soggetto
                all_postprandial_df.at[index, 'time_since_previous_meal_minutes'] = np.nan
            
            old_subj = subj_id
            
            
            #------------------------------------------------------------------
            # 12. metriche glicemiche 1, 2, 3, 6, 12, 24 ore pre-pasto
            #------------------------------------------------------------------
            pre_meal_metrics = self._calculate_pre_meal_metrics(glucose_series_for_features, meal_start, first_glucose_timestamp)
            
            for key, value in pre_meal_metrics.items():
                all_postprandial_df.at[index, key] = value
 
            
            # NEW FEATURES ADDED 
            
            #------------------------------------------------------------------
            # 13. Postprandial descriptive statistics and Time-In-Range (TIR)
            # These features characterize the glucose profile during the postprandial period
            # + FROM CLASSIFICATION TO REGRESSION
            # + INTERPOLATION OF SATURATION EVENTS
            #------------------------------------------------------------------
            # --- SATURATION FLAG --- 
            # We check for saturation on the original (unfiltered) data to be certain
            # Define the consistent analysis window, starting 30 minutes after the meal_start
            analysis_start_time = meal_start + pd.Timedelta(minutes=30)
            
            # If the analysis start time is after the meal end time, the window is invalid
            # We skip this meal entirely, leaving its flags as 0 from the initialization
            if analysis_start_time >= meal_end:
                continue
            
            postprandial_glucose_original = glucose_series.loc[analysis_start_time:meal_end]
            
            if not postprandial_glucose_original.empty:
                # This logic will find consecutive blocks of '39'
                saturated_segments = (postprandial_glucose_original == 39).astype(int).groupby(
                    (postprandial_glucose_original == 39).astype(int).diff().ne(0).cumsum()
                ).filter(lambda x: len(x) >= 3 and x.iloc[0] == 1) # x.iloc[0] == 1 ---> the block must be made of 1s (saturation), not 0s
                
                # Set the flag to 1 only if a sustained segment (>= 3 samples) is found
                is_saturated = 1 if not saturated_segments.empty else 0
                all_postprandial_df.at[index, 'is_saturated_at_39'] = is_saturated
            else:
                all_postprandial_df.at[index, 'is_saturated_at_39'] = 0
            
            # --- WAS INTERPOLATED FLAG ---
            # A meal is considered "interpolated" if and only if it had a saturation event
            # This ensures counts match perfectly and avoids spill-over effects
            all_postprandial_df.at[index, 'was_interpolated'] = is_saturated
            
            
            # --- REGRESSION TARGET & POSTPRANDIAL METRICS ---
            # First, store the original minimum value (from original filtered data) for reference
            postprandial_glucose_filtered = glucose_series_filtered.loc[analysis_start_time:meal_end]
            if not postprandial_glucose_filtered.empty and not postprandial_glucose_filtered.isnull().all():
                all_postprandial_df.at[index, 'postprandial_glucose_min_original'] = postprandial_glucose_filtered.min()
            else:
                all_postprandial_df.at[index, 'postprandial_glucose_min_original'] = np.nan

            # Now, use the NEW FEATURE SERIES for all final calculations
            postprandial_glucose = glucose_series_for_features.loc[analysis_start_time:meal_end]
            
            # Calculate the final regression target from the new data
            if not postprandial_glucose.empty and not postprandial_glucose.isnull().all():
                all_postprandial_df.at[index, 'postprandial_glucose_min'] = postprandial_glucose.min()
            else:
                all_postprandial_df.at[index, 'postprandial_glucose_min'] = np.nan
                
            # Check if there is valid data in the window to perform calculations
            if not postprandial_glucose.empty and not postprandial_glucose.isnull().all():
                # Basic Stats (Media, SD, Range) 
                all_postprandial_df.at[index, 'postprandial_mean_glucose'] = postprandial_glucose.mean()
                all_postprandial_df.at[index, 'postprandial_std_glucose'] = postprandial_glucose.std()
                all_postprandial_df.at[index, 'postprandial_range'] = postprandial_glucose.max() - postprandial_glucose.min()
                
                # Time-In-Range (TIR) calculation
                total_samples_in_window = postprandial_glucose.notna().sum()
                if total_samples_in_window > 0:
                    tir_target = ((postprandial_glucose >= 70) & (postprandial_glucose <= 180)).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_target'] = tir_target
                    tir_l1_hyper = ((postprandial_glucose > 180) & (postprandial_glucose <= 250)).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l1_hyper'] = tir_l1_hyper
                    tir_l2_hyper = (postprandial_glucose > 250).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l2_hyper'] = tir_l2_hyper
                    tir_l1_hypo = ((postprandial_glucose >= 54) & (postprandial_glucose < 70)).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l1_hypo'] = tir_l1_hypo
                    tir_l2_hypo = (postprandial_glucose < 54).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l2_hypo'] = tir_l2_hypo
            
            
            #------------------------------------------------------------------
            # 14. Short-Time features on the predictive window
            #------------------------------------------------------------------
            if pd.notna(time_to_peak1):
                # Define the predictive window: 30 mins pre-meal to 15 mins post-peak
                predictive_window_start = meal_start - pd.Timedelta(minutes=30)
                peak_timestamp = meal_start + pd.Timedelta(minutes=time_to_peak1)
                predictive_window_end = peak_timestamp + pd.Timedelta(minutes=15)
               
                # Ensure the window does not go beyond the end of the postprandial period
                predictive_window_end = min(predictive_window_end, meal_end)
                
                # Use the interpolated series for ROC calculation
                predictive_glucose = glucose_series_for_features.loc[predictive_window_start:predictive_window_end]
                #predictive_glucose = glucose_series_filtered.loc[predictive_window_start:predictive_window_end]
                
                short_time_feats = self._calculate_short_time_features(predictive_glucose)
                for key, value in short_time_feats.items():
                    all_postprandial_df.at[index, key] = value
            
            #------------------------------------------------------------------
           
    
            
            # WEARABLE FEATURES    
            
            # Load wearable data once per subject
            if subj_id not in subjects_wearable_data:
                subjects_wearable_data[subj_id] = self._load_all_wearable_data_for_subject(subj_id)
           
            
            #------------------------------------------------------------------
            # 15. HR Features
            #------------------------------------------------------------------
            # Extraction of HR features in two windows: from meal_start until 15 mins after the first glucose peak, and from 15 mins after the first glucose peak to the meal_end
            if row.get('has_heart_rate_data') == 1:
                hr_df_subj = subjects_wearable_data.get(subj_id, {}).get('HEART_RATE')
                steps_df_subj = subjects_wearable_data.get(subj_id, {}).get('STEPS')

                if hr_df_subj is not None and not hr_df_subj.empty:

                    # Calculate and cache personalized thresholds once per subject
                    if subj_id not in subjects_hr_thresholds:
                        hr_rest = hr_df_subj['HEART_RATE'].quantile(0.025)
                        hr_max = hr_df_subj['HEART_RATE'].quantile(0.975)
                        hrr = hr_max - hr_rest
                        subjects_hr_thresholds[subj_id] = {'hr_rest': hr_rest, 'hrr': hrr}
                    
                    hr_params = subjects_hr_thresholds[subj_id]
                    hr_rest = hr_params['hr_rest']
                    hrr = hr_params['hrr']
                    
                    # Calculate the simple threshold needed for the original functions
                    if hrr > 0:
                        personalized_hr_threshold = (hrr * self.HR_ACTIVITY_HRR_INTENSITY) + hr_rest
                    else:
                        personalized_hr_threshold = np.inf

                    # Define the two windows and extract the hr features for each one
                    time_to_peak1 = all_postprandial_df.at[index, 'time_to_first_glucose_peak_after_meal']

                    if pd.notna(time_to_peak1):
                        peak_timestamp = meal_start + pd.Timedelta(minutes=time_to_peak1)

                        # Window 1: From meal_start untilet 15 minutes after the peak
                        start_window_1 = meal_start
                        end_window_1 = peak_timestamp + pd.Timedelta(minutes=15)
                        end_window_1 = min(end_window_1, meal_end) # Ensure it doesn't exceed meal_end

                        # Window 2: From 15 mins post-peak to the end of the postprandial period
                        start_window_2 = end_window_1
                        end_window_2 = meal_end

                        # Extract features for Window 1 (Predictive Features)
                        if start_window_1 < end_window_1:
                            hr_features_15min = self._calculate_hr_features_for_window(hr_df_subj, start_window_1, end_window_1, personalized_hr_threshold)
                            for key, value in hr_features_15min.items():
                                all_postprandial_df.at[index, f'{key}_15min_post_peak'] = value


                        # Extract features for Window 2 (Analytical Features)
                        if start_window_2 < end_window_2:
                            hr_features_later = self._calculate_hr_features_for_window(hr_df_subj, start_window_2, end_window_2, personalized_hr_threshold)
                            for key, value in hr_features_later.items():
                                all_postprandial_df.at[index, f'{key}_later_post_peak'] = value

                    # Pre-meal HR feature extraction
                    first_hr_ts = hr_df_subj.index.min()
                    pre_hr_features = self._calculate_pre_meal_hr_metrics(hr_df_subj, meal_start, first_hr_ts, personalized_hr_threshold)
                    for key, value in pre_hr_features.items():
                        all_postprandial_df.at[index, key] = value
                    
                    
                    ##  PHYSICAL ACTIVITY
                    hr_df = subjects_wearable_data.get(subj_id, {}).get('HEART_RATE')
                    steps_df = subjects_wearable_data.get(subj_id, {}).get('STEPS')

                    # Calculate Post-Meal "During Rise" Activity
                    time_to_peak1 = all_postprandial_df.at[index, 'time_to_first_glucose_peak_after_meal']
                    if pd.notna(time_to_peak1):
                        end_rise_window = min(meal_start + pd.Timedelta(minutes=time_to_peak1 + 15), meal_end)
                        rise_activity_features = self._calculate_physical_activity_for_window(
                            hr_df, steps_df, meal_start, end_rise_window,  hr_rest, hrr
                        )
                        for key, value in rise_activity_features.items():
                            all_postprandial_df.at[index, f'{key}_during_rise'] = value
                    
                    # Calculate exploratory activity until glycemic nadir 
                    postprandial_glucose = glucose_series_filtered.loc[meal_start:meal_end]
                    if not postprandial_glucose.empty:
                        nadir_timestamp = postprandial_glucose.idxmin() # get timestamp of minimum value
                        if pd.notna(nadir_timestamp) and nadir_timestamp > meal_start:
                            nadir_activity_features = self._calculate_physical_activity_for_window(
                            hr_df, steps_df, meal_start, nadir_timestamp, hr_rest, hrr
                        )
                        for key, value in nadir_activity_features.items():
                            all_postprandial_df.at[index, f'{key}_until_nadir'] = value

                    # Calculate Pre-Meal Activity
                    pre_meal_activity_features = self._extract_pre_meal_activity_features(
                        hr_df, steps_df, meal_start,  hr_rest, hrr
                    )
                    for key, value in pre_meal_activity_features.items():
                        all_postprandial_df.at[index, key] = value
                    
                    
                    # =================================================================
                    # NEW: ACTIVITY BOUT, TIMING, AND INTERACTION FEATURE EXTRACTION
                    # =================================================================
                    if hr_df is not None and not hr_df.empty and steps_df is not None and not steps_df.empty and pd.notna(time_to_peak1):
                        
                        # --- 1. Bout Analysis (within the "during_rise" window) ---
                        # We need to recalculate the 'is_active_mask' specifically for this analysis
                        end_rise_window = min(meal_start + pd.Timedelta(minutes=time_to_peak1 + 15), meal_end)
                        
                        hr_mask = (hr_df.index >= meal_start) & (hr_df.index <= end_rise_window)
                        hr_rise = hr_df.loc[hr_mask]['HEART_RATE'].dropna()

                        steps_mask = (steps_df.index >= meal_start) & (steps_df.index <= end_rise_window)
                        steps_rise = steps_df.loc[steps_mask]['STEPS'].dropna()

                        if len(hr_rise) > 1 and len(steps_rise) > 1:
                            hr_resampled = hr_rise.resample('1min').ffill()
                            steps_resampled = steps_rise.resample('1min').ffill()
                            cadence = steps_resampled.diff().fillna(0)
                            cadence[cadence < 0] = 0
                            
                            combined_df = pd.DataFrame({'HR': hr_resampled, 'Cadence': cadence}).dropna()
                            
                            if not combined_df.empty:
                                thresholds = {
                                    'light': (hrr * self.HR_ACTIVITY_HRR_INTENSITIES['light'][0]) + hr_rest
                                }
                                # Activity is defined as movement (cadence > min) and elevated HR
                                is_active_mask = (combined_df['Cadence'] > self.MIN_CADENCE_FOR_ACTIVITY) & \
                                                 (combined_df['HR'] >= thresholds['light'])
                                
                                if is_active_mask.any():
                                    # Calculate bout features
                                    # A "bout" starts when is_active_mask switches from False to True
                                    activity_groups = is_active_mask.ne(is_active_mask.shift()).cumsum()
                                    active_blocks = activity_groups[is_active_mask]
                                    
                                    if not active_blocks.empty:
                                        bout_lengths = active_blocks.value_counts()
                                        all_postprandial_df.at[index, 'number_of_activity_bouts_during_rise'] = len(bout_lengths)
                                        all_postprandial_df.at[index, 'mean_bout_duration_during_rise'] = bout_lengths.mean()
                                        all_postprandial_df.at[index, 'max_bout_duration_during_rise'] = bout_lengths.max()
                                    
                                    # --- 2. Timing Analysis ---
                                    first_activity_timestamp = is_active_mask[is_active_mask].index[0]
                                    time_to_first_activity = (first_activity_timestamp - meal_start).total_seconds() / 60
                                    all_postprandial_df.at[index, 'time_from_meal_to_first_activity'] = time_to_first_activity

                                    # Activity before vs. after peak
                                    peak_timestamp = meal_start + pd.Timedelta(minutes=time_to_peak1)
                                    activity_before_peak = is_active_mask.loc[:peak_timestamp]
                                    activity_after_peak = is_active_mask.loc[peak_timestamp:]
                                    
                                    if not activity_before_peak.empty:
                                        all_postprandial_df.at[index, 'activity_proportion_before_peak'] = activity_before_peak.mean() * 100
                                    if not activity_after_peak.empty:
                                        all_postprandial_df.at[index, 'activity_proportion_after_peak'] = activity_after_peak.mean() * 100

                    # --- 3. Interaction Feature Calculation ---
                    activity_prop = all_postprandial_df.at[index, 'activity_proportion_during_rise']
                    pre_meal_glucose = all_postprandial_df.at[index, 'mean_glucose_1h_pre_meal']
                    rate_of_increase = all_postprandial_df.at[index, 'rate_of_increase']

                    if pd.notna(activity_prop) and pd.notna(pre_meal_glucose):
                        all_postprandial_df.at[index, 'activity_x_pre_meal_glucose'] = activity_prop * pre_meal_glucose
                    
                    if pd.notna(activity_prop) and pd.notna(rate_of_increase):
                        all_postprandial_df.at[index, 'activity_x_rate_of_increase'] = activity_prop * rate_of_increase
 

            #------------------------------------------------------------------
            # 16. Pulse Ox Features 
            #------------------------------------------------------------------
            if row.get('has_pulse_ox_data') == 1:
                pulse_ox_df_subj = subjects_wearable_data.get(subj_id, {}).get('PULSE_OX')
                if pulse_ox_df_subj is not None and not pulse_ox_df_subj.empty:
                    post_pulse_ox_features = self._extract_pulse_ox_features(pulse_ox_df_subj, meal_start, meal_end)
                    for key, value in post_pulse_ox_features.items(): 
                        all_postprandial_df.at[index, key] = value
                    first_pulse_ox_ts = pulse_ox_df_subj.index.min()
                    pre_pulse_ox_features = self._calculate_pre_meal_pulse_ox_metrics(pulse_ox_df_subj, meal_start, first_pulse_ox_ts)
                    for key, value in pre_pulse_ox_features.items(): 
                        all_postprandial_df.at[index, key] = value
            
            
            #------------------------------------------------------------------
            # 17. Meal Categorization and Step Features
            #------------------------------------------------------------------
            meal_hour = meal_start.hour
            if 5 <= meal_hour < 9:
                meal_category = 'Breakfast'
            elif 11 <= meal_hour < 14:
                meal_category = 'Lunch'
            elif 17 <= meal_hour < 19.5: # Up to 19:30
                meal_category = 'Dinner'
            else:
                meal_category = 'Snack'
            all_postprandial_df.at[index, 'meal_category'] = meal_category

            if row.get('has_steps_data') == 1:
                steps_df_subj = subjects_wearable_data.get(subj_id, {}).get('STEPS')
                if steps_df_subj is not None and not steps_df_subj.empty:
                    # Pre-meal steps calculation
                    for hour_window in [1, 2, 3, 4]:
                        start_window = meal_start - pd.Timedelta(hours=hour_window)
                        end_window = meal_start
                        
                        mask = (steps_df_subj.index >= start_window) & (steps_df_subj.index < end_window)
                        pre_meal_steps_series = steps_df_subj.loc[mask]['STEPS'].dropna()
                        
                        steps_taken = np.nan
                        if len(pre_meal_steps_series) >= 2:
                            # Calculate the difference between each consecutive point
                            diffs = pre_meal_steps_series.diff()
                            # Sum only the positive differences, ignoring decreases from resets
                            steps_taken = diffs[diffs > 0].sum()
                            # # For cumulative counters, steps taken = max - min in the window
                            # steps_taken = pre_meal_steps_series.max() - pre_meal_steps_series.min()
                        elif not pre_meal_steps_series.empty:
                            steps_taken = 0 # If only one sample, assume no change
                            
                        all_postprandial_df.at[index, f'steps_{hour_window}h_pre_meal'] = steps_taken    
                        
                        
            #------------------------------------------------------------------
            
            if (index + 1) % 1000 == 0:
                print(f"Elaborati {index + 1}/{len(all_postprandial_df)} periodi.")



        # Correctly count and report the number of saturation and interpolated events
        total_meals_with_saturation = int(all_postprandial_df['is_saturated_at_39'].sum())
        total_interpolated_meals = int(all_postprandial_df['was_interpolated'].sum())
        
        print(f"\nFound a total of {total_meals_with_saturation} meal events with sensor saturation in the analysis window.")
        print(f"Applied spline interpolation to {total_interpolated_meals} of these meals.")
        
        print("\n-->Estrazione feature completata<--")

        output_path = os.path.join(self.data_dir_db, self.output_filename)
        all_postprandial_df.to_csv(output_path, index=False) # index=False ---> prevents pandas from writing the DataFrame's index (which is typically just a range of numbers) as a separate column in the CSV file
        print(f"Salvato il DataFrame aggiornato con le feature a: {output_path}")
        return all_postprandial_df
















# # Esempio di utilizzo (per testare solo questa classe)
if __name__ == "__main__":
    # Assicurati che 'all_postprandial_data.csv' sia già stato generato da DatasetPreparer
    extractor = PostprandialFeatureExtractor(
        data_dir="data",
        data_dir_db="postprandial_db",
        postprandial_data_file='all_postprandial_data.csv', # Nome del file di input
        glucose_filename_pattern="{}_tma_processed.csv", # Nome del file glucosio processato
        hypo_threshold=70,
        deep_hypo_threshold=54,
        min_consecutive_samples_hypo=3,
        peak_search_min_distance_samples=6,
        peak_search_prominence=20
    )
    final_df_with_features = extractor.extract_features(
        output_filename='all_postprandial_data_with_features.csv' # Nome del file di output
    )
    print(final_df_with_features.head())