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
                 hr_activity_derivative_threshold=3.0): # 3 bpm/minute increase
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
        self.HR_ACTIVITY_DERIVATIVE_THRESHOLD = hr_activity_derivative_threshold


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
        hypo_duration = 0

        is_deep_hypo = 0
        deep_hypo_time = np.nan
        deep_hypo_duration = 0

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


    def _calculate_pre_meal_metrics(self, glucose_series_filtered, meal_start_time, first_glucose_timestamp):
        metrics = {}
        min_distance_samples_pre_meal = int(30 / self.SAMPLING_INTERVAL_MINUTES)
        if min_distance_samples_pre_meal < 1: min_distance_samples_pre_meal = 1
        
        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            keys_to_init = [f'mean_glucose_{interval_name}', f'std_glucose_{interval_name}', f'cv_glucose_{interval_name}', f'min_glucose_{interval_name}', f'max_glucose_{interval_name}', f'range_{interval_name}', f'prop_above_130_{interval_name}', f'prop_80_130_{interval_name}', f'prop_50_80_{interval_name}', f'prop_below_50_{interval_name}', f'num_positive_peaks_{interval_name}', f'num_negative_peaks_{interval_name}']
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
            
        return metrics
    
    
    def _detect_physical_activity(self, hr_window, personalized_hr_threshold):
       """
       Detects physical activity and returns a binary series (1=active, 0=not active).

       Args:
           hr_window (pd.Series): Series of heart rate values for a specific time window.
           personalized_hr_threshold (float): The subject-specific HR threshold for activity.

       Returns:
           pd.Series: A binary series with 1s for active periods and 0s otherwise.
       """
       # Calculate the heart rate derivative (rate of change per minute)
       hr_derivative = hr_window.diff() / (hr_window.index.to_series().diff().dt.total_seconds() / 60.0)
       
       # Create a boolean mask where both activity conditions are met
       is_active_mask = (hr_window > personalized_hr_threshold) & (hr_derivative > self.HR_ACTIVITY_DERIVATIVE_THRESHOLD)
       
       # Convert the boolean mask to a binary series of 1s and 0s
       activity_series = is_active_mask.astype(int)
       
       return activity_series
    
    
    def _extract_heart_rate_features(self, heart_rate_df, start_time, end_time, personalized_hr_threshold):
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
            'hr_slope': np.nan, # to measures the direction and rate of change of the heart rate
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
        
        # Slope
        # Create a numeric time variable (seconds from start)
        time_in_minutes = (hr_window.index - hr_window.index[0]).total_seconds() / 60
        # Use numpy's polyfit to get the slope of a linear fit
        slope, _ = np.polyfit(time_in_minutes, hr_window.values, 1)
        features['hr_slope'] = slope
        
        # Physical Activity
        # First get the binary series, then calculate the percentage
        activity_series = self._detect_physical_activity(hr_window, personalized_hr_threshold)
        if not activity_series.empty:
            # The mean of a 0/1 series is the proportion of 1s
            features['hr_prop_time_active'] = activity_series.mean() * 100


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
                    f'hr_max_{interval_name}', f'hr_range_{interval_name}', f'hr_slope_{interval_name}',
                    f'hr_variability_{interval_name}', f'hr_prop_time_active_{interval_name}'
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

            successive_diffs = np.diff(hr_pre_meal_window.values)
            if len(successive_diffs) > 0:
                metrics[f'hr_variability_{interval_name}'] = np.sqrt(np.mean(successive_diffs ** 2))
            
            time_in_minutes = (hr_pre_meal_window.index - hr_pre_meal_window.index[0]).total_seconds() / 60.0
            if time_in_minutes.nunique() > 1:
                slope, _ = np.polyfit(time_in_minutes, hr_pre_meal_window.values, 1)
                metrics[f'hr_slope_{interval_name}'] = slope
            
            # Physical Activity
            # First get the binary series, then calculate the percentage
            activity_series = self._detect_physical_activity(hr_pre_meal_window, personalized_hr_threshold)
            if not activity_series.empty:
                metrics[f'hr_prop_time_active_{interval_name}'] = activity_series.mean() * 100

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
    
    
    def _calculate_daily_steps(self, steps_df):
        """
        Calculates the total sum of steps for each day.
        """
        if steps_df is None or steps_df.empty:
            return None
        # Resample by calendar day ('D') and sum the steps
        daily_steps = steps_df['STEPS'].resample('D').max()
        return daily_steps

    
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
            'time_since_previous_meal_minutes', 'time_peak_hypo_event', 'target_hypoglycemia', 'hypo_to_check_flag',  
            
            # NEW GLUCOSE features
            'postprandial_mean_glucose', 'postprandial_std_glucose', 'postprandial_range',
            
            # New columns for postprandial Time-In-Range (TIR)
            'postprandial_tir_target', 'postprandial_tir_l1_hyper', 'postprandial_tir_l2_hyper', 'postprandial_tir_l1_hypo', 'postprandial_tir_l2_hypo',
            
            # HEART_RATE features
            'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range', 'hr_slope', 'hr_variability',
            
            # PULSE_OX features
            'pulse_ox_mean', 'pulse_ox_std', 'pulse_ox_min', 'pulse_ox_max', 'pulse_ox_range'
        ]
        
        pre_meal_glucose_cols = []
        pre_meal_hr_cols = []
        pre_meal_pulse_ox_cols = []
        # Aggiungi le colonne per le metriche pre-pasto dinamicamente
        for interval_hours in self.PRE_MEAL_INTERVALS_HOURS:
            interval_name = f"{interval_hours}h_pre_meal"
            pre_meal_glucose_cols.extend([
                f'mean_glucose_{interval_name}', f'std_glucose_{interval_name}', f'cv_glucose_{interval_name}', f'min_glucose_{interval_name}', f'max_glucose_{interval_name}', f'range_{interval_name}',
                f'prop_above_130_{interval_name}', f'prop_80_130_{interval_name}', f'prop_50_80_{interval_name}', f'prop_below_50_{interval_name}',
                f'num_positive_peaks_{interval_name}', f'num_negative_peaks_{interval_name}', 
            ])
            pre_meal_hr_cols.extend([ 
                f'hr_mean_{interval_name}', f'hr_std_{interval_name}', f'hr_min_{interval_name}',  f'hr_max_{interval_name}',  f'hr_range_{interval_name}', 
                f'hr_slope_{interval_name}', f'hr_variability_{interval_name}', f'hr_prop_time_active_{interval_name}' 
            ])
            pre_meal_pulse_ox_cols.extend([
                f'pulse_ox_mean_{interval_name}', f'pulse_ox_std_{interval_name}', f'pulse_ox_min_{interval_name}', f'pulse_ox_max_{interval_name}', f'pulse_ox_range_{interval_name}'
            ])

        new_columns = base_columns + pre_meal_glucose_cols + pre_meal_hr_cols + pre_meal_pulse_ox_cols
        
        # inizializzazione a NaN
        for col in new_columns:
            if col not in all_postprandial_df.columns:
                all_postprandial_df[col] = np.nan


        subjects_glucose_data = {} # initialize the glucos data
        subjects_wearable_data = {} # initialize the wearable data
        subjects_daily_steps_data = {} # initialize the daily steps
        subjects_hr_thresholds = {} # initialize the subjects' threshold

        old_subj = ''

        print("--> Estrazione feature CGM + pasto <--")
        
        for index, row in all_postprandial_df.iterrows():
            
            subj_id = row['subj_id']
            meal_start = row['meal_time_start']
            meal_end = row['meal_time_end']

            # Carica dati glucosio del soggetto
            if subj_id not in subjects_glucose_data:
                subjects_glucose_data[subj_id] = self._load_glucose_df_for_subject(subj_id)
            
            glucose_df_subj = subjects_glucose_data[subj_id]

            if glucose_df_subj is None or 'glucose_filtered' not in glucose_df_subj.columns:
                print(f"Salto pasto per {subj_id} a {meal_start}: dati glucosio filtrati non disponibili.")
                continue

            # considero le tre serie di interesse: CGM, CGM filtrato e derivata del CGM
            glucose_series = glucose_df_subj['glucose']
            glucose_series_filtered = glucose_df_subj['glucose_filtered']
            glucose_derivative_series = glucose_df_subj['glucose_derivative']
            
            # Ottieni il primo timestamp disponibile per i dati di glucosio del soggetto
            first_glucose_timestamp = glucose_series_filtered.index.min()
            
            if pd.isna(first_glucose_timestamp):
                 print(f"Salto pasto per {subj_id} a {meal_start}: Dati glucosio vuoti.")
                 continue


            #-----------------------------------------------------------------
            # 1. Target: Rilevamento Ipoglicemia
            #-----------------------------------------------------------------
            is_hypo, hypo_time, hypo_duration, is_deep_hypo, deep_hypo_time, deep_hypo_duration = \
                self._detect_hypoglycemia(glucose_series_filtered, glucose_derivative_series, meal_start, meal_end)
            
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
                self._find_postprandial_peaks(glucose_series, meal_start, meal_end)
            
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
            glucose_at_meal_start = glucose_series_filtered.get(meal_start, np.nan)
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
                glucose_15min_after_peak1 = glucose_series.asof(time_15min_after_peak1) # 'asof' prende il valore più recente <= timestamp

            all_postprandial_df.at[index, 'glucose_derivative_15min_before_peak1'] = deriv_15min_before_peak1
            all_postprandial_df.at[index, 'glucose_derivative_15min_after_peak1'] = deriv_15min_after_peak1
            all_postprandial_df.at[index, 'glucose_value_15min_after_peak1'] = glucose_15min_after_peak1

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
            pre_meal_metrics = self._calculate_pre_meal_metrics(glucose_series_filtered, meal_start, first_glucose_timestamp)
            
            for key, value in pre_meal_metrics.items():
                all_postprandial_df.at[index, key] = value
 
            
            # NEW FEATURES ADDED 
            
            #------------------------------------------------------------------
            # 13. Postprandial descriptive statistics and Time-In-Range (TIR)
            # These features characterize the glucose profile during the postprandial period
            #------------------------------------------------------------------
            # Select the postprandial glucose data from the filtered series
            postprandial_glucose = glucose_series_filtered.loc[meal_start:meal_end]

            # Check if there is valid data in the window to perform calculations
            if not postprandial_glucose.empty and not postprandial_glucose.isnull().all():
                # Basic Stats (Media, SD, Range) 
                all_postprandial_df.at[index, 'postprandial_mean_glucose'] = postprandial_glucose.mean()
                all_postprandial_df.at[index, 'postprandial_std_glucose'] = postprandial_glucose.std()
                all_postprandial_df.at[index, 'postprandial_range'] = postprandial_glucose.max() - postprandial_glucose.min()

                # Time-In-Range (TIR) calculation as a proportion of the time in the window
                total_samples_in_window = postprandial_glucose.notna().sum()
                if total_samples_in_window > 0:
                    # Time in Target (70-180 mg/dL)
                    tir_target = ((postprandial_glucose >= 70) & (postprandial_glucose <= 180)).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_target'] = tir_target

                    # Time in L1 Hyperglycemia (181-250 mg/dL)
                    tir_l1_hyper = ((postprandial_glucose > 180) & (postprandial_glucose <= 250)).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l1_hyper'] = tir_l1_hyper

                    # Time in L2 Hyperglycemia (> 250 mg/dL)
                    tir_l2_hyper = (postprandial_glucose > 250).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l2_hyper'] = tir_l2_hyper

                    # Time in L1 Hypoglycemia (54-69 mg/dL)
                    tir_l1_hypo = ((postprandial_glucose >= 54) & (postprandial_glucose < 70)).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l1_hypo'] = tir_l1_hypo

                    # Time in L2 Hypoglycemia (< 54 mg/dL), also known as Time Below Range (TBR) Level 2
                    tir_l2_hypo = (postprandial_glucose < 54).sum() / total_samples_in_window
                    all_postprandial_df.at[index, 'postprandial_tir_l2_hypo'] = tir_l2_hypo
            
            #------------------------------------------------------------------
            
            # WEARABLE FEATURES    
            
            # Load wearable data once per subject
            if subj_id not in subjects_wearable_data:
                subjects_wearable_data[subj_id] = self._load_all_wearable_data_for_subject(subj_id)
           
            
            #------------------------------------------------------------------
            # 14. HR Features
            #------------------------------------------------------------------
            if row.get('has_heart_rate_data') == 1:
                hr_df_subj = subjects_wearable_data.get(subj_id, {}).get('HEART_RATE')
                if hr_df_subj is not None and not hr_df_subj.empty:
                    
                    # Calculate and cache personalized thresholds once per subject
                    if subj_id not in subjects_hr_thresholds:
                        hr_rest = hr_df_subj['HEART_RATE'].quantile(0.025)
                        hr_max = hr_df_subj['HEART_RATE'].quantile(0.975)
                        hrr = hr_max - hr_rest
                        # Ensure hrr is not zero to avoid division by zero or invalid threshold
                        if hrr > 0:
                            personalized_threshold = (hrr * self.HR_ACTIVITY_HRR_INTENSITY) + hr_rest
                        else:
                            personalized_threshold = np.inf # Set an impossibly high threshold if HRest/HRmax are the same
                        subjects_hr_thresholds[subj_id] = personalized_threshold
                    
                    personalized_hr_threshold = subjects_hr_thresholds[subj_id]

                    # Pass the threshold to the helper functions
                    post_hr_features = self._extract_heart_rate_features(hr_df_subj, meal_start, meal_end, personalized_hr_threshold)
                    for key, value in post_hr_features.items(): 
                        all_postprandial_df.at[index, key] = value
                    first_hr_ts = hr_df_subj.index.min()
                    pre_hr_features = self._calculate_pre_meal_hr_metrics(hr_df_subj, meal_start, first_hr_ts, personalized_hr_threshold)
                    for key, value in pre_hr_features.items(): 
                        all_postprandial_df.at[index, key] = value
            
            #------------------------------------------------------------------
            # 15. Pulse Ox Features 
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
            # 16. Daily Target Steps
            #------------------------------------------------------------------
            if row.get('has_steps_data') == 1:
                if subj_id not in subjects_daily_steps_data:
                    # Calculate and save daily steps for the subject 
                    steps_df_subj = subjects_wearable_data.get(subj_id, {}).get('STEPS')
                    subjects_daily_steps_data[subj_id] = self._calculate_daily_steps(steps_df_subj)

                daily_steps = subjects_daily_steps_data[subj_id]
                
                if daily_steps is not None:
                    meal_date = meal_start.date()
                    
                    # Normalize the meal_start timestamp to midnight to get a key that perfectly matches the DatetimeIndex of the 'daily_steps' Series
                    # Use .get() to safely access the date, in case there are no steps for that day
                    meal_date_key = meal_start.normalize()
                    total_steps_for_day = daily_steps.get(meal_date_key, 0)
                    
                    if total_steps_for_day >= self.DAILY_STEP_TARGET:
                        all_postprandial_df.at[index, 'daily_step_target_reached'] = 1
                    else:
                        all_postprandial_df.at[index, 'daily_step_target_reached'] = 0


            #------------------------------------------------------------------
            
            if (index + 1) % 1000 == 0:
                print(f"Elaborati {index + 1}/{len(all_postprandial_df)} periodi.")



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