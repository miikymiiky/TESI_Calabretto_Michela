# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:43:05 2025

@author: prend
"""

# src/utils/utils4processing.py

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks # Aggiunto qui perché usato dalle derivate per TMA_meal
from scipy.interpolate import interp1d # For spline interpolation in saturation events

def apply_zero_phase_low_pass_filter(signal, sampling_rate, cutoff_freq, order):
    """Applica un filtro passa-basso zero-phase."""
    sampling_rate_freq = 1/sampling_rate
    nyquist_freq = 0.5 * sampling_rate_freq
    # Evita il caso in cui normalized_cutoff sia 0 o troppo grande
    if nyquist_freq == 0:
        return signal # O gestire l'errore diversamente
    normalized_cutoff = cutoff_freq / nyquist_freq
    if normalized_cutoff >= 1: # Frequenza di taglio troppo alta, nessuna filtrazione significativa
        return signal
    
    b, a = butter(order, normalized_cutoff, btype='low', analog=False) # analog=False ---> = digital filter
                                                                       # b, a = filters coefficients (numerator and denominator)
    # Assicurati che il segnale non contenga NaN per filtfilt
    if pd.isna(signal).all(): # returns True if all elements are NaN
        return np.full_like(signal, np.nan) # creates a new NumPy array with the same shape and type as signal, filled with NaN values
    
    # Gestisce NaN riempiendoli temporaneamente e ripristinandoli dopo
    nan_mask = pd.isna(signal)
    # Interpolation on the full length of the signal starting from the valid values of the signal (x, xp, fp)
    signal_no_nan = np.interp(np.arange(len(signal)), # returns the a vector of indexes which lenth is the length of the signal
                              np.arange(len(signal))[~nan_mask], # returns the indexes where the signal is not NaN
                              signal[~nan_mask]) # takes the signal value where it is not NaN
    
    filtered_signal = filtfilt(b, a, signal_no_nan)
    
    # Ripristina i NaN nelle posizioni originali
    filtered_signal[nan_mask] = np.nan
    
    return filtered_signal


def calculate_derivative(signal, time):
    """Calcola la derivata prima e seconda del segnale."""
    # Assicurati che time sia un array di float (minuti, secondi, ecc.)
    # e non un DatetimeIndex diretto per np.diff
    
    # Gestione dei NaN nel segnale: interpola per calcolare le derivate
    nan_mask = pd.isna(signal)
    if nan_mask.all(): # Se tutti i valori sono NaN, non possiamo calcolare derivate
        return np.full_like(signal, np.nan), np.full_like(signal, np.nan)
    
    # Interpola solo se ci sono NaN
    if nan_mask.any():
        signal_interp = np.interp(np.arange(len(signal)), 
                                  np.arange(len(signal))[~nan_mask], 
                                  signal[~nan_mask])
    else:
        signal_interp = signal
        
    # Calcola dt in minuti
    dt = np.diff(time.astype('int64') / 1e9 / 60) # Converti nanosecondi a minuti
                                                  # time.astype('int64') converts datetime values to Unix timestamps in nanoseconds
                                                  # .diff calculates the difference between consecutive time points
    
    if len(dt) == 0: # Caso di un singolo punto
        return np.array([np.nan]), np.array([np.nan])
    
    # Evita divisione per zero se dt è 0 (campioni identici nel tempo, raro ma possibile)
    dt[dt == 0] = np.finfo(float).eps # Piccolo numero per evitare divisione per zero

    dg = np.diff(signal_interp)
    derivative = dg / dt
    
    # Padding per la stessa dimensione del segnale originale
    derivative = np.concatenate(([derivative[0] if len(derivative) > 0 else np.nan], derivative)) 

    # Calcolo della seconda derivata
    d_derivative = np.diff(derivative)
    second_derivative = d_derivative / dt
    second_derivative = np.concatenate(([second_derivative[0] if len(second_derivative) > 0 else np.nan], second_derivative)) 
    
    # Ripristina NaN nelle posizioni originali
    derivative[nan_mask] = np.nan
    second_derivative[nan_mask] = np.nan

    return derivative, second_derivative


def interpolate_saturation_events(glucose_series_to_interpolate: pd.Series, glucose_series_for_detection: pd.Series,
                                  saturation_value: int = 39,
                                  min_consecutive_samples: int = 3, points_around: int = 4,
                                  interp_kind: str = 'quadratic') -> tuple[pd.Series, bool]:
    """
    Finds saturation events in a glucose series and interpolates them using a spline.

    Args:
        glucose_series_to_interpolate (pd.Series): The series whose values will be replaced (e.g., filtered data).
        glucose_series_for_detection (pd.Series): The series used to find saturation events (e.g., raw data).
        saturation_value (int): The glucose value indicating sensor saturation (e.g., 39).
        min_consecutive_samples (int): Min number of consecutive samples to be a saturation event.
        points_around (int): Number of valid data points before and after the event to use for fitting.
        interp_kind (str): The kind of spline for scipy.interpolate.interp1d (e.g., 'linear', 'quadratic', 'cubic').

    Returns:
        tuple: A tuple containing:
            - pd.Series: A new series with the saturated segments interpolated.
            - bool: A flag that is True if an interpolation was performed, otherwise False.
    """
    # Create a copy of the series that will be modified
    interpolated_series = glucose_series_to_interpolate.copy()
    
    # Initialize a flag to track if any changes are made
    was_changed = False

    # Use the raw/unfiltered series to accurately detect saturation events
    is_saturated = (glucose_series_for_detection == saturation_value)
    
    # Find blocks of consecutive saturation events
    saturated_groups = is_saturated.ne(is_saturated.shift()).cumsum() # .ne(...) checks where the mask changes
                                                                      # .cumsum() turns these changes into group IDs
                                                                      # .shift() shifts the series by one row
    true_saturated_blocks = saturated_groups[is_saturated]
    
    if true_saturated_blocks.empty:
        return interpolated_series, was_changed # No saturation, return original series

    block_sizes = true_saturated_blocks.value_counts()
    valid_event_groups = block_sizes[block_sizes >= min_consecutive_samples].index

    for group_id in valid_event_groups:
        event_indices = true_saturated_blocks[true_saturated_blocks == group_id].index
        start_event_loc = glucose_series_to_interpolate.index.get_loc(event_indices[0])
        end_event_loc = glucose_series_to_interpolate.index.get_loc(event_indices[-1])

        # Define the window for finding context points (before and after the event)
        context_start_loc = max(0, start_event_loc - points_around)
        context_end_loc = min(len(glucose_series_to_interpolate) - 1, end_event_loc + points_around)
        
        # Get context indices and values, excluding the saturated part
        context_indices_for_fit = []
        # Before event
        context_indices_for_fit.extend(range(context_start_loc, start_event_loc)) # range generates the indexes from context_start_loc to start_event_loc-1
        # After event
        context_indices_for_fit.extend(range(end_event_loc + 1, context_end_loc + 1))
        
        if len(context_indices_for_fit) < 4: # Need enough points for a robust fit
            print(f"Warning: Not enough context points to interpolate event at {event_indices[0]}. Skipping.")
            continue
            
        # Prepare data for interpolation
        # Use numeric timestamps for x-axis
        # These are the “anchor points” that the spline will be fit through
        x_fit = glucose_series_to_interpolate.index[context_indices_for_fit].astype(np.int64) # numeric timestamps (x-coordinates)
        y_fit = glucose_series_to_interpolate.iloc[context_indices_for_fit].values # glucose readings at those times (y-coordinates)

        try:
            # Create the interpolation function (spline)
            spline = interp1d(x_fit, y_fit, kind=interp_kind, bounds_error=False, fill_value="extrapolate") # bounds_error=False ---> don’t crash if asked for values outside the known x-range
                                                                                                            # fill_value="extrapolate" ---> if needed, allow extrapolation instead of returning NaN

            # Get the timestamps of the saturated segment to be interpolated
            # These are the time points where we need to fill in new values
            x_interp = event_indices.astype(np.int64) # event_indices are the timestamps of the saturated readings
                                                      # .astype(np.int64) converts them to numbers
            
            # Perform interpolation
            y_interp = spline(x_interp)

            # Replace the values in our copied series
            interpolated_series.loc[event_indices] = y_interp
            # Set the interpolation flag to True 
            was_changed = True
            
        except Exception as e:
            print(f"Error during interpolation for event at {event_indices[0]}: {e}")

    return interpolated_series, was_changed


def re_position_meals(df, derivative_threshold, second_derivative_threshold, search_window_before_meal, search_window_after_meal, sampling_rate_minutes):
    """
    Args:
        df (pd.DataFrame): DataFrame del glucosio con colonne 'meal', 'glucose_derivative', 'glucose_second_derivative'.
        derivative_threshold (float): Soglia per la derivata prima.
        second_derivative_threshold (float): Soglia per la derivata seconda.
        search_window_before_meal (int): Minuti prima del pasto segnalato in cui cercare l'inizio reale.
        search_window_after_meal (int): Minuti dopo il pasto segnalato in cui cercare l'inizio reale.
        sampling_rate_minutes (int): Intervallo di campionamento in minuti (e.g., 5 per 5 minuti).

    Returns:
        pd.Series: Serie booleana con i pasti riposizionati ('TMA_meal_corrected').
    """
    tma_meal_corrected = pd.Series(0, index=df.index, dtype=int) # initialize the series of tm_meal_corrected
    meal_reported_indices = df[df['TMA_meal'] == 1].index.tolist()

    for meal_time in meal_reported_indices:
        # Definizione della finestra di ricerca (indici)
        start_search_window = meal_time - pd.Timedelta(minutes=search_window_before_meal)
        end_search_window = meal_time + pd.Timedelta(minutes=search_window_after_meal)

        # Filtra il DataFrame nella finestra di ricerca
        search_df = df.loc[start_search_window:end_search_window].copy()
        glucose_derivative = search_df['glucose_derivative'].to_numpy()
        glucose_second_derivative = search_df['glucose_second_derivative'].to_numpy()

        if len(glucose_derivative) > 0:
            # Cerca l'inizio della salita più significativa NELLA finestra
            #potential_meal_start_index = -1
            max_derivative = -np.inf # initialization
            best_start_index_relative = -1
            if abs(df.loc[meal_time]['glucose_derivative']) > derivative_threshold:

    
                for i in range(len(glucose_derivative) - 3, 0, -1): # (start, stop, step)
                    
                    if (glucose_derivative[i] < derivative_threshold) and (glucose_derivative[i+1] >= derivative_threshold) and (glucose_derivative[i+2] >= derivative_threshold): # Cerchiamo una salita consistente
                        if glucose_derivative[i+1] > max_derivative:
                            max_derivative = glucose_derivative[i+1]
                            best_start_index_relative = i
                            
                if best_start_index_relative == -1:
                    
                    for i in range(len(glucose_second_derivative) - 3, 0, -1):
                        # Cerchiamo un punto in cui la derivata seconda diventa positiva
                        # e la derivata prima è anch'essa positiva ma non ancora al picco
                        condA = glucose_second_derivative[i] <= second_derivative_threshold
                        condB = glucose_second_derivative[i+1] > second_derivative_threshold
                        condD = glucose_second_derivative[i+2] >= derivative_threshold
                        condC = glucose_second_derivative[i+1] > derivative_threshold
                        
                        if condA and condB and condC and condD :
                            best_start_index_relative = i
                            break # Prendiamo la prima occorrenza
                            
            if best_start_index_relative != -1:
                corrected_meal_times = search_df.index[best_start_index_relative]
                tma_meal_corrected.loc[corrected_meal_times] = 1
            else:
                tma_meal_corrected.loc[meal_time] = 1
            
    return tma_meal_corrected


def create_postprandial_dataframe(subject_df, all_wearable_data, subject_id, postprandial_duration_hours=3):
    """
    Crea un DataFrame contenente i periodi post-prandiali, includendo flag per la disponibilità
    dei dati wearable e i tempi di inizio/fine.

    Args:
        subject_df (pd.DataFrame): DataFrame del soggetto con glucosio e 'TMA_meal_corrected'.
        all_wearable_data (dict): Dizionario di DataFrames wearable per il soggetto.
        subject_id (str): ID del soggetto.
        postprandial_duration_hours (int): Durata standard del periodo post-prandiale in ore.

    Returns:
        pd.DataFrame: DataFrame con i dati post-prandiali.
    """
    postprandial_data = []
    
    # Converti gli indici a DatetimeIndex
    subject_df.index = pd.to_datetime(subject_df.index)
    
    # Assicurati che 'TMA_meal_corrected' sia di tipo numerico/booleano per il confronto
    meal_start_times = subject_df.index[subject_df['TMA_meal_corrected'] == 1].tolist()

    list_of_wearable_keys = list(all_wearable_data.keys()) # Ottieni le chiavi una volta sola

    for i, start_time in enumerate(meal_start_times):
        meal_end_time = start_time + pd.Timedelta(hours=postprandial_duration_hours)

        # Trova il tempo del pasto successivo
        next_meal_index = i + 1
        if next_meal_index < len(meal_start_times):
            next_meal_start_time = meal_start_times[next_meal_index]
            # Il periodo post-prandiale finisce al massimo all'inizio del pasto successivo
            meal_end_time = min(meal_end_time, next_meal_start_time)
            
        row_data = {
            'subj_id': subject_id,
            'meal_time_start': start_time,
            'meal_time_end': meal_end_time,
            'postprandial_duration_minutes': (meal_end_time - start_time).total_seconds() / 60
        }

        # Controlla se ci sono dati wearable (1 se ci sono, 0 se percentuale di missing > 50%)
        for key_wearable in list_of_wearable_keys:
            wearable_df = all_wearable_data[key_wearable]
            # Assicurati che l'indice del df wearable sia DatetimeIndex
            if not isinstance(wearable_df.index, pd.DatetimeIndex):
                wearable_df.index = pd.to_datetime(wearable_df.index, format='%d-%b-%Y %H:%M:%S')

            wearable_df_range = wearable_df.loc[start_time:meal_end_time]
            
            # Calcola la percentuale di missing solo se ci sono dati nell'intervallo
            if not wearable_df_range.empty and not wearable_df_range[wearable_df_range.columns[0]].empty: # Check if the selected range has data
                total_samples = len(wearable_df_range)
                missing_samples = wearable_df_range.isnull().sum().sum() # Sum across all columns if multiple
                if total_samples > 0:
                    percent_missing = (missing_samples / total_samples) * 100
                else:
                    percent_missing = 100 # No samples means 100% missing
            else:
                percent_missing = 100 # Nessun dato nell'intervallo

            # Flag per la disponibilità
            flag_name = f'has_{key_wearable.lower()}_data'
            row_data[flag_name] = 1 if percent_missing <= 50 else 0
            
        postprandial_data.append(row_data)

    return pd.DataFrame(postprandial_data)