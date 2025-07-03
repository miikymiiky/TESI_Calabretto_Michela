import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.dates as mdates # for date formats

# BASE VARIABLES
patient_number = 50
patient_code = f"BERNPBHF{patient_number:05d}" # format the patient's code adding the patient number that will be a five digit number, with leading zeros (i.e.00037)


# BASE PATH
BASE_PATH = os.path.expanduser("~/Desktop/UNIVERSITA'/TESI/TESI_magistrale/dataset") # uses ~ as a shortcut for a user's home directory ---> gets the home directory 
patient_csv_path = os.path.join(BASE_PATH, "data", patient_code, f"{patient_code}_tma_processed.csv")
postprandial_csv_path = os.path.join(BASE_PATH, "postprandial_db", "all_postprandial_data.csv")
heart_rate_csv_path = os.path.join(BASE_PATH, "data", patient_code, "HEART_RATE.csv")
hrv_csv_path = os.path.join(BASE_PATH, "data", patient_code, "HEART_RATE_VARIABILITY.csv")


# LOAD DATA
#if not os.path.exists(patient_csv_path):
#    raise FileNotFoundError(f"File not found for patient: {patient_code}")

patient_df = pd.read_csv(patient_csv_path) 
postprandial_df = pd.read_csv(postprandial_csv_path)

print(f"Loaded data for patient {patient_code}")


# THRESHOLDS
HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180


# EXTRACT COLUMNS TO PLOT FROM patient_df
patient_df['Time'] = pd.to_datetime(patient_df['Time'], errors='coerce') # converts the Time column from String to a dateTime 
                                                                         # "errors='coerce'" to convert invalid dates to NaT because there were values that were not dateTime
patient_df['glucose'] = pd.to_numeric(patient_df['glucose'], errors='coerce') # ensure glucose is numeric

postprandial_df['meal_time_start'] = pd.to_datetime(postprandial_df['meal_time_start'], errors='coerce') # convert also meal time columns in postprandial_df from string to dateTime objects
postprandial_df['meal_time_end'] = pd.to_datetime(postprandial_df['meal_time_end'], errors='coerce')

# Extract data (including NaNs)
time = patient_df['Time']
glucose = patient_df['glucose']

meal_reported = patient_df['meal'] # red triangles
meal_template = patient_df['TMA_meal'] # yellow X
meal_corrected = patient_df['TMA_meal_corrected'] # green dots


# CHECK NANS
nans_time = patient_df['Time'].isna().sum()
nans_glucose = patient_df['glucose'].isna().sum()

idx_nans_time = patient_df[patient_df['Time'].isna()]
idx_nans_glucose = patient_df[patient_df['glucose'].isna()]

print(f"\nTotal number of rows with missing 'Time' values: {nans_time}")
print(f"Total number of rows with missing 'glucose' values: {nans_glucose}")


# NUMBER OF MEALS
total_meal_reported = patient_df['meal'].notna().sum() # meals reported where either 0 or 1 or 2 or 3 (the rest is nans)
total_meal_template = patient_df['TMA_meal'].sum() # meals from the template and after processing where marked as 1 (the rest is 0s)
total_meal_corrected = patient_df['TMA_meal_corrected'].sum()

print(f"\nTotal number of 'Meal Reported' events: {total_meal_reported}")
print(f"Total number of 'Meal Template' events: {total_meal_template}")
print(f"Total number of 'Meal Corrected' events: {total_meal_corrected}")



# PLOTS
fig, axs = plt.subplots(3, 1, figsize=(15, 18)) 


## PLOT 1: GLUCOSE AND MEAL EVENTS ---
ax1 = axs[0]
ax1.plot(time, glucose, color='blue', zorder=1) # zorder=number ---> to decide what is drawn above what (0 goes to the back, and the higher number at the top)

# Create boolean masks to identify hypo- and hyperglycemia episodes
hypo_mask = glucose < HYPO_THRESHOLD
hyper_mask = glucose > HYPER_THRESHOLD
#glycemic_event_mask = hypo_mask | hyper_mask # Combine masks 
glucose_events = glucose.where(hypo_mask | hyper_mask) # series with the glycemic events (glucose value) and NaN everywhere else

# Plot the parts of the glucose line that are hypo/hyper in red
# Use fill_between for  highlighting glycemic events
ax1.fill_between(time, glucose, HYPER_THRESHOLD, # fill_between will shade the area between the glucose line and the HYPER_THRESHOLD horizontal line
                 where=(glucose > HYPER_THRESHOLD), # when this condition is True, the function will fill the area
                 color='red', alpha=0.3, interpolate=True, zorder=2) # alpha ---> opacity of the area; 
                                                                     # interpolate=True ---> finds the exact point in where the glucose line crosses the threshold line and to start/stop the shading at that intersection; otherwise, it would use the glucose point (the shadowing wouldn't be precise)
ax1.fill_between(time, glucose, HYPO_THRESHOLD,
                 where=(glucose < HYPO_THRESHOLD),
                 color='red', alpha=0.3, interpolate=True, zorder=2)


# Add horizontal dashed lines for glycemic thresholds
ax1.axhline(y=HYPO_THRESHOLD, color='red', linewidth=1.5, zorder=3)
ax1.axhline(y=HYPER_THRESHOLD, color='red', linewidth=1.5, zorder=3)

# Plot meal markers and add them to the legend
ax1.scatter(time[meal_reported.notna()], glucose[meal_reported.notna()], # .notna() because meals were reported as either 0 or 1 or 2 or 3; NaN when there was no meal ---> to translate the vector into True/False values I need to check if the values are not nans
            color='red', marker='^', label='Meal Reported', zorder=6)
ax1.scatter(time[meal_template.astype(bool)], glucose[meal_template.astype(bool)], # .astype(bool) to convert the 0/1 to False/True
           color='yellow', marker='x', label='Meal Template', zorder=6)
ax1.scatter(time[meal_corrected.astype(bool)], glucose[meal_corrected.astype(bool)], 
           color='green', marker='o', label='Meal Corrected', zorder=6)

# Add vertical dashed lines for the start and end of postprandial periods
patient_postprandial = postprandial_df[postprandial_df['subj_id'] == patient_code] # get the postprandial data of the subject considered
for _, row in patient_postprandial.iterrows(): # .iterrows() iterates over the rows of patient_posprndial and returns two values: the index (=_) and the value (=row)
    start = row['meal_time_start']
    end = row['meal_time_end']
    ax1.axvline(x=start, color='gray', linestyle='--', alpha=0.7, zorder=0)
    ax1.axvline(x=end, color='gray', linestyle='--', alpha=0.7, zorder=0)

# Add titles and labels to the axes
ax1.set_title(f'Glucose and Meal Data for Patient {patient_code}')
ax1.set_ylabel('Glucose [mg/dL]')
ax1.set_xlabel('Time')
ax1.legend() # generates the legend


## PLOT 2: HEART RATE ---
ax2 = axs[1]
# Set title and labels for the second subplot
ax2.set_title(f'Heart Rate for Patient {patient_code}')
ax2.set_ylabel('Beats Per Minute [bpm]')
ax2.set_xlabel('Time')

# Check if the heart rate file exists for this patient
if os.path.exists(heart_rate_csv_path):
    # If it exists, load the data
    hr_df = pd.read_csv(heart_rate_csv_path)
    # Convert Time column to datetime
    hr_df['Time'] = pd.to_datetime(hr_df['Time'], errors='coerce')
    # Replace all -1 values with NaN (-1 was put as a placeholder when there was to much motion since the data collected was not reliable)
    hr_df['HEART_RATE'] = hr_df['HEART_RATE'].replace(-1, np.nan)
    nans_time_hr = hr_df['Time'].isna().sum()
    nans_hr = hr_df['HEART_RATE'].isna().sum()
    print(f"\nTotal number of rows with missing 'Time' values in HEART_RATE: {nans_time_hr}")
    print(f"Total number of rows with missing 'HEART_RATE' values in HEART_RATE: {nans_hr}")
    # Plot Heart Rate against Time
    ax2.plot(hr_df['Time'], hr_df['HEART_RATE'], color='purple')
else:
    # If the file doesn't exist, print a message
    print(f"\nHEART_RATE.csv not found for patient {patient_code}.")


## PLOT 3: HEART RATE VARIABILITY ---
ax3 = axs[2]
# Set title and labels for the third subplot
ax3.set_title(f'Heart Rate Variability for Patient {patient_code}')
ax3.set_ylabel('HRV [ms]')
ax3.set_xlabel('Time')

# Check if the HRV file exists for this patient
if os.path.exists(hrv_csv_path):
    # If it exists, load the data
    hrv_df = pd.read_csv(hrv_csv_path)
    # Convert Time column to datetime
    hrv_df['Time'] = pd.to_datetime(hrv_df['Time'], errors='coerce')
    nans_time_hrv = hrv_df['Time'].isna().sum()
    nans_hrv = hrv_df['HEART_RATE_VARIABILITY'].isna().sum()
    print(f"\nTotal number of rows with missing 'Time' values in HEART_RATE_VARIABILITY: {nans_time_hrv}")
    print(f"Total number of rows with missing 'HEART_RATE_VARIBILITY' values in HEART_RATE_VARIABILITY: {nans_hrv}")
    # Plot Heart Rate Variability against Time
    ax3.plot(hrv_df['Time'], hrv_df['HEART_RATE_VARIABILITY'], color='teal')
else:
    # If the file doesn't exist, print a message
    print(f"HEART_RATE_VARIABILITY.csv not found for patient {patient_code}.")


# Display the entire figure with all subplots
plt.tight_layout() # automatically adjusts the spacing between the subplots to ensure that titles and axis labels don't overlap
plt.show()


################################### TIME FRAME SELECTION #######################################

# BASE VARIABLES
meal_to_visualize = 1  
minutes_before = 120    # time window to show before the meal starts 
minutes_after = 200    # time window to show after the meal starts 

# MEAL SELECTION AND TIME WINDOW CALCULATION 
# Filter all postprandial data for the selected patient
patient_meals_df = postprandial_df[postprandial_df['subj_id'] == patient_code].copy() # .copy() to create a new dataFrame from posprandial_df

# Count total meals and print
total_meals = len(patient_meals_df)
print(f"\nPatient {patient_code} has a total of {total_meals} meals recorded.")

# Validate the chosen meal number
if meal_to_visualize > total_meals or meal_to_visualize < 1:
    raise ValueError(f"Invalid meal number. Please choose a number between 1 and {total_meals}.")

# Select the specific meal
selected_meal = patient_meals_df.iloc[meal_to_visualize - 1] # meal_to_visualize - 1 because dataframe indices are 0-based
                                                             # .iloc[] selects the row of interest
selected_meal_time = selected_meal['meal_time_start']
print(f"Visualizing Meal {meal_to_visualize}, which occurred at: {selected_meal_time}")

# Define the time window for the plot
plot_start_time = selected_meal_time - pd.Timedelta(minutes=minutes_before) # pd.Timedelta() allows to perform date and time arithmetic
                                                                            # the input argument is the time to be subtracted ---> creates a duration object of minutes_before minutes
plot_end_time = selected_meal_time + pd.Timedelta(minutes=minutes_after)
print(f"Plotting time window from {plot_start_time} to {plot_end_time}\n")


# FILTER DATA TO THE TIME WINDOW
# Filter the main patient dataframe to the specified window
patient_df_window = patient_df[(patient_df['Time'] >= plot_start_time) & (patient_df['Time'] <= plot_end_time)]

# Extract main data series for plotting from the windowed dataframe
time_window = patient_df_window['Time']
glucose_window = patient_df_window['glucose']
meal_reported_window = patient_df_window['meal']
meal_template_window = patient_df_window['TMA_meal']
meal_corrected_window = patient_df_window['TMA_meal_corrected']



# PLOTS 
fig, axs = plt.subplots(3, 1, figsize=(15, 18)) 
fig.suptitle(f'Analysis of Meal {meal_to_visualize} for Patient {patient_code}', fontsize=16)


## PLOT 1: GLUCOSE AND MEAL EVENTS ---
ax1 = axs[0]
ax1.plot(time_window, glucose_window, color='blue', zorder=1)

# Glycemic event plotting
HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180
hypo_mask_window = glucose_window < HYPO_THRESHOLD
hyper_mask_window = glucose_window > HYPER_THRESHOLD
glucose_events_window = glucose_window.where(hypo_mask_window | hyper_mask_window)

# Use fill_between highlighting glycemic events
ax1.fill_between(time_window, glucose_window, HYPER_THRESHOLD,
                 where=(glucose_window > HYPER_THRESHOLD),
                 color='red', alpha=0.3, interpolate=True, zorder=2)
ax1.fill_between(time_window, glucose_window, HYPO_THRESHOLD,
                 where=(glucose_window < HYPO_THRESHOLD),
                 color='red', alpha=0.3, interpolate=True, zorder=2)

ax1.axhline(y=HYPO_THRESHOLD, color='red', linewidth=1.5, zorder=3)
ax1.axhline(y=HYPER_THRESHOLD, color='red', linewidth=1.5, zorder=3)

# Plot meal markers from the windowed data
ax1.scatter(time_window[meal_reported_window.notna()], glucose_window[meal_reported_window.notna()],
            color='red', marker='^', label='Meal Reported', zorder=6)
ax1.scatter(time_window[meal_template_window.astype(bool)], glucose_window[meal_template_window.astype(bool)],
            color='yellow', marker='x', label='Meal Template', zorder=6)
ax1.scatter(time_window[meal_corrected_window.astype(bool)], glucose_window[meal_corrected_window.astype(bool)],
            color='green', marker='o', label='Meal Corrected', zorder=6)

# Add vertical lines for any postprandial periods that fall within the window
for _, row_window in patient_meals_df.iterrows():
    if row_window['meal_time_start'] >= plot_start_time and row_window['meal_time_end'] <= plot_end_time:
        ax1.axvline(x=row_window['meal_time_start'], color='gray', linestyle='--', alpha=0.7, zorder=0)
        ax1.axvline(x=row_window['meal_time_end'], color='gray', linestyle='--', alpha=0.7, zorder=0)

ax1.set_title('Glucose and Meal Events')
ax1.set_xlabel('Time')
ax1.set_ylabel('Glucose [mg/dL]')
ax1.legend()

date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax1.xaxis.set_major_formatter(date_format)


## PLOT 2: HEART RATE ---
ax2 = axs[1]
ax2.set_title('Heart Rate')
ax2.set_xlabel('Time')
ax2.set_ylabel('Beats Per Minute [bpm]')

date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax2.xaxis.set_major_formatter(date_format)

if os.path.exists(heart_rate_csv_path):
    hr_df = pd.read_csv(heart_rate_csv_path)
    hr_df['Time'] = pd.to_datetime(hr_df['Time'], errors='coerce')
    hr_df['HEART_RATE'] = hr_df['HEART_RATE'].replace(-1, np.nan)
    # Filter to the time window
    hr_df_window = hr_df[(hr_df['Time'] >= plot_start_time) & (hr_df['Time'] <= plot_end_time)]
    ax2.plot(hr_df_window['Time'], hr_df_window['HEART_RATE'], color='purple')
else:
    print(f"\nHEART_RATE.csv not found for patient {patient_code}.")


## PLOT 3: HEART RATE VARIABILITY ---
ax3 = axs[2]
ax3.set_title('Heart Rate Variability')
ax3.set_xlabel('Time')
ax3.set_ylabel('HRV [ms]')

# Create the date formatter
# %Y = Year, %m = month, %d = day, %H = hour, %M = minute
date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax3.xaxis.set_major_formatter(date_format)

if os.path.exists(hrv_csv_path):
    hrv_df = pd.read_csv(hrv_csv_path)
    hrv_df['Time'] = pd.to_datetime(hrv_df['Time'], errors='coerce')
    # Filter to the time window
    hrv_df_window = hrv_df[(hrv_df['Time'] >= plot_start_time) & (hrv_df['Time'] <= plot_end_time)]
    ax3.plot(hrv_df_window['Time'], hrv_df_window['HEART_RATE_VARIABILITY'], color='teal')
else:
    print(f"\nHEART_RATE_VARIABILITY.csv not found for patient {patient_code}.")

# Final display
plt.tight_layout() 
plt.show()


#################################### HYPOGLYCEMIC EVENTS #######################################

# BASE VARIABLES
hypo_event_to_visualize = 75
minutes_before_hypo =120 # minutes before the BEGINNING of the event
minutes_after_hypo = 200 # minutes after the END of the event

# HYPOGLYCEMIC EVENT DETECTION
# Create a new, separate DataFrame for the hypoglycemic events
hypo_df = patient_df[['Time', 'glucose']].copy()

# Create a boolean column that is True if glucose is in hypoglycemia
hypo_df['is_hypo'] = hypo_df['glucose'] < HYPO_THRESHOLD

# Identify blocks of continuous hypoglycemia ---> will have True only at the beginning of a new event
# A new event starts when 'is_hypo' changes from False to True
hypo_df['hypo_event_start'] = (hypo_df['is_hypo'] & ~hypo_df['is_hypo'].shift(1).fillna(False)) # .shift(1) shifts the hypo_df of one row ---> it goes to the previous row
                                                                                                # .fillna(False) to fill the first row (which has become NaN) with False (because when I shift of one, the first row will be empty because there's no data before time 0)
# Assign a unique ID to each continuous event
hypo_df['hypo_event_id'] = hypo_df['hypo_event_start'].cumsum() # .cumsum() will put the same event_id on all the rows until it encounters a new hypo_event_start and will add 1

# Filter to only include the actual hypoglycemic periods
hypo_periods = hypo_df[hypo_df['is_hypo']]

# Calculate the start time and duration for each unique event
hypo_events_summary = hypo_periods.groupby('hypo_event_id').agg( # .groupby(...) gorups the hypo_events by their id
                                                                 # .agg() to run multiple operations in one go to the data
    start_time=('Time', 'first'), # create a new column with the start time by looking at the "Time" column of the group and selecting the "first"
    end_time=('Time', 'last') # create a new column with the end time by looking at the "Time" column of the group and selecting the "last"
).reset_index() # after a groupby operation, the group labels become the index of the new summary DataFrame ---> .reset_index() converts this index back into a regular column

# Calculate duration and filter out event ID 0 (if it exists, it means data started in hypo)
#hypo_events_summary = hypo_events_summary[hypo_events_summary['hypo_event_id'] > 0]
hypo_events_summary['duration'] = hypo_events_summary['end_time'] - hypo_events_summary['start_time']

# Save total number of events
total_hypo_events = len(hypo_events_summary)
print(f"\nA total of {total_hypo_events} hypoglycemic events were found for patient {patient_code}\n")


# EVENT SELECTION FOR PLOTTING
if total_hypo_events == 0:
    print("No hypoglycemic events to visualize.")
else:
    # Validate the chosen event number
    if hypo_event_to_visualize > total_hypo_events or hypo_event_to_visualize < 1:
        raise ValueError(f"Invalid event number. Please choose a number between 1 and {total_hypo_events}.")

    # Select the specific event
    selected_event = hypo_events_summary.iloc[hypo_event_to_visualize - 1] #iloc[] selects the row of interest
    selected_event_time = selected_event['start_time']
    selected_event_duration = selected_event['duration']
    print(f"Visualizing Hypoglycemic Event {hypo_event_to_visualize}, which started at: {selected_event_time} and lasted {selected_event_duration} minutes")

    # Define the time window for the plot
    plot_start_time = selected_event_time - pd.Timedelta(minutes=minutes_before_hypo)
    plot_end_time = selected_event_time + pd.Timedelta(minutes=minutes_after_hypo)
    print(f"Plotting time window from {plot_start_time} to {plot_end_time}\n")

    # FILTER ALL DATA TO THE SELECTED TIME WINDOW 
    patient_df_hypo_window = patient_df[(patient_df['Time'] >= plot_start_time) & (patient_df['Time'] <= plot_end_time)]
    time_hypo_window = patient_df_hypo_window['Time']
    glucose_hypo_window = patient_df_hypo_window['glucose']
    meal_reported_hypo_window = patient_df_hypo_window['meal']
    meal_template_hypo_window = patient_df_hypo_window['TMA_meal']
    meal_corrected_hypo_window = patient_df_hypo_window['TMA_meal_corrected']

    # PLOT
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(f'Analysis of Hypoglycemic Event {hypo_event_to_visualize} for Patient {patient_code}', fontsize=16)

    ## PLOT 1: GLUCOSE AND MEAL EVENTS ---
    ax1 = axs[0]
    ax1.plot(time_hypo_window, glucose_hypo_window, color='blue', zorder=1)

    ax1.fill_between(time_hypo_window, glucose_hypo_window, HYPER_THRESHOLD, where=(glucose_hypo_window > HYPER_THRESHOLD), color='red', alpha=0.3, interpolate=True, zorder=2)
    ax1.fill_between(time_hypo_window, glucose_hypo_window, HYPO_THRESHOLD, where=(glucose_hypo_window < HYPO_THRESHOLD), color='red', alpha=0.3, interpolate=True, zorder=2)
    
    ax1.axhline(y=HYPO_THRESHOLD, color='red', linewidth=1.5, zorder=3)
    ax1.axhline(y=HYPER_THRESHOLD, color='red', linewidth=1.5, zorder=3)
    
    ax1.scatter(time_hypo_window[meal_reported_hypo_window.notna()], glucose_hypo_window[meal_reported_hypo_window.notna()], color='red', marker='^', label='Meal Reported', zorder=6)
    ax1.scatter(time_hypo_window[meal_template_hypo_window.astype(bool)], glucose_hypo_window[meal_template_hypo_window.astype(bool)], color='yellow', marker='x', label='Meal Template', zorder=6)
    ax1.scatter(time_hypo_window[meal_corrected_hypo_window.astype(bool)], glucose_hypo_window[meal_corrected_hypo_window.astype(bool)], color='green', marker='o', label='Meal Corrected', zorder=6)
    
    # Add vertical lines for any postprandial periods that fall within the window
    for _, row_hypo in patient_meals_df.iterrows():
        if row_hypo['meal_time_start'] >= plot_start_time and row_hypo['meal_time_end'] <= plot_end_time:
            ax1.axvline(x=row_hypo['meal_time_start'], color='gray', linestyle='--', alpha=0.7, zorder=0)
            ax1.axvline(x=row_hypo['meal_time_end'], color='gray', linestyle='--', alpha=0.7, zorder=0)

    ax1.set_title('Glucose and Meal Events')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Glucose [mg/dL]')
    ax1.legend()
    
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax1.xaxis.set_major_formatter(date_format)

    ## PLOT 2: HEART RATE ---
    ax2 = axs[1]
    ax2.set_title('Heart Rate')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Beats Per Minute [bpm]')
    
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax2.xaxis.set_major_formatter(date_format)
    
    if os.path.exists(heart_rate_csv_path):
        hr_df = pd.read_csv(heart_rate_csv_path)
        hr_df['Time'] = pd.to_datetime(hr_df['Time'], errors='coerce')
        hr_df['HEART_RATE'] = hr_df['HEART_RATE'].replace(-1, np.nan)
        hr_df_hypo_window = hr_df[(hr_df['Time'] >= plot_start_time) & (hr_df['Time'] <= plot_end_time)]
        ax2.plot(hr_df_hypo_window['Time'], hr_df_hypo_window['HEART_RATE'], color='purple')
    else:
        print(f"\nHEART_RATE.csv not found for patient {patient_code}.")
        
    ## PLOT 3: HEART RATE VARIABILITY ---
    ax3 = axs[2]
    ax3.set_title('Heart Rate Variability')
    ax3.set_ylabel('HRV [ms]')
    ax3.set_xlabel('Time')
    
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax3.xaxis.set_major_formatter(date_format)
    
    if os.path.exists(hrv_csv_path):
        hrv_df = pd.read_csv(hrv_csv_path)
        hrv_df['Time'] = pd.to_datetime(hrv_df['Time'], errors='coerce')
        hrv_df_hypo_window = hrv_df[(hrv_df['Time'] >= plot_start_time) & (hrv_df['Time'] <= plot_end_time)]
        ax3.plot(hrv_df_hypo_window['Time'], hrv_df_hypo_window['HEART_RATE_VARIABILITY'], color='teal')
    else:
        print(f"\nHEART_RATE_VARIABILITY.csv not found for patient {patient_code}.")

    # Final display
    plt.tight_layout()
    plt.show()




















