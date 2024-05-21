
# -----------------------------------------------------
# nhts.py
# -----------------------------------------------------
# Description: Contains the helper functions used to clean and preprocess the NHTS data.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.nhts"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import pandas as pd
import datetime
import datetime as dt
import logging
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

#Get today
year = int(dt.datetime.today().strftime('%Y'))
month = int(dt.datetime.today().strftime('%m'))
day = int(dt.datetime.today().strftime('%d'))

# Predict number of cars based on 
variables_of_interest = {
    "VIKT_individ" : {
            "type":"numerical",
            "categories": {}
    }, # Weight of individual
    # LPNR - Unique ID
    "LPNR": {
            "type":"numerical",
            "categories": {}
    },
    
    # Weight of individual
    "Kön":{
            "type":"categorical",
            "categories": {
                1: "Kvinnor",
                2.0: "Män",
                3: "Other"
            }
    },              
    # Gender
    "Åldersgrupp":{
            "type":"categorical",
            "categories": {
                1.0 :	"16-24",
                2.0 :	"25-34",
                3.0 :	"35-44",
                4.0 :	"45-54",
                5.0 :	"55-64",
                6.0 :	"65-74",
                7.0 :	"75+"

            }

    },            
    # Age
    "Bostadstyp":{
            "type":"categorical",
            "categories": {
                1: "Apartment",
                2: "Villa",
                3: "Other",

    }
    },       
    # Type of residence
    "Antal_barn":{
            "type":"numerical",
            "categories": {}

    },       
    # Number of children
    "Antal_vuxna" : {
            "type":"numerical",
            "categories": {}

    },      
    # Number of adults
    "Hushållstyp" : {
            "type":"categorical",
            "categories": {
                1.0 : "Single", # no child
                2.0 : "Single", # with child 
                3.0 : "Couple", # no child
                4.0 : "Couple", # with child
                5.0 : "Other"


    }
    },      
    # Type of household
    "Antal_bilar" : {
            "type":"categorical",
            "categories": {
                0: 0, # "No car",
                1: 1, # "One car",
                2: 2 # "Two cars or more",

    }       # Number of cars - This we want to predict eventually
        },
        # Start time
    "Starttid" : {
            "type":"time",
            "categories": {}
        },
        # Activity
    "Arende" : {
            "type":"categorical",
            "categories": {
                1: "Work",
                2: "Travel",
                3: "Education",
                4: "Pickup/Dropoff child",
                5: "Grocery", # Changed from "Grocery shopping"
                6: "Shopping", # Changed from "Other shopping"
                7: "Leisure",
                8: "Home",
                9: "Other",
                10: "Healthcare"
    }
        }, # Mode
    "Huvud_fm" : {
            "type":"categorical",
            "categories": {
                1: "Boat",
                2: "Flight",
                3: "Other",
                4: "Transportation service",
                5: "Train/Tram",
                6: "Bus",
                7: "Taxi",
                8: "Car",
                9: "Moped",
                10: "Bicycle/E-bike",
                11: "Walking",
            }
        }, # End time
    "Sluttid" : {
            "type":"time",
            "categories": {}
        }, # Distance in km
    "Reslängd" : {
            "type":"numerical",
            "categories": {}
    },
    # Distance in km
    'VARDAG_HELG': {
            "type":"numerical",
            "categories": {}
    },
    # Weekday or weekend
    'Antal_resor_per_pers' : {
            "type":"numerical",
            "categories": {}
    },
    # Number of trips per person
      
}



def preprocess_data(df,variables_of_interest = variables_of_interest, weekday = True, unique_trips_only = False, process_durations = False):
    """
    Preprocess the NHTS data.

    Args:
        df (pd.DataFrame): The input dataframe containing NHTS data.
        variables_of_interest (dict): A dictionary specifying the variables of interest and their types (categorical, numerical, time).
        weekday (bool): If True, processes data for weekdays; if False, processes data for weekends. Defaults to True.
        unique_trips_only (bool): If True, removes duplicate trips based on id, start_time, and end_time. Defaults to False.
        process_durations (bool): If True, processes the durations of activities and travels. Defaults to False.

    Returns:
        pd.DataFrame: The preprocessed dataframe.

    Notes:
        - The function replaces specific values indicating no data with NaN and drops rows with NaN values.
        - It filters the dataframe to include only the specified variables of interest.
        - The function processes categorical, numerical, and time variables according to their specified types.
        - It renames columns to standardize names and drops certain columns not needed for further analysis.
        - If `unique_trips_only` is True, duplicate trips are removed.
        - If `process_durations` is True, the function calculates the travel and activity durations.

    Examples:
        >>> variables_of_interest = {
        ...     'Kön': {'type': 'categorical', 'categories': {1: 'Male', 2: 'Female'}},
        ...     'Åldersgrupp': {'type': 'categorical', 'categories': {1: '0-17', 2: '18-34', 3: '35-64', 4: '65+'}},
        ...     'Starttid': {'type': 'time'},
        ...     'Sluttid': {'type': 'time'},
        ...     'Reslängd': {'type': 'numerical'},
        ...     # additional variables...
        ... }
        >>> df = preprocess_data(df, variables_of_interest, weekday=True, unique_trips_only=True, process_durations=True)
    """

    if weekday:
        wd = 1
    else:
        wd = 2



    # Replace ',' with '.' in 'Reslängd'
    df['Reslängd'] = df['Reslängd'].str.replace(',', '.')
    # Replace ',' with '.' in 'VIKT_individ'
    df['VIKT_individ'] = df['VIKT_individ'].str.replace(',', '.')
    # Filter df to only include variables of interest
    df = df[list(variables_of_interest.keys())].reset_index(drop=True)
    # Replace no data with NaN (-111, 99998, '', blank)
    df = df.replace([-111, 999998, 99998, '', ' '], float('NaN'))
    # Drop rows with NaN
    df = df.dropna().reset_index(drop=True)
    # Sort by age
    #VARDAG_HELG == 1 for weekday
    #VARDAG_HELG == 2 for weekend
    df = df[df['VARDAG_HELG'] == wd].reset_index(drop=True)
    #Antal_resor_per_pers > 1
    df = df[df['Antal_resor_per_pers'] > 1].reset_index(drop=True)
    #TODO - Remove entries for an LPNR with "Home" not the last trip
    #Arende == 8 not the last trip

    # Replace categorical values with strings
    for variable in variables_of_interest:
        if variables_of_interest[variable]['type'] == 'categorical':
            # Change data type to int
            df[variable] = df[variable].astype(float)
            # Replace values with strings
            df[variable] = df[variable].replace(variables_of_interest[variable]['categories'])
        elif variables_of_interest[variable]['type'] == 'numerical':
            # Change data type to int
            df[variable] = df[variable].astype(float)
        elif variables_of_interest[variable]['type'] == 'time':
            # Change data type to datetime with format HH:MM:SS
            df[variable] = pd.to_datetime(df[variable], format='%H:%M:%S')
            # Only show hour and minute
            #df[variable] = df[variable].dt.strftime('%H:%M')
            # Set year, month and day to Today from datetime
            df[variable] = df[variable].apply(lambda dt: dt.replace(year=year, month=month, day=day))

    # Rename columns - kön to kon Ålder to alder
    df = df.rename(columns={
        'LPNR': 'id',
        'Kön': 'sex',
        'Åldersgrupp': 'age_group',
        'Bostadstyp' : 'house_type',
        'Hushållstyp' : 'household_type',
        'Antal_barn' : 'child_count',
        'Antal_bilar' : 'car_count',
        'Antal_vuxna' : 'adult_count',
        'Starttid' : 'start_time',
        'Huvud_fm' : 'mode',
        'Sluttid' : 'end_time',
        'Reslängd' : 'distance_km',
        'Arende' : 'purpose',
        'VIKT_individ' : 'weight_individual'
        })

    #Drop 'VARDAG_HELG' and 'Antal_resor_per_pers'
    df = df.drop(['VARDAG_HELG', 'Antal_resor_per_pers'], axis=1)


    df = df.sort_values(by=['id', 'start_time'])
    # Create a column called activity_sequence which is the activity number starting from 0 for each id
    df['activity_sequence'] = df.groupby('id').cumcount()
    # create a column called duration of previous activity
    # This is the time between end_time of previous activity and start_time of current activity
    # If there is no previous activity, then duration is 0
    # If there is no next activity, then duration is 0
    # Sort dataframe based on id and activity_sequence
    df = df.sort_values(by=['id', 'start_time'])
    if unique_trips_only:
        # Drop duplicates based on id, start_time and end_time
        df = df.drop_duplicates(subset=['id', 'start_time', 'end_time']).reset_index(drop=True)
    
    if process_durations:
        # Create the necessary columns for the entire dataframe
        df['travel_duration_minutes'] = df['end_time'] - df['start_time']

        # Shift the start times within each group
        df['next_travel_start_time'] = df.groupby('id')['start_time'].shift(-1)

        # Calculate activity durations
        df['activity_duration_minutes'] = df['next_travel_start_time'] - df['end_time']

        # Fill NaN values in 'next_travel_start_time' and 'activity_duration_minutes' if required
        df['next_travel_start_time'] = df['next_travel_start_time'].fillna(method='ffill')
        df['activity_duration_minutes'] = df['activity_duration_minutes'].fillna(pd.Timedelta(seconds=0))
    return df




