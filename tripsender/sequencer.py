
# -----------------------------------------------------
# sequencer.py
# -----------------------------------------------------
# Description: Contains the helper functions used to process activity data.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.sequencer"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import json
import random
import logging
from datetime import datetime
import pandas as pd
import tripsender.nhts as nhts
from tripsender.person import Person
from tripsender.household import Household
from tripsender.activity import ActivitySequence, Activity
from tripsender.sampler import DurationSampler, StartTimeSampler
from tripsender import sequencer_c as sc

# Constants and initial data loading

CHILDREN_ACTIVITY_SCHEMA_PATH = 'models/CHILDREN_ACTIVITY.json'
START_TIME_DIST_PATH = "models/START_TIME_DIST.json"
DURATION_DIST_PATH = "models/DURATION_DIST.json"
NHTS_PATH = 'data/raw/NHTS/Data_RVU_2017_GBG_utanEXTRA.csv'
with open(CHILDREN_ACTIVITY_SCHEMA_PATH, encoding='utf-8') as f:
    CHILDREN_ACTIVITY_SCHEMA = json.load(f)
with open(START_TIME_DIST_PATH, encoding='utf-8') as f:
    START_TIME_DIST = json.load(f)
with open(DURATION_DIST_PATH, encoding='utf-8') as f:
    DURATION_DIST = json.load(f)



from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

duration_sampler = DurationSampler(DURATION_DIST)
start_time_sampler = StartTimeSampler(START_TIME_DIST)


def generate_individual_adult_sequence(adult, cars_available):
    primary_status = adult.primary_status
    if not primary_status:
        raise ValueError(f'Primary status is None for {adult} in {adult.household}')
    elif primary_status != "HOME":
        duration = duration_sampler.sample_duration(primary_status, min_duration=30, max_duration=576)
        start_time = start_time_sampler.sample_start_time(primary_status)
        mode, cars_available = select_mode_of_transport(adult, cars_available)
        sequence = ActivitySequence()
        sequence.add_activity(Activity(start_time, duration, primary_status, mode))
    else:
        sequence = ActivitySequence()
        #TODO Mode for home activities is not defined
    adult.activity_sequence = sequence
    return adult, cars_available


def select_mode_of_transport(adult, cars_available):
    if adult.has_car and cars_available:
        return "CAR", cars_available - 1
    return random.choice(["WALK", "BIKE", "PUBLIC"]), cars_available
#TODO There is a possibility of adding a more nuanced way to do this


def check_children_activities(household):
    pd_demand, l_demand = [], []
    for child in [person for person in household.members if person.age > 0 and person.age < 19]:
        child_sequence = sc.generate_child_activity_sequence(child.age, CHILDREN_ACTIVITY_SCHEMA)
        child.activity_sequence = child_sequence
        if not child_sequence:
            continue
        for activity in child_sequence.activities:
            if activity.mode == 'PARENT':
                if activity.purpose.startswith('EDUCATION'):
                    logger.info(f"Childs education activity {activity}")
                    pd_demand.append(activity)
                elif activity.purpose.startswith('LEISURE'):
                    logger.info(f"Childs lesiure activity {activity}")
                    l_demand.append(activity)
    return pd_demand, l_demand


def orchestrate_household_activity(adult_sequences, demand_list):
    # TODO This is a very simple orchestration, it should be improved
    for demand in demand_list:
        best_adult_sequence = _find_best_adult_for_child(demand, adult_sequences)
        #TODO Change demand.mode to the mode of the adult if there are activities in the adult sequence
        best_adult_sequence.add_activity(demand)
        #TODO Adults are also being forced to the same activity as the child, this is not realistic
    return adult_sequences


def _calculate_overlap(child_activity, adult_activity):
    # If the child activity ends before the adult activity starts or starts after it ends, they don't overlap
    if child_activity.end_time <= adult_activity.start_time or child_activity.start_time >= adult_activity.end_time:
        return 0
    # Calculate the overlapping time
    start_overlap = max(child_activity.start_time, adult_activity.start_time)
    end_overlap = min(child_activity.end_time, adult_activity.end_time)

    overlap_minutes = (datetime.combine(datetime.today(), end_overlap) - datetime.combine(datetime.today(), start_overlap)).total_seconds() / 60
    return overlap_minutes

def _find_best_adult_for_child(child_activity, adult_sequences):
    least_loss = float('inf')
    best_adult = None
    
    for adult_sequence in adult_sequences:
        total_loss = sum([_calculate_overlap(child_activity, activity) for activity in adult_sequence.activities])
        if total_loss < least_loss:
            least_loss = total_loss
            best_adult = adult_sequence
            
    return best_adult

def generate_activity_sequences_for_household(household):
    cars_available = household.cars
    adult_sequences = []

    for adult in household.members:
        if adult.age > 18:
            adult, cars_available = generate_individual_adult_sequence(adult, cars_available)
            adult_sequences.append(adult.activity_sequence)

    if household.children:
        pd_demand, l_demand = check_children_activities(household)
        if pd_demand:
            adult_sequences = orchestrate_household_activity(adult_sequences, pd_demand)
        if l_demand:
            adult_sequences = orchestrate_household_activity(adult_sequences, l_demand)
    # Close the activity sequence
    for adult_sequence in adult_sequences:
        adult_sequence.close_activity_sequence()


def generate_activity_sequence(households=Household.instances, verbose=True):
    if verbose:
        logger.setLevel(logging.DEBUG)
    for household in households:
        generate_activity_sequences_for_household(household)


# Import necessary libraries
import pandas as pd
import numpy as np
import random
# Define a function to remap household_type categories
def remap_household_type(household_type):
    # Mapping dictionary
    mapping = {
        'Barn': 'Other',
        'Ej ensamboende personer, övriga': 'Other',
        'Ensamboende': 'Single',
        'Ensamstående förälder': 'Single',
        'Person i gift par/registrerat partnerskap': 'Couple',
        'Personer i samboförhållande': 'Couple'
    }
    # Return the remapped value or 'Other' if not found
    return mapping.get(household_type, 'Other')

def remap_age(age):
    """Maps an age to an age group."""
    if 16 <= age <= 24:
        return '16-24'
    elif 25 <= age <= 34:
        return '25-34'
    elif 35 <= age <= 44:
        return '35-44'
    elif 45 <= age <= 54:
        return '45-54'
    elif 55 <= age <= 64:
        return '55-64'
    elif 65 <= age <= 74:
        return '65-74'
    elif age >= 75:
        return '75+'
    else:
        return None  # For ages below 16 or invalid ages

def prepare_activity_dataframes():
    # Prepare the synthetic adults DataFrame
    df_synthetic_adults = Person.return_adults_df()[["sex", "age", "house_type", "has_child", "household_type", "has_car", "primary_status","Person"]]

    # Prepare the survey adults DataFrame
    df_survey_adults = ActivitySequence.return_person_df()[["sex", "age_group", "house_type", "child_count", "household_type", "car_count", "is_worker", "ActivitySequence"]]

    # Turn primary_status into a boolean is_worker in survey DataFrame and drop the now redundant primary_status column
    df_synthetic_adults["is_worker"] = df_synthetic_adults["primary_status"] == "WORK"
    df_synthetic_adults.drop(columns=["primary_status"], inplace=True)

    # Convert car_count to boolean has_car in survey DataFrame
    df_survey_adults["has_car"] = df_survey_adults["car_count"] > 0
    # Drop the now redundant car_count column
    df_survey_adults.drop(columns=["car_count"], inplace=True)

    # Convert age to age_group in synthetic DataFrame using a predefined remap_age function
    df_synthetic_adults['age_group'] = df_synthetic_adults['age'].apply(remap_age)
    # Drop the now redundant age column
    df_synthetic_adults.drop(columns=["age"], inplace=True)

    # Apply the remap function to the household_type column in synthetic DataFrame
    df_synthetic_adults['household_type'] = df_synthetic_adults['household_type'].apply(remap_household_type)

    # Convert has_car to boolean in synthetic DataFrame (assuming 1 represents True)
    df_synthetic_adults["has_car"] = df_synthetic_adults["has_car"] == 1

    # Create has_child column in survey DataFrame from child_count
    df_survey_adults["has_child"] = df_survey_adults["child_count"] > 0
    # Drop the now redundant child_count column
    df_survey_adults.drop(columns=["child_count"], inplace=True)

    return df_synthetic_adults, df_survey_adults

def filter_survey_df(survey_df, synthetic_row, attributes):
    """Filter the survey DataFrame based on the values in a synthetic row."""
    conditions = [survey_df[attr] == synthetic_row[attr] for attr in attributes]
    return survey_df[np.logical_and.reduce(conditions)]

import copy
def assign_activity_sequences(df_synthetic, df_survey, attributes_order, min_matches=30):
    for index, synthetic_row in df_synthetic.iterrows():

        # Iteratively try matching with decreasing number of attributes
        for i in range(len(attributes_order), 0, -1):
            current_attributes = attributes_order[:i]

            matched_df = filter_survey_df(df_survey, synthetic_row, current_attributes)

            # If sufficient matches are found, assign an activity sequence
            if len(matched_df) >= min_matches:
                selected_sequence = matched_df['ActivitySequence'].sample(n=1).iloc[0]
                person = synthetic_row['Person']
                activity_sequence = copy.deepcopy(selected_sequence)
                person.activity_sequence = activity_sequence
                activity_sequence.person = person
                break


def process_activity_sequence(NHTS_PATH = NHTS_PATH):
    """ A function to read NHTS data and generate meaningful activity sequences and add it to the ActivitySequence.instances list"""

    logger.info("Starting activity sequence assignment...")
    # Clear previous instances
    ActivitySequence.instances = []
    # Read data
    nhts_data = pd.read_csv(NHTS_PATH, sep=';')
    # Preprocess data
    df_persons = nhts.preprocess_data(nhts_data,process_durations = True, weekday=True)
    # Setup counters
    pass_counter = 0
    fail_counter = 0
    
    logger.info("Generating activity sequences for all persons in NHTS data...")
    for uid in df_persons['id'].unique():
        activity_sequence = ActivitySequence()
        df = df_persons[df_persons['id'] == uid]
        passing = activity_sequence.from_nhts(df)
        if passing:
            pass_counter += 1
        else:
            fail_counter += 1

    logger.info("Generated activity sequences for all persons in NHTS data.")
    logger.info(f"Passing activity sequences: {pass_counter}")
    logger.info(f"Failing activity sequences: {fail_counter}")
    logger.info(f"Total activity sequences: {pass_counter + fail_counter}")
    logger.info(f"Passing %: {pass_counter/(pass_counter + fail_counter)}")


def assign_activity_sequence_to_synthetic_population(min_matches = 20):
    """ A function to assign activity sequences to the synthetic population"""
    # Get both surveyed and synthetic dataframes
    logger.info("Preparing dataframes from synthetic population...")
    df_synthetic_adults, df_survey_adults = prepare_activity_dataframes()

    # Define the order of attributes to match in terms of priority
    attributes_order = ['is_worker','age_group', 'sex', 'has_child', 'has_car', 'house_type', 'household_type']

    # Assign activity sequences. 
    logger.info("Assigning activity sequences to synthetic population...")
    assign_activity_sequences(df_synthetic_adults, df_survey_adults, attributes_order, min_matches=min_matches)

    logger.info("Activity sequence assignment complete.")

