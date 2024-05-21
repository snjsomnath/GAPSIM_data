
# -----------------------------------------------------
# sequencer_c.py
# -----------------------------------------------------
# Description: Contains the helper functions used to process childrens activity data.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.sequencer_c"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import json
import random
import numpy as np
from datetime import datetime, timedelta
import logging

from tripsender.activity import ActivitySequence, Activity

# Constants
FRITIDSHEM_PROBABILITY = 0.2
FRITIDSHEM_END_TIME = "1600"
TIME_FORMAT = "%H%M"

from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

def time_to_datetime(timestring):
    return datetime.strptime(timestring, TIME_FORMAT)

def datetime_to_time(dt_object):
    return dt_object.strftime(TIME_FORMAT)

def choose_mode_of_transport(age, mode):
    if age <= 5:
        return "PARENT"
    else:
        if mode == "linear":
            prob_parent = 1 - 0.06 * age
        elif mode == "exponential":
            prob_parent = np.exp(-0.2 * age)
        else:
            raise ValueError("mode must be either linear or exponential")

        return random.choices(["PARENT", "PUBLIC"], weights=[prob_parent, 1-prob_parent], k=1)[0]

def identify_age_group(age, activities):
    for group in activities:
        if age in group.get("age", []):
            #logging.info(f'Step 1 - Age is {age}')
            return group
    return None

def get_mandatory_activity(group):
    options = group.get("mandatory_activity", [])
    if not options:
        raise ValueError("No mandatory activity provided for the age group.")
        
    chosen = random.choices(options, weights=[o.get("probability", 1) for o in options], k=1)[0]
    #logging.info(f'Step 2 - Mandatory activity chosen as {chosen["name"]}')

    time_range = group.get("time_range", {})
    start_datetime = time_to_datetime(time_range.get("start", "0000"))
    duration = group.get("duration", 0)
    end_datetime = time_to_datetime(time_range.get("end", "2400")) - timedelta(hours=duration)
    
    random_start_time = start_datetime + (end_datetime - start_datetime) * random.random()
    random_start_time = random_start_time - timedelta(minutes=random_start_time.minute % 15)
    #logging.info(f'Step 3 - Random start time chosen as {random_start_time.strftime("%H:%M")}')
    
    end_time = random_start_time + timedelta(hours=duration)
    #logging.info(f'Step 4 - End time calculated as {end_time.strftime("%H:%M")}')
    return chosen["name"], random_start_time, end_time

def handle_fritidshem(age, start_time, mode_of_transport, activities):
    if age <= 5 and random.random() < FRITIDSHEM_PROBABILITY:
        #logging.info(f'Step 5 - Fritidshem chosen since age is {age}')
        
        end_time_for_activity = (start_time + timedelta(minutes=15)).time()
        
        activity_start = Activity(datetime_to_time(start_time), 15, "EDUCATION_fritidshem", mode_of_transport)
        activities.add_activity(activity_start)
        
        end_time = time_to_datetime(FRITIDSHEM_END_TIME)
        duration_until_end = (end_time - start_time - timedelta(minutes=15)).seconds // 60
        
        activity_end = Activity(end_time_for_activity, duration_until_end, "EDUCATION_fritidshem", mode_of_transport)
        activities.add_activity(activity_end)
        
        return activities, end_time

    return activities, start_time


def after_school_activities(group, end_time, mode_of_transport, activities):
    option = random.choice(group.get("after_school", []))
    activity_name = option.get("activity", "None")
    #logging.info(f'Step 6 - After school activity chosen as {activity_name}')
    
    if activity_name == "None":
        return activities, end_time
    else:
        duration_minutes = random.randint(
            int(option.get("duration", 0)) - int(option.get("variance", 0)),
            int(option.get("duration", 0)) + int(option.get("variance", 0))
        )
        duration = timedelta(minutes=duration_minutes)
        #logging.info(f'Step 7 - Duration of after school activity chosen as {duration_minutes} minutes')
        activity = Activity(datetime_to_time(end_time), duration_minutes, activity_name, mode_of_transport)
        activities.add_activity(activity)
        end_time += duration
        return activities, end_time

def generate_child_activity_sequence(age, data, mode_method="linear"):
    # Skip if age is 0
    if age == 0:
        #logging.warning("Age is 0, skipping the sequence generation.")
        return None
    activities = ActivitySequence()
    mode_of_transport = choose_mode_of_transport(age, mode_method)
    age_group = identify_age_group(age, data.get("activities", []))
    if not age_group:
        raise ValueError(f"No age group configuration found for age {age}.")
    activity_name, start_time, end_time = get_mandatory_activity(age_group)
    duration_minutes = (datetime.combine(datetime.today(), end_time.time()) - datetime.combine(datetime.today(), start_time.time())).seconds // 60
    activity = Activity(datetime_to_time(start_time), duration_minutes, activity_name, mode_of_transport)
    activities.add_activity(activity)
    activities, new_end_time = handle_fritidshem(age, end_time, mode_of_transport, activities)
    end_time = new_end_time
    activities, end_time = after_school_activities(age_group, end_time, mode_of_transport, activities)
    activities.close_activity_sequence()
    return activities
