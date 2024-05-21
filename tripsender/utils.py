
# -----------------------------------------------------
# utils.py
# -----------------------------------------------------
# Description: Contains the helper functions used to process synthetic population data.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.utils"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import requests
import json
import random
import re
import numpy as np
import uuid
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder as ohe

# Importing classes
from tripsender.population import Population
from tripsender.person import Person, age_group_from_age
from tripsender.household import Household
from tripsender.house import House
from tripsender.building import Building
from tripsender.activity import ActivitySequence

# Importing functions
from tripsender.fetcher import *

from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

# Constants
# TODO Move these to a config file and do the replacements here
MALE = 'Män'
FEMALE = 'Kvinnor'
COHABITING = 'Personer i samboförhållande'
MARRIED = 'Person i gift par/registrerat partnerskap'
LIVING_ALONE = 'Ensamboende'
SINGLE_PARENT = 'Ensamstående förälder'
SINGLE = 'Ensamstående'
CHILD = 'Barn'
OTHER = 'Övriga'
COUPLE = 'Sammanboende'

PATH_PRIMARY_AREA = "data/primary_area.csv"
def search_primary_area(query,df = pd.read_csv(PATH_PRIMARY_AREA)):
    """
    Search for the best matching primary area from the DataFrame based on the query.

    This function takes a query string and searches for the best matching primary area
    from the given DataFrame. It calculates the similarity between the query and each
    primary area in the DataFrame, and returns the best match.

    Args:
        query (str): The query string to search for.
        df (pd.DataFrame): The DataFrame containing primary areas. Defaults to reading from PATH_PRIMARY_AREA.

    Returns:
        str: The best matching primary area.

    The function performs the following steps:
    1. Converts the query to lowercase for case-insensitive comparison.
    2. Iterates through each primary area in the DataFrame, converting it to lowercase.
    3. Calculates the similarity between the query and the current primary area.
    4. Updates the best match if the current primary area has a higher similarity score.
    5. Returns the best matching primary area.
    """
    query = query.lower()  # Convert query to lowercase

    best_match = None
    best_similarity = 0

    for area in df['primary_area']:
        area_lower = area.lower()  # Convert area to lowercase

        similarity = calculate_similarity(query, area_lower)  # Calculate similarity

        # Update the best match if the current area has higher similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = area

    return best_match

def calculate_similarity(query, area):
    """
    Calculate the similarity between the query and the area.

    This function calculates the similarity score between the query string and the
    area string based on various criteria such as exact match, missing characters,
    and common characters.

    Args:
        query (str): The query string.
        area (str): The area string to compare with the query.

    Returns:
        float: The similarity score between the query and the area.

    The function performs the following steps:
    1. Checks for an exact match and returns a high similarity score if found.
    2. Checks for missing characters and returns a high similarity score if found.
    3. Calculates the similarity based on the number of common characters between the query and the area.
    4. Returns the similarity score.
    """
    query_len = len(query)
    area_len = len(area)

    # Check for exact match
    if query == area:
        return 100

    # Check for missing characters
    if query_len + 1 == area_len and area.startswith(query):
        return 90

    # Check for misspelled names with 1 missing character
    if query_len == area_len + 1 and query.startswith(area):
        return 90

    # Calculate similarity based on common characters
    common_chars = set(query).intersection(area)
    similarity = (len(common_chars) / max(query_len, area_len)) * 100

    return similarity
 
def impute_municipal_children_count(year,area):
    """
    Impute the count of children in households within a municipality based on provided data.

    This function fetches municipal children data for the specified year and area,
    processes the data to impute the count of children in different household categories,
    and returns the imputed data as dictionaries for households with children aged 0-24 years
    and 25 years or older.

    Args:
        year (int): The year for which the data is to be fetched.
        area (str): The geographical area for which the data is to be fetched.

    Returns:
        tuple: Two dictionaries containing imputed children counts for households with children
               aged 0-24 years and 25 years or older.

    The function performs the following steps:
    1. Fetches municipal children data for the specified year and area.
    2. Processes the data to create a nested dictionary with children counts for different household categories.
    3. Transforms the data to calculate the probability of having a certain number of children in each household category.
    4. Separates the data into two dictionaries based on the age of children (0-24 years and 25 years or older).
    5. Returns the two dictionaries with imputed children counts.
    """
    data = fetch_municipal_children_data(year)

    # Extract the relevant data from the JSON
    nested_dict = {}
    total_households = []
    for hh in data["data"]:
        key = hh["key"][1]
        nested_key = hh["key"][2]
        value = int(hh["values"][0])
        #total_households.append(value)
        if key not in nested_dict:
            nested_dict[key] = {}
        nested_dict[key][nested_key] = value
    if 'SAKNAS' in nested_dict:
        del nested_dict['SAKNAS']

    # Calculate the total number of households in the municipality
    for key in nested_dict:
        total_households.append(sum(nested_dict[key].values()))

    # Get probability of number of children in each household category
    #for i,key in enumerate(nested_dict):
    #    for nested_key in nested_dict[key]:
    #        nested_dict[key][nested_key] = nested_dict[key][nested_key]/total_households[i]

    # Creating a dictionary for name changes
    name_change = {
        'ESUB': 'Ensamstående utan barn',
        'ESMB24': 'Ensamstående med barn 0-24 år',
        'ESMB25': 'Ensamstående med barn 25 år eller äldre',
        'SMUB': 'Sammanboende utan barn',
        'SBMB24': 'Sammanboende med barn 0-24 år',
        'SBMB25': 'Sammanboende med barn 25 år eller äldre',
        'OVRIUB': 'Övriga hushåll utan barn',
        'ÖMB24': 'Övriga hushåll med barn 0-24 år',
        'ÖMB25': 'Övriga hushåll med barn 25 år eller äldre'
    }

    # If a key is found in the name_change dictionary, replace the key with the value
    municipal_children = {}
    for old_key,value in nested_dict.items():
        new_key = name_change.get(old_key, old_key)
        municipal_children[new_key] = value

    # Remove keys with utan barn
    municipal_children = {k: v for k, v in municipal_children.items() if 'utan barn' not in k}

    # Replace UB with 0, M1B with 1, M2B with 2, M3+B with 3
    for key,value in municipal_children.items():
        key_list = list(value.keys())
        # Check if any ['UB','M1B', 'M2B','M3+B','SAKNAS'] in key_list
        if any(x in key_list for x in ['UB','M1B', 'M2B','M3+B','SAKNAS']):
            municipal_children[key][0] = municipal_children[key].pop('UB')
            municipal_children[key][1] = municipal_children[key].pop('M1B')
            municipal_children[key][2] = municipal_children[key].pop('M2B')
            municipal_children[key][3] = municipal_children[key].pop('M3+B')
            municipal_children[key]['other'] = municipal_children[key].pop('SAKNAS')


    # Create two dicts from municipal_children dict, one for 0-24 years and one for 25+ years
    municipal_children_0_24 = {k: v for k, v in municipal_children.items() if '0-24' in k}
    municipal_children_25 = {k: v for k, v in municipal_children.items() if '25' in k}

    # split key and replace with first word
    municipal_children_0_24 = {k.split()[0]: v for k, v in municipal_children_0_24.items()}
    municipal_children_25 = {k.split()[0]: v for k, v in municipal_children_25.items()}

    # Replace ensamstående with single, sammanboende with couple and övriga hushåll with other
    municipal_children_0_24 = {k.replace('Ensamstående', 'Single').replace('Sammanboende', 'Couple').replace('Övriga', 'Other'): v for k, v in municipal_children_0_24.items()}
    municipal_children_25 = {k.replace('Ensamstående', 'Single').replace('Sammanboende', 'Couple').replace('Övriga', 'Other'): v for k, v in municipal_children_25.items()}

    # Calculate total households 0-24 and 25+
    total_households_0_24 = []
    total_households_25 = []
    for key in municipal_children_0_24:
        total_households_0_24.append(sum(municipal_children_0_24[key].values()))
    for key in municipal_children_25:
        total_households_25.append(sum(municipal_children_25[key].values()))


    # Calculate the probability of number of children in each household category
    municipal_children_0_24
    for i,key in enumerate(municipal_children_0_24):
        for hh in municipal_children_0_24[key]:
            municipal_children_0_24[key][hh] = municipal_children_0_24[key][hh]/total_households_0_24[i]
    for i,key in enumerate(municipal_children_25):
        for hh in municipal_children_25[key]:
            municipal_children_25[key][hh] = municipal_children_25[key][hh]/total_households_25[i]

    return municipal_children_0_24, municipal_children_25

def get_probability_of_children(year, area):
    """
    Calculate the probability of having children in different household types.

    This function fetches data on households with children for the specified year and area,
    processes the data to calculate the probability of having children in different household types,
    and returns the calculated probabilities.

    Args:
        year (int): The year for which the data is to be fetched.
        area (str): The geographical area for which the data is to be fetched.

    Returns:
        dict: A dictionary containing the probability of having children in each household type.

    The function performs the following steps:
    1. Fetches data on households with children for the specified year and area.
    2. Processes the data to calculate the total number of households in each household type.
    3. Calculates the probability of having children in each household type based on the fetched data.
    4. Returns the calculated probabilities.
    """
    data = fetch_older_children_data(year, area)
    p_children_age = {}
    for item in data["data"]:
        key = (item["key"][1].split(" ")[0])
        nested_key = item["key"][1].replace(key,"")
        value = item["values"][0]
        if 'utan barn' in nested_key:
            nested_key = 'No Kids'
        elif '0-24' in nested_key:
            nested_key = 'Kids Under 25'
        elif '25' in nested_key:
            nested_key = 'Kids Over 25'
        if key not in p_children_age:
            p_children_age[key] = {}
        p_children_age[key][nested_key] = int(value)

    # Total households in each key
    total_households = {}
    for key in p_children_age:
        total_households[key] = sum(p_children_age[key].values())

    # Replace Ensamstående with Single, Sammanboende with Couple and Övrigt with Other
    total_households['Single'] = total_households.pop('Ensamstående')
    total_households['Couple'] = total_households.pop('Sammanboende')
    total_households['Other'] = total_households.pop('Övriga')

    # Create a new dictionary called hasChild with two keys True and False and the values are the number of households
    hasChild = {}
    for key in p_children_age:
        hasChild[key] = {}
        hasChild[key][True] = p_children_age[key]['Kids Under 25'] + p_children_age[key]['Kids Over 25']
        hasChild[key][False] = p_children_age[key]['No Kids']

    # Replace Ensamstående with Single, Sammanboende with Couple and Övrigt with Other
    hasChild['Single'] = hasChild.pop('Ensamstående')
    hasChild['Couple'] = hasChild.pop('Sammanboende')
    hasChild['Other'] = hasChild.pop('Övriga')

    # Calculate the probability of having children in each household type
    p_children = {}
    for key in hasChild:
        p_children[key] = {}
        p_children[key][True] = hasChild[key][True]/total_households[key]
        p_children[key][False] = hasChild[key][False]/total_households[key]

    return p_children

def get_probability_of_housetype(year,area):
    """
    Calculate the probability of different household types in a given area.

    This function fetches data on household types for the specified year and area,
    processes the data to calculate the probability of each household type,
    and returns the calculated probabilities.

    Args:
        year (int): The year for which the data is to be fetched.
        area (str): The geographical area for which the data is to be fetched.

    Returns:
        dict: A dictionary containing the probability of each household type in the area.

    The function performs the following steps:
    1. Fetches data on household types for the specified year and area.
    2. Processes the data to map household types to simplified categories.
    3. Calculates the total number of households in each category.
    4. Calculates the percentage of each household type based on the fetched data.
    5. Returns the calculated probabilities.
    """
    # Fetch the data
    data = fetch_housetype_data(year,area)

    # Create a dictionary with the probability of each household type
    p_housetype = {}
    for item in data["data"]:
        key = (item["key"][1])
        nested_key = (item["key"][2])
        value = item["values"][0]
        if "Småhus" in nested_key:
            nested_key = "Villa"
        elif "Flerbostadshus" in nested_key:
            nested_key = "Apartment"
        elif "Specialbostad, övriga hus" in nested_key:
            nested_key = "Other"
        else:
            nested_key = "Not Available"
        if key not in p_housetype:
            p_housetype[key] = {}
        p_housetype[key][nested_key] = int(value)

    # Total households in each key
    total_households = {}
    for key in p_housetype:
        total_households[key] = sum(p_housetype[key].values())

    # Calculate the percentage of each household type
    p_housetype_percentage = {}
    for key in p_housetype:
        p_housetype_percentage[key] = {}
        for nested_key in p_housetype[key]:
            # Take care of divbyzero error
            if total_households[key] == 0:
                p_housetype_percentage[key][nested_key] = 0
            else:
                p_housetype_percentage[key][nested_key] = p_housetype[key][nested_key]/total_households[key]

    return p_housetype_percentage

def parse_num_children(data):
    """
    Parse the number of children from a string.

    This function extracts the number of children from the given data string using regular expressions.

    Args:
        data (str): The input string containing the number of children.

    Returns:
        int: The number of children extracted from the input string. Returns 0 if no number is found.

    Example:
        >>> parse_num_children("3 children")
        3
    """
    return int(re.search(r'\d+', data).group()) if re.search(r'\d+', data) else 0

def get_younger_child_probability_matrix(data):
    """
    Calculate the probability matrix for the number of younger children in households.

    This function processes the input data to calculate the probability of having a certain number
    of younger children in different household categories. It returns the resulting probability matrix.

    Args:
        data (dict): The input data containing household information.

    Returns:
        dict: A dictionary containing the probability matrix for the number of younger children in households.

    The function performs the following steps:
    1. Renames household types in the input data to simplified categories.
    2. Merges data for similar household types.
    3. Creates a nested dictionary with the count of younger children for each household category.
    4. Calculates the probability of having a certain number of younger children in each household category.
    5. Returns the resulting probability matrix.
    """
    # Step 1: Rename household types
    # Create a dictionary to map the original household types to their replacements
    household_types = {
        'Ensamstående': 'Single',
        'Sammanboende': 'Couple',
        'Övriga hushåll': 'Couple',
        'Uppgift saknas': 'Other'
    }

    # Iterate through each entry in the data
    for entry in data['data']:
        household_type = entry['key'][1]  # Get the household type from the entry
        # Replace the household type with its corresponding replacement if it exists in the dictionary
        entry['key'][1] = household_types.get(household_type, household_type)

    # Step 2: Merge couple households
    # Initialize an empty dictionary to store the merged data
    merged_data = {}

    # Iterate through each entry in the data
    for entry in data['data']:
        key = tuple(entry['key'][1:-1])  # Extract the key by excluding the area and year
        value = int(entry['values'][0])  # Convert the value to an integer
        merged_data[key] = merged_data.get(key, 0) + value
        # Add the value to the existing value for the corresponding key, or initialize it to 0 if the key doesn't exist

    # Step 2.1 Nest data
    # Initialize an empty dictionary to store the nested data
    probability_matrix = {}

    # Iterate through each (category, children) pair and its corresponding value in the merged data
    for (category, children), value in merged_data.items():
        # Create a nested dictionary if the category doesn't exist in the probability matrix
        probability_matrix.setdefault(category, {})[children] = value
        # Add the value to the corresponding children key in the category dictionary

    # Step 3: Find totals per children number
    # Iterate through each household type in the probability matrix
    for household_type in probability_matrix:
        # Calculate the total value by summing the values for all children numbers in the household type dictionary
        total = sum(probability_matrix[household_type].values())

        # Replace the values in the household type dictionary with their respective probabilities
        probability_matrix[household_type] = {
            children: value / total if total != 0 else 0
            for children, value in probability_matrix[household_type].items()
            # Divide each value by the total, accounting for the possibility of zero division
        }

    # Return the resulting probability matrix
    return probability_matrix

def get_older_child_probability_matrix(older_children_data):
    """
    Calculate the probability matrix for households with older children (aged 25+ years).

    This function processes the input data to calculate the probability of households having older children.
    It returns a dictionary with the probability of having older children in different household categories.

    Args:
        older_children_data (dict): The input data containing information on households with older children.

    Returns:
        dict: A dictionary containing the probability matrix for households with older children.

    The function performs the following steps:
    1. Processes the input data to extract household categories and the count of older children.
    2. Calculates the total number of households with older children.
    3. Calculates the probability of having older children in each household category.
    4. Returns the resulting probability matrix.
    """
    total_older_kids = 0
    keys = []
    probabilities = []

    for household_type in older_children_data['data']:
        key = household_type["key"][1]
        if "25" in key:
            if "Ensamstående" in key:
                key = "Single"
            elif "Sammanboende" in key:
                key = "Couple"
            elif "Övriga" in key:
                key = "Other"
            keys.append(key)

            value = int(household_type["values"][0])
            total_older_kids += value
            probabilities.append(value)

    probabilities = [value / total_older_kids for value in probabilities]
    # create a dict of keys and probabilities
    return dict(zip(keys, probabilities))

def sample_household_for_older_child(probability_matrix):
    """
    Sample a household type based on the probability matrix for older children.

    This function uses the provided probability matrix to randomly sample a household type
    that is likely to have older children.

    Args:
        probability_matrix (dict): The probability matrix for households with older children.

    Returns:
        str: The sampled household type.

    The function performs the following steps:
    1. Extracts household categories and their corresponding probabilities from the probability matrix.
    2. Randomly samples a household type based on the extracted probabilities.
    3. Returns the sampled household type.
    """
    # Sample a household type based on the probabilities
    household_categories = list(probability_matrix.keys())
    probabilities = list(probability_matrix.values())
    sampled_household_type = random.choices(household_categories, probabilities)[0]
    return(sampled_household_type)

def sample_children_category(probability_matrix, household_type):
    """
    Sample a children category based on the probability matrix for the given household type.

    This function uses the provided probability matrix to randomly sample a children category
    for the specified household type.

    Args:
        probability_matrix (dict): The probability matrix for children categories.
        household_type (str): The household type for which to sample a children category.

    Returns:
        int: The sampled children category. Returns None if the household type is not found in the probability matrix.

    The function performs the following steps:
    1. Checks if the household type exists in the probability matrix.
    2. Extracts children categories and their corresponding probabilities from the probability matrix.
    3. Randomly samples a children category based on the extracted probabilities.
    4. Returns the sampled children category.
    """
    if household_type in probability_matrix:
        children_categories = list(probability_matrix[household_type].keys())
        probabilities = list(probability_matrix[household_type].values())

        # Sample a children category based on the probabilities
        sampled_category = random.choices(children_categories, probabilities)[0]

        return(sampled_category)
    else:
        return(None)

def balance_lists(p1, p2):
    """
    Balance the lengths of two lists by moving excess individuals to a separate list.

    This function ensures that the two input lists have the same length by moving excess individuals
    from the longer list to a separate list of unmatched individuals.

    Args:
        p1 (list): The first list of individuals.
        p2 (list): The second list of individuals.

    Returns:
        tuple: A tuple containing the balanced lists and the list of unmatched individuals.

    The function performs the following steps:
    1. Checks if the lengths of the two lists are equal.
    2. If not, calculates the difference in lengths and moves excess individuals to a separate list.
    3. Returns the balanced lists and the list of unmatched individuals.
    """
    len_p1, len_p2 = len(p1), len(p2)

    # Check if the lists are already of equal length
    if len_p1 == len_p2:
        return p1, p2, []

    # Find the difference in lengths
    diff = abs(len_p1 - len_p2)

    # Move individuals to unmatched_individuals until the lengths are equal
    unmatched_individuals = []
    while diff > 0:
        if len_p1 > len_p2:
            unmatched_individuals.append(p1.pop())
        else:
            unmatched_individuals.append(p2.pop())
        diff -= 1

    return p1, p2, unmatched_individuals

def couples_from_individuals(p1,p2):
    """
    Create couples from two lists of individuals and form households.

    This function matches individuals from two lists (e.g., males and females) to create couples
    and form households. It balances the lists, matches individuals based on age, and creates
    Household instances for each couple.

    Args:
        p1 (list): The first list of individuals (e.g., males).
        p2 (list): The second list of individuals (e.g., females).

    Returns:
        tuple: A tuple containing the number of households created and the list of Household instances.

    The function performs the following steps:
    1. Balances the lengths of the two lists.
    2. Matches individuals from the two lists based on age and other criteria.
    3. Creates Household instances for each matched couple.
    4. Returns the number of households created and the list of Household instances.
    """
    households = []

    #logger.info(f"There are {len(p1)} males and {len(p2)} females in this group")
    # Check the length of the two groups, if they are not equal, move the extra individual into a new group
    #unmatched_individuals = []
    #logger.info("Length of partner lists",len(p1), len(p2))
    
    # P1 and p2 need to be of equal length
    # if moving extra individual to either p1 or p2 makes them equal, then do that
    # else move the extra individual to unmatched_individuals until the length of p1 and p2 are equal
    p1, p2, unmatched_individuals = balance_lists(p1, p2)
    
    # Check if unmatched_individuals in empty
    # if not empty check if it is odd
    # if odd, create one new person with the same attributes as a random unmatched_individual
    # and add it to the unmatched_individuals list
    # Once even, split the list into two and add them to p1 and p2
    if unmatched_individuals:
        if len(unmatched_individuals) % 2 == 1:
            # pick a random person
            person = random.choice(unmatched_individuals)
            #get age group of person
            person_age_group = age_group_from_age(person.age)
            new_person = Person(person_age_group, person.sex, person.household_type)
            # Add the new person to the unmatched_individuals list
            unmatched_individuals.append(new_person)
        # Split the unmatched_individuals list into two and add them to p1 and p2
        p1.extend(unmatched_individuals[:len(unmatched_individuals)//2])
        p2.extend(unmatched_individuals[len(unmatched_individuals)//2:])
    
    #logger.info("Length of partner lists",len(p1), len(p2))
    # sort the two groups by age
    p1.sort(key=lambda x: x.age, reverse=True)
    p2.sort(key=lambda x: x.age, reverse=True)
    # Get the age proxy for group 1
    age_proxy_list = []
    for person in p1:
        # Sample a number from distribution mean = 0 sd = 6
        age_proxy = np.random.normal(0, 6) + person.age
        age_proxy_list.append(age_proxy)
    # sort age_proxy_list in descending order and use it to sort p1
    p1 = [x for _, x in sorted(zip(age_proxy_list, p1), reverse=True)]

    # Match individuals in married_individuals_split_1 with individuals in p2
    for i, person in enumerate(p1):
        # assiging household head
        if person.age > p2[i].age:
            person.is_head = True
        else:
            p2[i].is_head = True
        household = Household('Couple')
        household.add_member(person)
        household.add_member(p2[i])
        # adding household to households
        households.append(household)
    households_created = len(p1)
    return households_created, households

def preprocess_household_data(aligned_columns = [], drop = [], onehotencode = False):
    """
    Preprocess household data for model input.

    This function preprocesses household data by ensuring all required columns are present,
    dropping specified columns, and optionally one-hot encoding categorical variables.

    Args:
        aligned_columns (list): List of columns to ensure are present in the DataFrame.
        drop (list): List of columns to drop from the DataFrame.
        onehotencode (bool): Whether to one-hot encode categorical variables. Defaults to False.

    Returns:
        tuple: A tuple containing the preprocessed DataFrame and the list of adult individuals.

    The function performs the following steps:
    1. Fetches household data and adult individuals.
    2. Ensures all specified columns are present in the DataFrame.
    3. Drops specified columns from the DataFrame.
    4. Optionally one-hot encodes categorical variables.
    5. Returns the preprocessed DataFrame and the list of adult individuals.
    """
    
    df, adults = Household.return_nhts(drop=drop, onehotencode=onehotencode)
    
    # Ensure all columns in aligned_columns are present in the DataFrame
    for column in aligned_columns:
        if column not in df.columns:
            df[column] = 0

    df = df[aligned_columns]
    return df, adults

def cap_cars_per_household(households, max_cars=4):
    """
    Cap the number of cars per household to a specified maximum.

    This function ensures that no household has more than the specified maximum number of cars.
    If a household has more cars than the specified maximum, the excess cars are removed and
    the car ownership status of individuals is updated accordingly.

    Args:
        households (list): The list of Household instances.
        max_cars (int): The maximum number of cars allowed per household. Defaults to 4.

    Returns:
        None

    The function performs the following steps:
    1. Iterates through each household in the list.
    2. Checks if the household has more cars than the specified maximum.
    3. If so, removes the excess cars and updates the car ownership status of individuals.
    """
    for household in households:
        if household.cars > max_cars:
            excess_cars = household.cars - max_cars
            household.cars -= excess_cars
            members_with_cars = [member for member in household.members if member.has_car]
            random.shuffle(members_with_cars)
            for person in members_with_cars[:excess_cars]:
                person.has_car = False

def assign_cars_to_households(year, area, classifier):
    """
    Assign car ownership to households based on a classifier model.

    This function assigns car ownership to households in the specified year and area
    using a pre-trained classifier model. It predicts the probability of car ownership
    for each household and assigns cars based on the top predictions.

    Args:
        year (int): The year for which to assign car ownership.
        area (str): The geographical area for which to assign car ownership.
        classifier (object): The pre-trained classifier model for predicting car ownership.

    Returns:
        None

    The function performs the following steps:
    1. Fetches car ownership data for the specified year and area.
    2. Preprocesses household data for model input.
    3. Predicts the probability of car ownership for each household.
    4. Assigns car ownership based on the top predictions.
    5. Caps the number of cars per household to a specified maximum.
    6. Logs the total number of cars after capping.
    """
    # Provide the estimated total cars and call the function to scale and optionally plot the results
    car_data = fetch_car_data(year,area)
    estimated_total_cars = int(car_data["data"][0]["values"][0])

    logger.info(f"Total estimated cars in neighborhood: {estimated_total_cars}")
    aligned_columns = [
        'child_count', 'adult_count', 'x0_Kvinnor', 'x0_Män', 'x0_Other',
        'x1_16-24', 'x1_25-34', 'x1_35-44', 'x1_45-54', 'x1_55-64', 'x1_65-74',
        'x1_75+', 'x2_Apartment', 'x2_Other', 'x2_Villa', 'x3_Couple',
        'x3_Other', 'x3_Single', 'x2_Not Available'
    ]
    df, adults = preprocess_household_data(aligned_columns = aligned_columns, drop = ['car_count', 'primary_status'], onehotencode = True)

    # Predict probabilities using the classifier
    probs = classifier.predict_proba(df)[:, 1]

    # Assign car_count based on the top estimated_total_cars predictions
    top_indices = (-probs).argsort()[:estimated_total_cars]
    df['car_count'] = 0
    df.loc[top_indices, 'car_count'] = 1

    # Assign cars to individuals and calculate household cars
    for i, person in enumerate(adults):
        person.has_car = df['car_count'][i]

    for household in Household.instances:
        household.cars = sum(person.has_car for person in household.members)

    # Cap the number of cars per household to a maximum of 3
    cap_cars_per_household(Household.instances)

    total_cars_after_capping = sum(household.cars for household in Household.instances)
    logger.info(f"Total cars in neighborhood after capping to 4: {total_cars_after_capping}")

def match_child_to_parent(parent, list_of_children, min_age_of_parent=20, initial_tolerance=5, max_tolerance=20, tolerance_increment=2):
    """
    Match a child to a parent based on age and add the child to the parent's household.

    This function matches a child to a parent from the list of children based on the age difference
    between the parent and the child. It uses a tolerance range to find a suitable match and adds
    the matched child to the parent's household.

    Args:
        parent (Person): The parent individual.
        list_of_children (list): The list of child individuals to match.
        min_age_of_parent (int, optional): The minimum age for a parent. Defaults to 20.
        initial_tolerance (int, optional): The initial tolerance range for matching. Defaults to 5.
        max_tolerance (int, optional): The maximum tolerance range for matching. Defaults to 20.
        tolerance_increment (int, optional): The increment in tolerance range for each iteration. Defaults to 2.

    Returns:
        None

    The function performs the following steps:
    1. Calculates the proxy age of the child based on the parent's age and minimum age of parent.
    2. Iterates through the list of children to find a match within the initial tolerance range.
    3. If no match is found, increases the tolerance range and retries until the maximum tolerance is reached.
    4. Adds the matched child to the parent's household and removes the child from the list of children.
    5. Logs the matching process and results.
    """

    # Age difference heuristic based on the parent's age
    proxy_age_of_child = parent.age - min_age_of_parent

    #logger.info(f"match_child_to_parent: Parent Age: {parent.age}. Calculated Proxy Age of Child: {proxy_age_of_child}.")

    tolerance = initial_tolerance
    children_to_remove = []

    while tolerance <= max_tolerance and not children_to_remove:
        #logger.info(f"match_child_to_parent: Checking with tolerance: {tolerance} years")

        # Get potential matches based on current tolerance
        potential_matches = [child for child in list_of_children if abs(child.age - proxy_age_of_child) < tolerance]
        
        # If there are potential matches, sort them by age (older children first)
        potential_matches.sort(key=lambda x: x.age, reverse=True)

        if potential_matches:
            # Pick the oldest child from potential matches
            chosen_child = potential_matches[0]
            #logger.info(f"match_child_to_parent: Matched child {chosen_child.age} to parent {parent.age} with tolerance {tolerance} years.")
            parent.household.add_child(chosen_child)
            logger.info(f"Matched child (age: {chosen_child.age}) to household with parent (age: {parent.age})")
            children_to_remove.append(chosen_child)
        else:
            tolerance += tolerance_increment
            #logger.info(f"match_child_to_parent: No suitable children found for parent {parent.age} of {parent.household_type} with current tolerance. Increasing tolerance.")

    # Remove matched children from the list
    for child in children_to_remove:
        list_of_children.remove(child)

    if not children_to_remove:
        logger.info(f"match_child_to_parent: No suitable children found for parent {parent.age} of {parent.household_type} even with max tolerance of {max_tolerance} years.")

def assign_primary_status_to_members_backup(classifier):
    """
    Assign primary status (e.g., work, education, home) to household members using a backup method.

    This function assigns primary status to household members using a pre-trained classifier model.
    It predicts the primary status for each individual based on household data and assigns the
    corresponding status to each member.

    Args:
        classifier (object): The pre-trained classifier model for predicting primary status.

    Returns:
        None

    The function performs the following steps:
    1. Preprocesses household data for model input.
    2. Predicts the primary status for each individual using the classifier.
    3. Assigns the predicted primary status to each member.
    4. Logs the distribution of primary status among the household members.
    """
    aligned_columns = [
        'child_count', 'adult_count', 'car_count', 'x0_Kvinnor', 'x0_Män', 'x0_Other',
        'x1_16-24', 'x1_25-34', 'x1_35-44', 'x1_45-54', 'x1_55-64', 'x1_65-74',
        'x1_75+', 'x2_Apartment', 'x2_Other', 'x2_Villa', 'x3_Couple',
        'x3_Other', 'x3_Single', 'x2_Not Available'
    ]
    
    df, adults = preprocess_household_data(aligned_columns = aligned_columns, drop=['primary_status'], onehotencode=True)

    # Predict the primary status for each person using the classifier
    df['primary_status'] = classifier.predict(df)

    # Add the primary status to each person
    for i, adult in enumerate(adults):
        numeric_primary_status = df['primary_status'][i]
        if numeric_primary_status == 1:
            adult.primary_status = 'WORK'
        elif numeric_primary_status == 2:
            adult.primary_status = 'EDUCATION'
        elif numeric_primary_status == 3:
            adult.primary_status = 'HOME'
        else:
            adult.primary_status = 'NA'
    
    workers, students, neither = [], [], []

    for person in Person.instances:
        if person.primary_status == "WORK":
            workers.append(person)
        elif person.primary_status == "EDUCATION":
            students.append(person)
        elif person.primary_status == "HOME":
            neither.append(person)
    logger.info(f"Number of workers: {len(workers)} - {len(workers)/len(Person.instances)*100:.2f}%")
    logger.info(f"Number of students: {len(students)} - {len(students)/len(Person.instances)*100:.2f}%")
    logger.info(f"Number of neither: {len(neither)} - {len(neither)/len(Person.instances)*100:.2f}%")
    logger.info(f"Total number of persons: {len(Person.instances)}")

def assign_primary_status_to_members(year,area,classifier):
    """
    Assign primary status (e.g., work, education, home) to household members.

    This function assigns primary status to household members for the specified year and area
    using a pre-trained classifier model. It predicts the probability of different primary statuses
    for each individual and assigns the status based on the top predictions.

    Args:
        year (int): The year for which to assign primary status.
        area (str): The geographical area for which to assign primary status.
        classifier (object): The pre-trained classifier model for predicting primary status.

    Returns:
        None

    The function performs the following steps:
    1. Fetches primary status data for the specified year and area.
    2. Preprocesses household data for model input.
    3. Predicts the probability of different primary statuses for each individual.
    4. Assigns primary status to individuals based on the top predictions.
    5. Logs the distribution of primary status among the household members.
    """
    primary_dict = fetch_primary_status(year, area)
    # {'WORK': '6140', 'STUDY': '505', 'INACTIVE': '285'}
    working_count = int(primary_dict['WORK'])
    study_count = int(primary_dict['STUDY'])
    home_count = int(primary_dict['INACTIVE'])

    aligned_columns = [
        'child_count', 'adult_count', 'car_count', 'x0_Kvinnor', 'x0_Män', 'x0_Other',
        'x1_16-24', 'x1_25-34', 'x1_35-44', 'x1_45-54', 'x1_55-64', 'x1_65-74',
        'x1_75+', 'x2_Apartment', 'x2_Other', 'x2_Villa', 'x3_Couple',
        'x3_Other', 'x3_Single', 'x2_Not Available'
    ]

    df, adults = preprocess_household_data(aligned_columns = aligned_columns, drop=['primary_status'], onehotencode=True)

    # Assume 'classifier' and 'df' are already defined and ready
    probs = classifier.predict_proba(df)

    # Initialize all to 'NA' (or another default status)
    df['primary_status'] = 'NA'

    # Sort indices based on probabilities for 'WORK'
    work_indices = np.argsort(-probs[:, 0])[:working_count]
    df.loc[work_indices, 'primary_status'] = 'WORK'

    # Exclude already assigned 'WORK' from 'STUDY'
    remaining_for_study = df[df['primary_status'] == 'NA']
    study_indices = np.argsort(-probs[remaining_for_study.index, 1])[:study_count]
    df.loc[study_indices, 'primary_status'] = 'EDUCATION'

    # Assign 'HOME' to the remaining, if needed
    remaining_for_home = df[df['primary_status'] == 'NA']
    if len(remaining_for_home) > home_count:
        home_indices = np.argsort(-probs[remaining_for_home.index, 2])[:home_count]
        df.loc[home_indices, 'primary_status'] = 'HOME'
    else:
        df.loc[remaining_for_home.index, 'primary_status'] = 'HOME'

    # Update the primary status of adults based on the DataFrame
    for i, adult in enumerate(adults):
        adult.primary_status = df.loc[i, 'primary_status']


    workers, students, neither = [], [], []

    for person in Person.instances:
        if person.primary_status == "WORK":
            workers.append(person)
        elif person.primary_status == "EDUCATION":
            students.append(person)
        elif person.primary_status == "HOME":
            neither.append(person)
    logger.info(f"Number of workers: {len(workers)} - {len(workers)/len(Person.instances)*100:.2f}%")
    logger.info(f"Number of students: {len(students)} - {len(students)/len(Person.instances)*100:.2f}%")
    logger.info(f"Number of neither: {len(neither)} - {len(neither)/len(Person.instances)*100:.2f}%")
    logger.info(f"Total number of persons: {len(Person.instances)}")


def split_households_by_householdtype():
    """
    Split individuals into different lists based on their household type.

    This function splits individuals into different lists based on their household type,
    such as children, single parents, living alone, married, cohabiting, and others.

    Returns:
        tuple: A tuple containing lists of individuals for each household type.

    The function performs the following steps:
    1. Iterates through each individual and categorizes them based on their household type.
    2. Adds the individuals to the corresponding list for their household type.
    3. Logs the number of individuals in each household type.
    4. Returns the lists of individuals for each household type.
    """
    # Create a list of households
    
    children = []
    single_parents = []
    living_alone = []
    married_males = []
    married_females = []
    cohabiting_males = []
    cohabiting_females = []
    others = []

    # Separate individuals into different lists based on their category
    for person in Person.instances:
        if person.household_type == CHILD:                                  # Child
            person.is_child = True
            children.append(person)
        elif person.household_type == SINGLE_PARENT:                        # Single parent
            #person.hasChild = True
            single_parents.append(person)
        elif person.household_type == LIVING_ALONE:                         # Living alone
            living_alone.append(person)
        elif person.household_type == MARRIED:                              # Married
            if person.sex == MALE:
                married_males.append(person)
            else:
                married_females.append(person)
        elif person.household_type == COHABITING:                           # Cohabiting
            if person.sex == MALE:
                cohabiting_males.append(person)
            else:
                cohabiting_females.append(person)
        else:
            others.append(person)                                           # Not living alone/ other

    # Print the number of individuals in each category
    logger.info(f"Number of children: {len(children)}")
    logger.info(f"Number of single parents: {len(single_parents)}")
    logger.info(f"Number of individuals living alone: {len(living_alone)}")
    logger.info(f"Number of married males: {len(married_males)}")
    logger.info(f"Number of married females: {len(married_females)}")
    logger.info(f"Number of cohabiting males: {len(cohabiting_males)}")
    logger.info(f"Number of cohabiting females: {len(cohabiting_females)}")
    logger.info(f"Number of others: {len(others)}")

    return children, single_parents, living_alone, married_males, married_females, cohabiting_males, cohabiting_females, others

def assign_house_type_to_households_old(year, area):
    """
    Assign house types to households based on probability data (old method).

    This function assigns house types to households in the specified year and area
    using probability data fetched from external sources. It uses an older method
    that directly maps household sizes to house types.

    Args:
        year (int): The year for which to assign house types.
        area (str): The geographical area for which to assign house types.

    Returns:
        None

    The function performs the following steps:
    1. Fetches probability data for house types based on the specified year and area.
    2. Maps household sizes to the corresponding house type key.
    3. Assigns house types to households based on the probability data.
    4. Logs any cases where household size does not match the probability data.
    """
    p_housetype = get_probability_of_housetype(year, area)

    # Define a mapping from household size to the corresponding key in p_housetype
    size_to_key_mapping = {
        1: '1 person',
        2: '2 personer',
        3: '3 personer',
        4: '4 personer',
        5: '5 personer',
        6: '6 eller fler personer'
    }

    for household in Household.instances:
        household_size = len(household.members)

        # Use the mapping to get the corresponding key for the household size
        key = size_to_key_mapping.get(household_size, None)
        if household_size > 6:  # Explicitly handle the case where size > 6
            key = '6 eller fler personer'
        
        if key:
            house_type = np.random.choice(list(p_housetype[key].keys()), p=list(p_housetype[key].values()))
            household.house_type = house_type
        else:
            logger.info(f"assign_house_type_to_households: Household size {household_size} not found in p_housetype")

def assign_house_type_to_households(year, area):
    """
    Assign house types to households based on probability data (improved method).

    This function assigns house types to households in the specified year and area
    using probability data fetched from external sources. It uses an improved method
    that accounts for varying household sizes and assigns house types accordingly.

    Args:
        year (int): The year for which to assign house types.
        area (str): The geographical area for which to assign house types.

    Returns:
        None

    The function performs the following steps:
    1. Fetches probability data for house types based on the specified year and area.
    2. Maps household sizes to the corresponding house type key, with special handling for larger households.
    3. Iterates through each household to assign house types based on the probability data.
    4. Logs any cases where valid house type data is not available for certain household sizes.
    """
    p_housetype = get_probability_of_housetype(year, area)

    # Define a mapping from household size to the corresponding key in p_housetype
    size_to_key_mapping = {
        1: '1 person',
        2: '2 personer',
        3: '3 personer',
        4: '4 personer',
        5: '5 personer',
        6: '6 eller fler personer'
    }

    for household in Household.instances:
        household_size = len(household.members)
        
        # For households larger than 6, start checking from '6 eller fler personer'
        if household_size > 6:
            current_size = 6
        else:
            current_size = household_size

        # Flag to indicate if a house type has been assigned
        assigned = False

        while current_size >= 1:
            key = size_to_key_mapping[current_size]

            # Check if there's valid probability data for the current size
            probabilities = list(p_housetype.get(key, {}).values())
            if sum(probabilities) > 0:
                house_type = np.random.choice(list(p_housetype[key].keys()), p=probabilities)
                household.house_type = house_type
                assigned = True
                break  # Exit the loop once a house type is assigned
            
            current_size -= 1  # Decrement size to check the next smaller size

        if not assigned:  # If no valid house type was assigned
            logger.info(f"No valid house type data for household size {household_size} or smaller. No house type assigned.")


def match_list_household_children(list_household, list_children):
    """
    Match children to households based on the number of children needed.

    This function matches children to households based on the number of children each household
    needs. It expands the household list based on the number of children required and pairs
    each child with a household.

    Args:
        list_household (list): The list of households.
        list_children (list): The list of children to be matched.

    Returns:
        None

    The function performs the following steps:
    1. Expands the household list based on the number of children needed in each household.
    2. Sorts the household list based on the age of a randomly chosen parent.
    3. Sorts the children list by age.
    4. Pairs each child with a household and adds the child to the household.
    5. Logs the matching process and results.
    """
    new_list_household = []

    # Expand household list based on the number of children in each household
    for household in list_household:
        stack_number = household.children
        temp_household_stack = [household for _ in range(stack_number)]
        new_list_household.extend(temp_household_stack)

    # Sort the household list based on the age of a randomly chosen parent (from oldest to youngest)
    #new_list_household.sort(key=lambda household: random.choice([member.age for member in household.members if not member.is_child]))

    # Sort the children list by age (oldest first)
    list_children.sort(key=lambda child: child.age, reverse=True)

    # Pair each child with a household
    for child in list_children:
        household = new_list_household.pop()
        parents = [member for member in household.members if not member.is_child]
        parent = random.choice(parents)
        #logger.info(f"Matched child (age: {child.age}) to household with parent (age: {parent.age})")
        household.add_child(child)


def assign_children_to_households(year, area, children , age_split, min_age_of_parent = 25):
    """
    Assign children to households based on age and probability data.

    This function assigns children to households for the specified year and area
    using probability data and age-based categorization. It splits households into
    different categories based on the age of the head and assigns children accordingly.

    Args:
        year (int): The year for which to assign children.
        area (str): The geographical area for which to assign children.
        children (list): The list of children to be assigned.
        age_split (int): The age threshold to categorize households.
        min_age_of_parent (int, optional): The minimum age for a parent. Defaults to 25.

    Returns:
        None

    The function performs the following steps:
    1. Determines if a household has children based on probability data.
    2. Splits households into different categories based on the age of the head.
    3. Assigns the number of children to each household based on the probability matrix.
    4. Randomly adjusts the number of children in households to match the total number of children.
    5. Matches children to households based on the number of children needed.
    6. Updates the has_child attribute for each person in the household.
    7. Logs the assignment process and results.
    """
    # Step 1 - Determine if a household has children
    households = Household.instances
    households.sort(key=lambda household: random.choice([member.age for member in household.members if not member.is_child]))

    # Get probability of having children
    p_children = get_probability_of_children(year, area)

    # Given the household type, calculate the probability of having children in household.has_children
    for household in households:
        category = household.category
        # Single parent households are assumed to have children already
        if category == 'Couple':
            household.has_children = np.random.choice([True, False], p=[p_children['Couple'][True], p_children['Couple'][False]])
        elif category == 'Other':
            household.has_children = np.random.choice([True, False], p=[p_children['Other'][True], p_children['Other'][False]])
    
    # Step 2 - Determine the age at which a person had their first child 25 years ago

    # Filter households that have children and are either Couple or Other
    households_with_children = [household for household in households if household.has_children and household.category in ['Single', 'Couple', 'Other']]

    # Split households based on the age of the head
    # These will get older children above 45
    households_with_children_above_age_split = [household for household in households_with_children if household.members[0].age > age_split]

    # These will get younger children below 45
    households_with_children_below_age_split = [household for household in households_with_children if household.members[0].age <= age_split]

    # Get the probability matrix for the number of children in each household category
    p_matrix_0_24, p_matrix_25 = impute_municipal_children_count(year, area)
 
    # Step 3 - Assign number of children to households with children
    for household in households_with_children_below_age_split:
        # Get household category
        category = household.category

        # Get the probability matrix for the household category
        if category == 'Single':
            p_matrix = p_matrix_0_24['Single']
        elif category == 'Couple':
            p_matrix = p_matrix_0_24['Couple']
        elif category == 'Other':
            p_matrix = p_matrix_0_24['Other']
        
        # Sample from p_matrix
        household.children = np.random.choice([1,2,3], p=[ p_matrix[1], p_matrix[2], p_matrix[3]])

    # Repeat for households with children and age of head above 50
    for household in households_with_children_above_age_split:
        # Get household category
        category = household.category
        # Get the probability matrix for the household category
        if category == 'Single':
            p_matrix = p_matrix_25['Single']
        if category == 'Couple':
            p_matrix = p_matrix_25['Couple']
        elif category == 'Other':
            p_matrix = p_matrix_25['Other']
        
        # Sample from p_matrix
        household.children = np.random.choice([1,2,3], p=[p_matrix[1], p_matrix[2], p_matrix[3]])
    
    # log number of children in households_with_children
    total_children_in_households = sum(household.children for household in households_with_children)
    
    

    # Randomly increase household.children to households with children < 3 until len(children) == total_children_in_households
    while len(children) > total_children_in_households:
        # Optionally shuffle the households to randomize which ones get extra children
        random.shuffle(households_with_children)
        
        increased = False  # Flag to check if any household's child count was increased in this iteration
        
        for household in households_with_children:
            if household.children < 3:
                household.children += 1
                total_children_in_households += 1
                increased = True
                if total_children_in_households == len(children):
                    break

        # If no household's child count was increased in this iteration, break out of the loop to prevent infinite looping
        if not increased:
            break
    logger.info(f"Total number of children to be assigned: {total_children_in_households}")
    logger.info(f"Total number of children: {len(children)}")
    
    match_list_household_children(households_with_children, children)

    # Update has_child attribute for each person in the household based on the number of children in the household
    for household in households:
        household.update_has_child()

def clear_instances():
    """
    Clear all instances of Population, Person, House, Household, Building, and ActivitySequence.

    This function clears all instances of the specified classes to reset the data.
    It ensures that no residual data remains from previous runs.

    Returns:
        None

    The function performs the following steps:
    1. Clears all instances of the Population class.
    2. Clears all instances of the Person class.
    3. Clears all instances of the House class.
    4. Clears all instances of the Household class.
    5. Clears all instances of the Building class.
    6. Clears all instances of the ActivitySequence class.
    """
    Population.clear_instances()
    Person.clear_instances()
    House.clear_instances()
    Household.clear_instances()
    Building.clear_instances()
    ActivitySequence.clear_instances()



    