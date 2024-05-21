# -----------------------------------------------------
# synthpop.py
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
__name__ = "tripsender.synthpop"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import logging
import random
import joblib

import tripsender.utils as tsutils

from tripsender.household import Household
from tripsender.person import Person

from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

def generate_persons(population, num_persons=None):
    """
    Generate individual person objects based on the given population data.

    This function generates a number of Person instances based on the demographic distribution
    provided in the population data. It uses the age, household type, and sex categories to
    proportionally create individuals that reflect the overall population structure.

    Args:
        population (Population): An instance of the Population class that contains demographic data.
        num_persons (int, optional): The number of persons to generate. If not specified, the function
                                      uses the total population size provided in the population data.

    Returns:
        None

    The function performs the following steps:
    1. If num_persons is specified, it uses that number; otherwise, it defaults to the total population size.
    2. It clears any existing Person instances to start fresh.
    3. It calculates the proportional distribution of individuals across different categories of age,
       household type, and sex based on the provided population data.
    4. It iterates through the categories, creating the appropriate number of Person instances for each
       category, and assigns them to the respective age, household type, and sex.

    Example usage:
        >>> population_data = Population(year=2023, area="Haga")
        >>> generate_persons(population_data)
    """
    if num_persons:
        # If num_persons is specified, use that number
        logger.info(f"Number of persons: {num_persons}") #TODO Not implemented
        pass
    else:
        # Otherwise, use the total population
        num_persons = population.total_population
        logger.info(f"Number of persons: {num_persons}")
    pop_distribution = population.array_age_household_sex/population.total_population
    household_labels = population.variable_categories['Hushållsställning']
    age_labels = population.variable_categories['Ålder']
    sex_labels = population.variable_categories['Kön']

    # Clear the instances of Person
    Person.clear_instances()

    # Generating persons based on the population distribution
    for i, age in enumerate(age_labels):
        for j, household in enumerate(household_labels):
            for k, sex in enumerate(sex_labels):
                person_range = int(pop_distribution[i,j,k]*num_persons)
                for _ in range(person_range):
                    Person(age, sex, household)

def synthesise_population(population, age_split = 45, min_age_of_parent = 25):
    """
    Synthesise a population by creating households and assigning attributes to individuals and households.

    This function creates and assigns households, assigns children to households, and sets various
    attributes such as house type and car ownership for the generated population. The process involves
    multiple steps to ensure that the synthesized population accurately reflects real-world demographic
    patterns and household structures.

    Args:
        population (Population): An instance of the Population class containing demographic data.
        age_split (int, optional): The age threshold used to categorize children into different age groups.
                                   Defaults to 45.
        min_age_of_parent (int, optional): The minimum age for an individual to be considered a parent.
                                           Defaults to 25.

    Returns:
        None

    The function performs the following steps:
    1. Retrieves the total number of households for the specified year and area.
    2. Clears any existing Household instances to start fresh.
    3. Splits individuals into different lists based on their household type categories (e.g., children,
       single parents, living alone, married, cohabiting, others).
    4. Sorts the lists to prioritize older individuals for certain categories.
    5. Creates households for individuals living alone and single parents.
    6. Creates households for married and cohabiting couples.
    7. Assigns remaining individuals to 'Other' category households.
    8. Assigns children to appropriate households based on their age and household type.
    9. Assigns house types to households based on the specified year and area.
    10. Assigns car ownership to households using a pre-trained classifier model.
    11. Assigns primary status (e.g., studying, working, inactive) to individuals using a pre-trained classifier model.

    Example usage:
        >>> population_data = Population(year=2023, area="Haga")
        >>> synthesise_population(population_data)
    """
    # Get total households
    year = population.year
    area = population.area
    total_households = tsutils.fetch_total_households(year, area)
    logger.info(f"Total households: {total_households}")

    # Split individuals into different lists based on their category
    children, single_parents, living_alone, married_males, married_females, cohabiting_males, cohabiting_females, others = tsutils.split_households_by_householdtype()
    
    # # Print counts and totals
    # logger.info(f"Children: {len(children)}")
    # logger.info(f"Single Parents: {len(single_parents)}")
    # logger.info(f"Living Alone: {len(living_alone)}")
    # logger.info(f"Married Male: {len(married_males)}")
    # logger.info(f"Married Female: {len(married_females)}")
    # logger.info(f"Cohabiting Male: {len(cohabiting_males)}")
    # logger.info(f"Cohabiting Female: {len(cohabiting_females)}")
    # logger.info(f"Others: {len(others)}")

    # logger.info(f"Total: {len(children)+len(single_parents)+len(living_alone)+len(married_males)+len(married_females)+len(cohabiting_males)+len(cohabiting_females)+len(others)}")

    # Sort children by age
    # Trying reverse to prioritize older parents first
    children.sort(key=lambda x: x.age, reverse = True)
    # Sort single parents by age
    single_parents.sort(key=lambda x: x.age, reverse = True)

    Household.clear_instances()
    households = []

    # Step 1 - Create Households
    # Living alone -> Single
    logger.info(f"Total households after living alone: ie {total_households} - {len(living_alone)} = {total_households-len(living_alone)}")
    for person in living_alone:
        household = Household('Single')
        household.add_member(person)
        households.append(household)
    total_households -= len(living_alone)
    
    
    # Single parents -> Single
    logger.info(f"Total households after single parents: ie {total_households} - {len(single_parents)} = {total_households-len(single_parents)}")
    for person in single_parents:
        household = Household('Single')
        household.add_member(person)
        household.has_children = True
        households.append(household)
    total_households -= len(single_parents)
  

    # Married Couple / Cohabiting -> Couple
    
    len_couple_households,couple_households = tsutils.couples_from_individuals(married_males+cohabiting_males,married_females+cohabiting_females)
    logger.info(f"Len of couple households: {len_couple_households}")
    logger.info(f"Actual len of couple households: {len(couple_households)}")
    households.extend(couple_households)
    logger.info(f"Total households after couples: ie {total_households} - {len_couple_households} = {total_households-len_couple_households}")
    
    total_households -= len_couple_households
    

    # Other -> Other
    # Currently being undercounted
    random.shuffle(others)
    for person in others:
        person = others.pop()
        household = Household('Other')
        household.add_member(person)
        households.append(household)
        total_households -= 1
    logger.info(f"Total households after others: ie {total_households} - {len(others)} = {total_households-len(others)}")
    
    other_households = [household for household in households if household.category == 'Other']
    logger.info(f"Total other households: {len(other_households)}")
    while others and other_households:
        person = others.pop()
        household = random.choice(other_households)
        household.add_member(person)

    # Step 2 - Assigning Children
    tsutils.assign_children_to_households(year, area, children,age_split)
    Household.sync_children_in_households()
    # Step 3 - Assigning a housetype to the households
    tsutils.assign_house_type_to_households(year,area)

    # Step 4 - Assigning car ownership to households
    car_classifier = joblib.load('models/NHTS_CAR_OWNERSHIP_CLASSIFIER.joblib')
    tsutils.assign_cars_to_households(year, area, car_classifier)

    # Step 5 - Assigning primary status to persons
    primary_status_classifier = joblib.load('models/NHTS_PRIMARY_STATUS_CLASSIFIER.joblib')
    tsutils.assign_primary_status_to_members(year,area,primary_status_classifier)