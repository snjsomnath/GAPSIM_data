# -----------------------------------------------------
# person.py
# -----------------------------------------------------
# Description: Contains the Person class used to represent a person in the simulation.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.person"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------


# Importing libraries
import uuid
import re
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional, List, Union
if TYPE_CHECKING:
    from tripsender.household import Household
from tripsender.activity import ActivitySequence
from shapely.geometry import Point
import logging
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

class Person:
    """
    Represents an individual with attributes like age, gender, household status, etc.

    Attributes:
    -----------
    uuid : uuid.UUID
        Unique identifier for the person.
    household_uuid : Optional[uuid.UUID]
        Identifier for the person's household.
    parent_uuid : List[uuid.UUID]
        Identifiers for the person's parents.
    age : int
        Age of the person.
    sex : str
        Gender of the person.
    household_type : str
        Type of household the person belongs to.
    household : Optional[Household]
        Household instance the person is part of.
    has_car : bool
        Whether the person owns a car.
    child_count : int
        Number of children the person has.
    is_head : bool
        If the person is the head of the household.
    is_child : bool
        If the person is a child in the household.
    origin : Optional[Point]
        Origin of the person.
    activity_sequence : Optional[ActivitySequence]
        Activity sequence associated with the person.
    instances : list[Person]
        Class attribute to track all person instances.
    """

    instances: List['Person'] = []
    def __init__(self, age: str, sex: str, household_type: str):
        """Initialize the Person with given attributes."""
        self.uuid: uuid.UUID = uuid.uuid4()
        self.household_uuid: Optional[uuid.UUID] = None 
        self.parent_uuid: List[uuid.UUID] = []
        self.age: int = self.sample_from_bounds(age)
        self.sex: str = sex
        self.household_type: str = household_type  # TODO: Rename to household_status?
        self.household: Optional[Household] = None
        self.has_car: bool = False  # TODO: Check car logic
        self.child_count: int = 0  # TODO: Verify child count logic
        self.is_head: bool = False
        self.is_child: bool = False
        self.origin: Optional[Point] = None
        self.activity_sequence: Optional[ActivitySequence] = None
        self.primary_status: Optional[str] = None
        self.instances.append(self)
        self.age_group = age_group_from_age(self.age)
        self.work_location: Optional[Point] = None
        self.house_type = None
        self.has_child = False
        self.location_mapping = None

    def __repr__(self):
        """Representation of the Person instance."""
        return f"A {self.age} year old {self.sex} with {self.household_type} household status."
    
    def __str__(self):
        """String representation of the Person instance."""
        sex = "Male" if self.sex =="Män" else "Female"
        return f"A {self.age} year old {sex}."

    def info(self):
        """Retrieve a dictionary containing information about the person."""
        info_dict = {
            "UUID": self.uuid,
            "Household UUID": self.household_uuid,
            "Parent UUID": self.parent_uuid,
            "Age": self.age,
            "Sex": self.sex,
            "Household Type": self.household_type,
            "Household": self.household.uuid if self.household else None,
            "Is Head of household": self.is_head,
            "Has Car": self.has_car,
            "Has Child": self.has_child,
            "Child Count": self.child_count,
            "Origin": self.origin
        }
        return info_dict


    @classmethod
    def clear_instances(cls):
        """Class method to clear all instances stored in the class."""
        cls.instances = []

    @classmethod
    def get_adults(cls):
        """Class method to get all persons above 17 years of age."""
        return [person for person in cls.instances if person.age >= 17]    

    @classmethod
    def return_adults_df(cls):
        """Class method to return a dataframe of all Person instances."""
        adult_df = pd.DataFrame([vars(person) for person in cls.get_adults()])
        # Create a column called Person that contains the Person instance
        adult_df['Person'] = cls.get_adults()
        return adult_df

    @classmethod
    def get_female(cls):
        """Class method to get all female instances of the class."""
        return [person for person in cls.instances if person.sex == "Kvinnor"]
    
    @classmethod
    def get_male(cls):
        """Class method to get all male instances of the class."""
        return [person for person in cls.instances if person.sex == "Män"]

    @classmethod
    def return_dataframe(cls):
        """Class method to return a dataframe of all Person instances."""
        return pd.DataFrame([vars(person) for person in cls.instances])

    @staticmethod
    def sample_from_bounds(bounds: str) -> int:
        """
        Static method to sample an age from a given range.

        Parameters:
        -----------
        bounds : str
            A string representing the age range (e.g., '20-30').

        Returns:
        --------
        int
            A random age sampled from the given range.
        """
        numbers = re.findall(r'\d+', bounds)
        if not numbers:
            raise ValueError(f"Invalid age bounds: {bounds}")
            
        lower_bound = int(numbers[0])
        upper_bound = int(numbers[1]) if len(numbers) > 1 else 100

        return np.random.randint(lower_bound, upper_bound + 1)

def age_group_from_age(age_number):
    if age_number >= 0 and age_number <= 5:
        return '0-5 år'
    elif age_number >= 6 and age_number <= 15:
        return '6-15 år'
    elif age_number >= 16 and age_number <= 18:
        return '16-18 år'
    elif age_number >= 19 and age_number <= 24:
        return '19-24 år'
    elif age_number >= 25 and age_number <= 34:
        return '25-34 år'
    elif age_number >= 35 and age_number <= 44:
        return '35-44 år'
    elif age_number >= 45 and age_number <= 54:
        return '45-54 år'
    elif age_number >= 55 and age_number <= 64:
        return '55-64 år'
    elif age_number >= 65 and age_number <= 74:
        return '65-74 år'
    elif age_number >= 75 and age_number <= 84:
        return '75-84 år'
    else:
        return '85- år'