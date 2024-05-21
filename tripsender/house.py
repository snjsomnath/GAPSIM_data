# -----------------------------------------------------
# house.py
# -----------------------------------------------------
# Description: Contains the House class used to represent a house in the simulation.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.house"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import uuid
import logging
import pandas as pd
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

class House:
    """
    Represents a housing unit occupied by residents in Gothenburg.

    Each household is assigned a house object that represents the housing unit that the residents occupy.
    A house is associated with a physical building in Gothenburg. For single-family houses, a single-house
    object is related to a building; for multi-family houses, multiple house objects are associated with a building.

    The house object contains:
        - Floor area
        - Reference to a building
        - Reference to a household
        - Unique identifier

    Attributes:
        uuid (UUID): Unique identifier for the house.
        household (Household): The household occupying the house.
        building (Building): The building the house is associated with.
        building_uuid (UUID): Unique identifier of the building.
        area (float): Floor area of the house.
    """
    instances = []
    
    
    def __init__(self,household, building):
        """Initialize the House with given attributes."""
        self.uuid = uuid.uuid4()
        self.household = household
        self.building = building
        self.building_uuid = building.uuid
        # https://www.scb.se/en/finding-statistics/statistics-by-subject-area/household-finances/income-and-income-distribution/households-housing/pong/statistical-news/households-housing-2019/
        self.area = 36*len(household.members)
        self.building.houses.append(self)
        self.building.population_total += len(self.household.members)
        self.building.isEmpty = False
        self.instances.append(self)
                
        # Add the origin on the building to the individuals in the household
        for member in self.household.members:
            member.origin = self.building.coord
        
        # For the household used to create the house, set the house attribute to the house
        household.house = self


    def __repr__(self):
        return f"A house with {len(self.household.members)} people."

    @classmethod
    def clear_instances(cls):
        """ Clear the instances list. """
        cls.instances = []
    
    @classmethod
    def return_dataframe(cls):
        """ Return a dataframe with all the instances. """
        
        data = []
        for instance in cls.instances:
            data.append(instance.info())
        return pd.DataFrame(data)
    
    def info(self):
        """Returns a dictionary with information about the house."""
        return {
            "House UUID": self.uuid,
            "Household UUID": self.household.uuid,
            "Building UUID": self.building.uuid,
            "Members in house" : len(self.household.members),
            "Adults in house" : len([member for member in self.household.members if member.is_child == False]),
            "Children in house" : len([member for member in self.household.members if member.is_child == True]),
            "Cars in the household" : self.household.cars,
            "Area": self.area
        }