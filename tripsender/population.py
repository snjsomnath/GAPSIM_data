
# -----------------------------------------------------
# population.py
# -----------------------------------------------------
# Description: Contains the Population class used to represent the population in the simulation.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.population"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import uuid
from typing import List, Tuple, Dict, Any, Union
import numpy as np
from tripsender import fetcher
import logging
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)


class Population:
    """
    A class that represents population data for a specific year and area.

    This class facilitates the representation and manipulation of population data.
    It offers methods to initialize population arrays and provides easy access to
    various population metrics such as age, household type, sex, etc.

    Attributes:
    - year (int): The year for which the population data is represented.
    - area (str): The geographical area for which the population data is represented.
    - variables (List[str]): List of variables associated with the population data.
    - variable_categories (List[str]): List of categories for each variable.
    - comment (str): Additional comments or notes associated with the population data.
    - array_age_household_sex (np.ndarray): 3D array representing population data by age, household, and sex.
    - array_age_sex (np.ndarray): 2D array representing population data by age and sex.
    - array_age_household (np.ndarray): 2D array representing population data by age and household.
    - Other arrays...

    Class Attributes:
    - instances (List[Population]): List to keep track of all instances of the Population class.

    Example:
        >>> population_2022_haga = Population(2023, "Haga")
        >>> print(population_2022_haga.year)
        2022
        >>> print(population_2022_haga.area)
        "Haga"
    """
    instances: List['Population'] = []

    def __init__(self, year: int, area: str) -> None:
        """
        Initializes the Population object by fetching the relevant data.

        Parameters:
        - year (int): The year for which the population data is required.
        - area (str): The geographical area for which the population data is required.

        Returns:
        None
        """
        logger.info(f"Initializing Population object for year {year} and area {area}")
        result: Dict[str, Any] = fetcher.fetch_population_data(year, area)
        data, year, area, variables, variable_categories, comment = self._get_population_params(result)
        self.year: int = year
        self.area: str = area
        self.variables: List[str] = variables
        self.variable_categories: List[str] = variable_categories
        self.comment: str = comment

        # Initialize arrays
        self.array_age_household_sex: np.ndarray = np.array([])
        self.array_age_sex: np.ndarray = np.array([])
        self.array_age_household: np.ndarray = np.array([])
        self.array_sex_household: np.ndarray = np.array([])
        self.array_age_household_male: np.ndarray = np.array([])
        self.array_age_household_female: np.ndarray = np.array([])
        self.total_population: int = 0
        self.instances.append(self)
        self._instantiate(data)

    @classmethod
    def clear_instances(cls):
        cls.instances = []

    @classmethod
    def info(cls):
        info = {
            "description": "A population class with information about the population in a given area and year",
            "year": cls.instances[0].year,
            "area": cls.instances[0].area,
            "variables": cls.instances[0].variables,
            "variable_categories": cls.instances[0].variable_categories,
            "comment": cls.instances[0].comment,
            "total_population": cls.instances[0].total_population
         }
        return info

    def _create_2d_array(self, data, dimension1, dimension2, dimension3=None):

        categories_dim1 = np.unique([item['key'][dimension1] for item in data])
        categories_dim2 = np.unique([item['key'][dimension2] for item in data])
        
        
        
        if dimension3!=None:
            categories_dim3 = np.unique([item['key'][dimension3] for item in data]) if dimension3 else [None]
            array_2d = np.zeros((len(categories_dim1), len(categories_dim2), len(categories_dim3)), dtype=int)
            for item in data:
                if not dimension3 or item['key'][dimension3] in categories_dim3:
                    index_dim1 = np.where(categories_dim1 == item['key'][dimension1])[0][0]
                    index_dim2 = np.where(categories_dim2 == item['key'][dimension2])[0][0]
                    index_dim3 = np.where(categories_dim3 == item['key'][dimension3])[0][0] if dimension3 else 0
                    array_2d[index_dim1, index_dim2, index_dim3] += int(item['values'][0])
        else:
            categories_dim3 = [None]
            array_2d = np.zeros((len(categories_dim1), len(categories_dim2)), dtype=int)
            for item in data:
                index_dim1 = np.where(categories_dim1 == item['key'][dimension1])[0][0]
                index_dim2 = np.where(categories_dim2 == item['key'][dimension2])[0][0]
                array_2d[index_dim1, index_dim2] += int(item['values'][0])
    
        return array_2d

    def _instantiate(self, data: Dict[str, Any]) -> None:
        self.array_age_sex = self._create_2d_array(data, dimension1=1, dimension2=2)
        self.array_age_household = self._create_2d_array(data, dimension1=1, dimension2=3)
        self.array_sex_household = self._create_2d_array(data, dimension1=2, dimension2=3)
        self.array_age_household_sex = self._create_2d_array(data, dimension1=1, dimension2=3, dimension3=2)
        self.array_age_household_female = self.array_age_household_sex[:,:,0]
        self.array_age_household_male = self.array_age_household_sex[:,:,1]

        # Calculate the total population by summing all the values in each array
        total_alder_kon = np.sum(self.array_age_sex)
        total_alder_hushall = np.sum(self.array_age_household)
        total_kon_hushall = np.sum(self.array_sex_household)

        # Checking if the totals are equal
        if total_alder_kon == total_alder_hushall and total_alder_kon == total_kon_hushall:
            self.total_population = total_alder_kon
        else:
            self.total_population = None
            logger.info("Total population could not be calculated")

    def _get_population_params(self, result: Dict[str, Any]) -> Tuple:
        """
        Extracts population parameters from the given result.

        Parameters:
        - result (Dict[str, Any]): The fetched result containing population data.

        Returns:
        Tuple: Extracted data, year, area, variables, variable_categories, and comment.
        """
        json_data = result
        # Extract the relevant data from the JSON
        data = json_data["data"]

        logger.info(data)

        # Extract additional attributes
        year = data[0]['key'][4]
        area = data[0]['key'][0]
        variables = [column['code'] for column in json_data['columns'] if column['type'] == 'd']
        variable_categories = {
            column['code']: np.unique([item['key'][i] for item in data])
            for i, column in enumerate(json_data['columns']) if column['type'] == 'd'
        }
        comment = json_data['columns'][3]['comment']
        return data, year, area, variables, variable_categories, comment