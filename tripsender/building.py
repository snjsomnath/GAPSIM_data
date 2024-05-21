# -----------------------------------------------------
# building.py
# -----------------------------------------------------
# Description: Contains the Building class used to represent a building in the simulation.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.building"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------
# Importing libraries
import uuid
import geopandas as gpd
import logging
from typing import TYPE_CHECKING, Optional, List, Union
import random
from tripsender.activity import Location
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPoint
# -----------------------------------------------------
#Setup logger
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)


class Building:
    """
    Represents a building in Gothenburg, including its physical attributes and population data.

    A building object refers to an existing building and includes:
        - Footprint area
        - Total built-up area
        - Coordinates (in EPSG:3006 coordinate reference system)
        - Height of the building (calculated from laser point-cloud data)
        - Population per floor
        - Total feasible population for the building
        - Unique identifier
        - List of house objects contained within it
        - List of all people living in the building

    Attributes:
        uuid (UUID): Unique identifier for the building.
        type (str): Type of the building (e.g., residential, commercial).
        area (float): Total area of the building.
        height (float): Height of the building.
        floors (int): Number of floors in the building.
        footprint (Polygon): Footprint area of the building.
        population_per_floor (int): Population per floor in the building.
        population_total (int): Total population in the building.
        built_up_area (float): Total built-up area of the building.
        houses (List): List of house objects contained within the building.
        workers (int): Number of workers in the building.
        worker_list (List): List of workers in the building.
        coord (Point): Coordinates of the building's centroid.
        preferred_locations (Optional[PreferredLocations]): Preferred locations for the building.
    """

    instances = []
    def __init__(self, building_type, building_area, building_height, building_floors, footprint,population_per_floor,built_up_area):
        if footprint == None:
            raise ValueError("Building footprint is None.")
        self.uuid = uuid.uuid4()
        self.type = building_type
        self.area = building_area
        self.height = building_height
        self.floors = building_floors
        self.footprint = footprint
        self.population_per_floor = population_per_floor
        self.population_total = 0
        self.built_up_area = built_up_area
        self.houses = []
        self.instances.append(self)
        self.workers: int = 0
        self.worker_list: List = []
        #self.isEmpty = True
        self.coord = footprint.centroid
        # Initialize the preferred locations for this building
        self.preferred_locations : Optional(PreferredLocations) = None
    
    @property
    def is_empty(self):
        return len(self.houses) == 0

    @classmethod
    def clear_instances(cls):
        cls.instances = []

    def __repr__(self):
        return f"A {self.type} building with {self.floors} floors and {self.population_total} people."

    def info(self):
        """Returns a dictionary with information about the building."""
        return {
            "Building UUID": self.uuid,
            "Building Type": self.type,
            "Building Area": self.area,
            "Building Height": self.height,
            "Building Floors": self.floors,
            "Building Footprint": self.footprint,
            "Population per Floor": self.population_per_floor,
            "Population Total": self.population_total,
            "Houses in Building": [house.info() for house in self.houses],
            "Built up Area": self.built_up_area,
            "Is building empty" : self.isEmpty,
            "Number of workers": self.workers,
        }
    
    @classmethod
    def instantiate_buildings(cls, gdf_residential: gpd.GeoDataFrame):
        """Instantiate building objects based on input data."""
        cls.clear_instances()
        for _, row in gdf_residential.iterrows():
            Building(
                row['byggnadsundergrupp'],
                row['area'],
                row['height'],
                row['floors'],
                row['geom'],
                row['population_per_floor'],
                row['BTA']
            )
        if len(cls.instances) == 0:
            raise ValueError("Unable to instantiate buildings.")
    
    def add_houses(self, house):

        self.houses.append(house)
        house.building = self
        self.population_total += len(house.household.members)
        #self.isEmpty = False
        house.building_uuid = self.uuid
    
    @classmethod
    def return_gdf(cls):
        """Returns a GeoDataFrame with all buildings."""
        gdf = gpd.GeoDataFrame()
        gdf['uuid'] = [building.uuid for building in cls.instances]
        gdf['type'] = [building.type for building in cls.instances]
        gdf['area'] = [building.area for building in cls.instances]
        gdf['height'] = [building.height for building in cls.instances]
        gdf['floors'] = [building.floors for building in cls.instances]
        gdf['footprint'] = [building.footprint for building in cls.instances]
        gdf['population_per_floor'] = [building.population_per_floor for building in cls.instances]
        gdf['population_total'] = [building.population_total for building in cls.instances]
        gdf['built_up_area'] = [building.built_up_area for building in cls.instances]
        gdf['workers'] = [building.workers for building in cls.instances]
        gdf['is_empty'] = [building.is_empty for building in cls.instances]
        gdf['building'] = [building for building in cls.instances]
        gdf['coord'] = [building.coord for building in cls.instances]
        gdf['preferred_locations'] = [building.preferred_locations for building in cls.instances]
        
        # Set geometry to footprint
        gdf = gdf.set_geometry('footprint')
        # Set crs to EPSG:3006
        gdf.crs = "EPSG:3006"
        # If there are no buildings raise an error
        if len(gdf) == 0:
            raise ValueError("There are no buildings in the simulation.")

        return gdf
    import contextily as ctx
    def plot(self):
        
        """
        Plots the building footprint on a map with a basemap using contextily.
        """
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': [self.footprint]}, crs="EPSG:3006")

        # Convert the GeoDataFrame to the Web Mercator projection (used by most contextily basemaps)
        gdf = gdf.to_crs(epsg=3857)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, alpha=0.5, color='blue')  # Adjust alpha and color to your preference

        # Add basemap with contextily
        ctx.add_basemap(ax)

        # Optionally set bounds
        ax.set_xlim([gdf.total_bounds[0] - 1000, gdf.total_bounds[2] + 1000])
        ax.set_ylim([gdf.total_bounds[1] - 1000, gdf.total_bounds[3] + 1000])

        ax.axis('off')  # Turn off axis
        plt.show()

from typing import List

class PreferredLocations:
    """
    Represents a collection of preferred locations categorized by type.
    
    Each attribute in this class represents a different type of preferred location.
    While some locations have only one preferred spot (e.g., schools), others
    can have multiple preferred spots (e.g., leisure locations).
    """

    instances = []
    all_locations_coords = []  # Class-level list to store coordinates
    
    def __init__(self, locations: List[Location]):
        self.EDUCATION_förskola: Location = None
        self.EDUCATION_förskoleklass: Location = None
        self.EDUCATION_grundskola: Location = None
        self.EDUCATION_gymnasieskola: Location = None
        self.EDUCATION_fritidshem: Location = None
        self.LEISURE_sports: List[Location] = []
        self.LEISURE_playground: List[Location] = []
        self.EDUCATION: List[Location] = []
        self.SHOPPING_GROCERY: List[Location] = []
        self.SHOPPING_OTHER: List[Location] = []
        self.LEISURE: List[Location] = []
        self.HEALTHCARE: List[Location] = []
        self.origin: None
        self.instances.append(self)
        for location in locations:
            self.all_locations_coords.append(location.location_coordinates)

        for location in locations:
            if location.location_type == "EDUCATION_förskola":
                self.EDUCATION_förskola = location
            elif location.location_type == "EDUCATION_förskoleklass":
                self.EDUCATION_förskoleklass = location
            elif location.location_type == "EDUCATION_grundskola":
                self.EDUCATION_grundskola = location
            elif location.location_type == "EDUCATION_gymnasieskola":
                self.EDUCATION_gymnasieskola = location
            elif location.location_type == "EDUCATION_fritidshem":
                self.EDUCATION_fritidshem = location
            elif location.location_type == "LEISURE_sports":
                self.LEISURE_sports.append(location)
            elif location.location_type == "LEISURE_playground":
                self.LEISURE_playground.append(location)
            elif location.location_type == "EDUCATION":
                self.EDUCATION.append(location)
            elif location.location_type == "SHOPPING_OTHER":
                self.SHOPPING_OTHER.append(location)
            elif location.location_type == "LEISURE":
                self.LEISURE.append(location)
            elif location.location_type == "HEALTHCARE":
                self.HEALTHCARE.append(location)
            elif location.location_type == "SHOPPING_GROCERY":
                self.SHOPPING_GROCERY.append(location)
    
    def __repr__(self):
        return (
            f"Preferred locations for this household:\n"
            f"  EDUCATION_förskola: {self.EDUCATION_förskola}\n"
            f"  EDUCATION_förskoleklass: {self.EDUCATION_förskoleklass}\n"
            f"  EDUCATION_grundskola: {self.EDUCATION_grundskola}\n"
            f"  EDUCATION_gymnasieskola: {self.EDUCATION_gymnasieskola}\n"
            f"  EDUCATION_fritidshem: {self.EDUCATION_fritidshem}\n"
            f"  LEISURE_sports: {self.LEISURE_sports}\n"
            f"  LEISURE_playground: {self.LEISURE_playground}\n"
            f"  EDUCATION: {self.EDUCATION}\n"
            f"  SHOPPING_OTHER: {self.SHOPPING_OTHER}\n"
            f"  LEISURE: {self.LEISURE}\n"
            f"  HEALTHCARE: {self.HEALTHCARE}\n"
            f"  SHOPPING_GROCERY: {self.SHOPPING_GROCERY}\n"
            f"  origin: {self.origin}\n"
        )

    def get_dict(self):
        dictionary = {
            "EDUCATION_förskola": self.EDUCATION_förskola,
            "EDUCATION_förskoleklass": self.EDUCATION_förskoleklass,
            "EDUCATION_grundskola": self.EDUCATION_grundskola,
            "EDUCATION_gymnasieskola": self.EDUCATION_gymnasieskola,
            "EDUCATION_fritidshem": self.EDUCATION_fritidshem,
            "LEISURE_sports": self.LEISURE_sports,
            "LEISURE_playground": self.LEISURE_playground,
            "EDUCATION": self.EDUCATION,
            "SHOPPING_OTHER": self.SHOPPING_OTHER,
            "LEISURE": self.LEISURE,
            "HEALTHCARE": self.HEALTHCARE,
            "SHOPPING_GROCERY": self.SHOPPING_GROCERY,
            "origin": self.origin
        }
        return dictionary



    def random_location(self):
        """Returns a random preferred location."""
        # Extracting all locations into a flat list
        return random.choice(self.all_locations_coords)

    def return_gdf(self):
        """Returns a GeoDataFrame of the preferred locations."""
        # Extracting all locations into a flat list
        all_locations = []
        origin = self.origin
        for attr, value in self.__dict__.items():
            # Skip origin
            if attr == "origin":
                continue
            if isinstance(value, list):
                all_locations.extend(value)
            elif value is not None:
                all_locations.append(value)
        
        # Convert locations to GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'LocationType': [loc.location_type for loc in all_locations],
            'geometry': [loc.location_coordinates for loc in all_locations]
        })

        # Add origin (Point) to GeoDataFrame
        gdf.loc[-1] = ["Origin", origin]
        
        # Set crs to EPSG:3006
        gdf.crs = "EPSG:3006"
        
        return gdf
    
    def plot(self,figsize=(10,10),ax=None):

        """Plots the preferred locations using different colors for each activity type."""
        gdf = self.return_gdf()
        
        # Defining colors for each location type for visualization
        colors = {
            "EDUCATION_förskola": "blue",
            "EDUCATION_förskoleklass": "cyan",
            "EDUCATION_grundskola": "green",
            "EDUCATION_gymnasieskola": "yellow",
            "EDUCATION_fritidshem": "purple",
            "LEISURE_sports": "red",
            "LEISURE_playground": "orange",
            "EDUCATION": "pink",
            "SHOPPING_GROCERY": "brown",
            "SHOPPING_OTHER": "gray",
            "LEISURE": "magenta",
            "HEALTHCARE": "black"
        }

        # Plotting
        if not ax:
            fig, ax = plt.subplots()
        for location_type, color in colors.items():
            gdf[gdf['LocationType'] == location_type].plot(ax=ax, color=color, label=location_type)

        ax.legend(loc="upper left")
        plt.title("Preferred Locations by Activity Type")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()
    

    @classmethod
    def return_convex_hull(cls):
        """Returns the convex hull of all preferred locations."""
        
        # Convert list of coordinates to MultiPoint
        #Chcekc that all_locations_coords is not empty
        if len(cls.all_locations_coords) == 0:
            raise ValueError("There are no preferred locations in the simulation.")
        
        multi_point = MultiPoint(cls.all_locations_coords)

        # Return the convex hull
        return multi_point.convex_hull



