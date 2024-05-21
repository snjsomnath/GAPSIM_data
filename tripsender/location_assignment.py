# -----------------------------------------------------
# location_assignment.py
# -----------------------------------------------------
# Description: Functions to assign prefered locations to households.
# Author: Sanjay Somanath
# Created: 2023-10-28
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.location_assignment"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import logging
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.metrics import pairwise_distances

from tripsender.activity import Location
from tripsender.building import Building
from tripsender.building import PreferredLocations
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)


# Constants
ALL_AMENITIES_PATH = "data/processed/ALL_AMENITIES_updated.shp"
PROPOSED_AMENITIES_PATH = "data/processed/PROPOSED_AMENITIES_B.geojson"

gdf_amenities = gpd.read_file(ALL_AMENITIES_PATH)



class LocationFinder:
    """
    LocationFinder class helps in finding locations based on the given GeoDataFrame and location counts.
    
    This class uses BallTree data structure to efficiently query nearby locations. Different types of locations
    (like schools, playgrounds, healthcare, etc.) can be associated with different default counts to prioritize
    their significance.
    """
    
    def __init__(self, gdf, location_counts=None):
        """
        Initialize the LocationFinder with a given GeoDataFrame and optional location counts.

        Parameters:
        - gdf (GeoDataFrame): The input GeoDataFrame containing location data.
        - location_counts (dict, optional): A dictionary specifying the counts for different location types.
        """
        self.gdf = gdf
        self.ball_trees = {}  # Dictionary to store BallTrees for different location types
        count_multiple = 3   # A multiplier for location counts
        
        # Default counts for various location types if none are provided
        self.default_location_counts = {
            "EDUCATION_förskola" : 1,
            "EDUCATION_förskoleklass" : 1,
            "EDUCATION_grundskola" : 1,
            "EDUCATION_gymnasieskola" : 1,
            "EDUCATION_fritidshem" : 1,
            "LEISURE_sports" : count_multiple,
            "LEISURE_playground" : count_multiple,
            "EDUCATION" : count_multiple,
            "SHOPPING_OTHER"    : count_multiple,
            # "SHOPPING_GROCERY" is handled separately
            "LEISURE"   : count_multiple,
            "HEALTHCARE"    : count_multiple
        }

        # If custom location counts are provided, update the default counts
        if location_counts:
            self.default_location_counts.update(location_counts)

        self.populate_ball_trees()  # Create BallTrees for the location types
        self.set_grocery_data()    # Handle grocery data separately (method implementation is not provided)

    def populate_ball_trees(self):
        """
        Populate the BallTrees for the different location types based on the GeoDataFrame.
        
        This method creates a BallTree for each location type listed in default_location_counts, 
        allowing for efficient spatial queries. The BallTree, along with the associated GeoDataFrame
        subset and its count, is stored in the ball_trees dictionary for each location type.
        """
        
        # Iterate over each location type and its associated count
        for loc_type, count in self.default_location_counts.items():
            
            # Filter the GeoDataFrame based on the current location type
            temp_gdf = self.gdf[self.gdf['activity'] == loc_type]
            
            # Extract the coordinates from the 'geometry' column of the filtered GeoDataFrame
            coords = [(point.x, point.y) for point in temp_gdf['geometry'].values]
            
            # If there are coordinates (i.e., there are entries for this location type in the GeoDataFrame),
            # create a BallTree for them
            if coords:
                # Create a BallTree with the coordinates and use the euclidean metric for spatial queries
                tree = BallTree(np.array(coords), metric='euclidean')
                
                # Store the BallTree, the subset of the GeoDataFrame, and the count in the ball_trees dictionary
                self.ball_trees[loc_type] = (tree, temp_gdf, count)

        
    def set_grocery_data(self):
        """
        Initialize and set the grocery-related data attributes.
        
        This method extracts the data related to the "SHOPPING_GROCERY" activity from the main GeoDataFrame.
        It sets up the grocery GeoDataFrame, the coordinates of the grocery locations, and their associated areas.
        """
        logger.info("Setting up grocery data...")
        # Filter the main GeoDataFrame to extract only the rows related to "SHOPPING_GROCERY" activity
        self.grocery_gdf = self.gdf[self.gdf['activity'] == "SHOPPING_GROCERY"]
        
        # Extract the x and y coordinates from the 'geometry' column of the grocery GeoDataFrame
        self.grocery_coords = [(point.x, point.y) for point in self.grocery_gdf['geometry'].values]
        
        # Extract the 'area' values corresponding to each grocery location
        self.grocery_areas = self.grocery_gdf['area'].values

    def find_closest_locations(self, origin_point, k=None):
        """
        Find the closest locations to a given origin point for each location type.

        This method queries the BallTree for each location type to find the closest locations.
        The number of closest locations for each type is determined by the 'k' parameter or 
        the default count associated with the location type.

        Parameters:
        - origin_point (Point): The origin point from which distances are measured.
        - k (int, optional): The number of closest locations to return for each location type. 
                            If not provided, the default count for each type is used.

        Returns:
        - results (list): A list containing Location objects for each of the closest locations.
        """
        
        results = []  # List to store the resulting Location objects
        origin = (origin_point.x, origin_point.y)  # Convert origin point to tuple format

        # Iterate over each location type and its associated BallTree, GeoDataFrame subset, and count
        for loc_type, (tree, relevant_gdf, count) in self.ball_trees.items():
            #logger.info(f"Fetching location for location type : {loc_type}")
            # Determine the number of closest locations to query. Use provided k or default count.
            current_k = k if k is not None else count  

            # Query the BallTree to find the closest locations
            distances, indices = tree.query([origin], k=current_k)
            
            # Extract location data from the relevant GeoDataFrame for each of the closest locations
            for distance, index in zip(distances[0], indices[0]):
                closest_row = relevant_gdf.iloc[index]
                #logger.info(f"Found a location for amenity {loc_type} : {closest_row['name']}, {distance}m away")
                # Create a Location object and add it to the results list
                location = Location(loc_type, closest_row['name'], closest_row['geometry'], closest_row['amenity'])
                results.append(location)

        return results

    @staticmethod
    def gravity_score(distance, area, alpha=0, beta=2):
        """
        Calculate the gravity-based score for a location based on its area and distance from an origin.
        
        The gravity model, used here, is a spatial interaction model which is based on the idea 
        that the interaction between two places can be determined by the product of the size of 
        one (or both) and divided by their separation distance raised to a power (distance decay).
        
        Parameters:
        - distance (float): The distance from the origin to the location.
        - area (float): The size (area) of the location.
        - alpha (float, optional): The exponent for the area (default is 1.5).
        - beta (float, optional): The exponent for the distance decay (default is 2).
        
        Returns:
        - float: The gravity score for the location.
        """
        
        epsilon = 1e-10  # Small constant to prevent division by zero
        
        # Calculate the gravity score using the formula: (area^alpha) / (distance + epsilon)^beta
        return 1 / ((distance + epsilon)**beta)

    def find_closest_grocery_locations(self, origin_point, k=2):
        """
        Find the closest grocery locations to a given origin point based on a gravity-based scoring system.
        
        This method first computes the euclidean distance between the origin point and each grocery location.
        It then calculates a gravity-based score for each grocery location using the gravity_score method.
        The top 'k' grocery locations with the highest gravity scores are returned.
        
        Parameters:
        - origin_point (Point): The origin point from which distances and scores are calculated.
        - k (int, optional): The number of top-scoring grocery locations to return (default is 3).
        
        Returns:
        - results (list): A list containing Location objects for each of the top 'k' grocery locations.
        """
        
        # Convert the origin point to a list format [x, y]
        origin = [origin_point.x, origin_point.y]
        
        # Calculate the pairwise euclidean distances between the origin and each grocery location
        dist_metric = DistanceMetric.get_metric('euclidean')
        distances = dist_metric.pairwise([origin], self.grocery_coords).flatten()

        areas = self.grocery_areas  # Get the areas of the grocery locations
        # Set all areas to 1 to neutralize their effect
        areas = [1] * len(distances)  # Assuming there is one area value for each distance calculated

        # Compute gravity scores for each grocery location
        scores = [self.gravity_score(dist, area) for dist, area in zip(distances, areas)]
        
        # Get the indices of the top 'k' grocery locations based on the gravity scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []  # List to store the resulting Location objects

        # Extract location data from the grocery GeoDataFrame for each of the top 'k' locations
        for index in top_indices:
            row = self.grocery_gdf.iloc[index]
            
            # Create a Location object and add it to the results list
            location = Location("SHOPPING_GROCERY", row['name'], row['geometry'], row['amenity'])
            results.append(location)

        return results



def gravity_model_simulation(gdf_homes, gdf_jobs, density_weight=0.5,  distance_decay=2.5, plot=True):
    """
    Simulate job attraction based on the gravity model.
    
    The gravity model is a spatial interaction model that suggests that interaction between two places
    (for example, the number of people that commute from one place to another for work) is proportional to 
    the product of the population of the two places and inversely proportional to the square of the distance 
    between them.
    
    Parameters:
    - gdf_homes (GeoDataFrame): A GeoDataFrame containing home data. It must have 'footprint' and 'workers' columns.
    - gdf_jobs (GeoDataFrame): A GeoDataFrame containing job location data. It must have a 'job_density' column.
    - density_weight (float): The weight to apply to job density in the attraction calculation. Default is 1.
    - distance_decay (float): The power to which distance is raised in the attraction calculation. Default is 2.
    - plot (bool): Whether or not to plot the results. Default is True.
    
    Returns:
    - gdf_worker_jobs (GeoDataFrame): A GeoDataFrame representing the relationship between homes and potential job locations.

    To skew the attraction towards job density - increase the density_weight.
    To skew the attraction towards distance - increase the distance_decay.

    """
    
    # 1. Calculate Distance Matrix
    # Extracting the coordinates of home and job centroids
    homes_coords = gdf_homes.footprint.centroid.apply(lambda geom: (geom.x, geom.y)).tolist()
    jobs_coords = gdf_jobs.centroid.apply(lambda geom: (geom.x, geom.y)).tolist()
    # Computing the distance matrix
    distance_matrix = pairwise_distances(homes_coords, jobs_coords, metric="euclidean")

    # 2. Gravity Model Calculation
    # Compute attraction based on job density and distance decay
    attraction_matrix = (gdf_jobs['job_density'].values ** density_weight) / (distance_matrix ** distance_decay)

    def get_job_locations(row_index, n_jobs):
        """
        For each home, get the most attractive job locations based on the gravity model.
        
        Parameters:
        - row_index (int): The index of the home row.
        - n_jobs (int): The number of jobs associated with the home.
        
        Returns:
        - list: A list of the most attractive job locations' centroids.
        """
        # Getting the indices of job locations with the highest attraction
        top_jobs_indices = np.argsort(attraction_matrix[row_index])[-n_jobs:]
        unique_jobs = set()
        # Picking the top n_jobs from the list
        for idx in reversed(top_jobs_indices):
            if len(unique_jobs) < n_jobs:
                unique_jobs.add(idx)
        return gdf_jobs.iloc[list(unique_jobs)]['centroid'].tolist()

    # Assign potential job locations for each home based on number of workers and attraction
    gdf_homes['potential_jobs'] = [get_job_locations(idx, int(workers)) for idx, workers in enumerate(gdf_homes['workers'])]

    logger.info("Assigning jobs to workers...")

    # Assign job locations to workers
    assign_jobs_to_workers(gdf_homes)
    
    # Data aggregation
    # Constructing the resultant data
    data = []
    for _, home_row in gdf_homes.iterrows():
        home_footprint = home_row.footprint
        home_centroid = home_row.footprint.centroid
        for job_point in home_row['potential_jobs']:
            data.append({
                'home_footprint': home_footprint,
                'home_centroid': home_centroid,
                'worker': 1,
                'job_location': job_point
            })

    # Convert the list of dictionaries to a DataFrame and then to a GeoDataFrame
    df = pd.DataFrame(data)
    gdf_worker_jobs = gpd.GeoDataFrame(df, geometry='job_location', crs=gdf_homes.crs)

    if plot:
        # Visualize the result
        fig, ax = plt.subplots(figsize=(10,10))
        gpd.GeoSeries(gdf_worker_jobs['home_footprint']).plot(ax=ax, color='grey', alpha=0.5)
        gpd.GeoSeries(gdf_worker_jobs['home_centroid']).plot(ax=ax, color='red', markersize=1)
        gpd.GeoSeries(gdf_worker_jobs['job_location']).plot(ax=ax, color='blue', markersize=1)
        plt.show()

    return gdf_worker_jobs

def assign_jobs_to_workers(gdf_homes):
    """
    Assign job locations to workers residing in the buildings of gdf_homes.
    
    For each building in the GeoDataFrame, the function takes the potential job locations and 
    assigns these to every worker residing in that building.
    
    Parameters:
    - gdf_homes (GeoDataFrame): A GeoDataFrame containing home data. It should have 'building' 
      and 'potential_jobs' columns, where 'building' objects have a 'worker_list' attribute, and each 
      worker in this list has a 'work_location' attribute.
    
    Returns:
    - None: The function modifies the 'work_location' attribute of each worker in-place.
    """
    
    # Go through each row of the GeoDataFrame
    for _, row in gdf_homes.iterrows():
        building = row['building']           # Retrieve the building object
        job_locations = row['potential_jobs'] # Retrieve the potential job locations for this building
        
        # Iterate over each worker in the building
        for worker,job_point in zip(building.worker_list, job_locations):
            # Assign the job location to the worker's work_location attribute
            job_location = Location("WORK", "Work", job_point, "WORK")
            worker.work_location = job_location




def compute_job_density(file_path, radius=1000):
    """
    Computes job densities for each point in the GeoDataFrame using the given radius.
    
    The function loads a shapefile, calculates the centroid for each geometry and then computes
    the job density around each centroid using a given radius. The BallTree data structure is 
    utilized for efficient spatial queries.

    Parameters:
    - file_path (str): Path to the shapefile.
    - radius (float): Radius in meters for which job density is calculated. Default is 1000 meters.

    Returns:
    - gdf_jobs (GeoDataFrame): Updated GeoDataFrame with job densities.
    """
    logger.info("Computing job densities...")
    # Load data from the shapefile
    gdf = gpd.read_file(file_path)

    # Compute centroids for each geometry in the GeoDataFrame
    gdf['centroid'] = gdf['geometry'].centroid
    
    # Create a new GeoDataFrame with only the 'jobs' column and the computed centroids
    # Set the 'centroid' column as the geometry for this new GeoDataFrame
    gdf_jobs = gdf[['jobs', 'centroid']].copy()
    gdf_jobs = gdf_jobs.set_geometry('centroid')
    
    # Convert the GeoDataFrame's point geometries to a matrix format suitable for BallTree
    coordinates = np.array(list(zip(gdf_jobs.geometry.x, gdf_jobs.geometry.y)))
    
    # Construct a BallTree for efficient spatial queries
    ball_tree = BallTree(coordinates)
    
    # Define an inner function to perform spatial queries and calculate job density
    def job_density(point):
        """
        Compute job density for a given point by querying nearby jobs within the specified radius.
        
        Parameters:
        - point (tuple): A tuple representing the x and y coordinates of the point.

        Returns:
        - float: Job density for the given point.
        """
        # Get the indices of points within the given radius using the BallTree
        indices = ball_tree.query_radius([point], r=radius, return_distance=False)[0]
        # Sum up the jobs for the queried points to compute the density
        return gdf_jobs.iloc[indices]['jobs'].sum()

    # Compute job densities for each point in the GeoDataFrame
    gdf_jobs['job_density'] = [job_density(point) for point in coordinates]
    
    return gdf_jobs

def compute_preferred_locations(add_proposed_locations = False):
    """
    #TODO This can be made faster by grouping buildings into chunks of 200x200m and assigning them together.
    Computes and assigns preferred locations for each building instance based on their coordinates.
    
    For each building in the list of instances, this function:
    1. Determines the building's coordinate.
    2. Finds the closest general locations to that coordinate.
    3. Finds the closest grocery locations to that coordinate.
    4. Merges the two lists of locations.
    5. Initializes a PreferredLocations object using the combined locations.
    6. Assigns the PreferredLocations object to the building's preferred_locations attribute.
    """
    logger.info("Computing preferred locations...")
    gdf_amenities = gpd.read_file(ALL_AMENITIES_PATH)

    if add_proposed_locations:
        # Check if the proposed amenities file exists
        if os.path.exists(PROPOSED_AMENITIES_PATH):
            
            # Load the proposed amenities GeoJSON file
            gdf_proposed_amenities = gpd.read_file(PROPOSED_AMENITIES_PATH)

            if len(gdf_proposed_amenities) > 0:

                logger.info("Proposed amenities file found. Combining amenities for location assignment")
                # Set the CRS for the proposed amenities GeoDataFrame
                gdf_proposed_amenities.set_crs(epsg=3006, inplace=True,allow_override=True)

                # Concatenate the existing and proposed amenities into one GeoDataFrame
                gdf_amenities = pd.concat([gdf_amenities, gdf_proposed_amenities], ignore_index=True)

                logger.info(gdf_proposed_amenities.activity.value_counts())
        else:
            logger.info("Proposed amenities are empty or no proposed amenities file found. Proceeding with location assignment")

    # Initialize the LocationFinder object outside the loop 
    # (since it probably doesn't need to be re-initialized for each building).
    location_finder = LocationFinder(gdf_amenities)
    
    # Loop through all the building instances
    for building in Building.instances:
        # Extract the coordinate (centroid) of the building
        origin = building.coord
        
        # Find the closest general locations to the building's coordinate
        locations = location_finder.find_closest_locations(origin)
        
        # Find the closest grocery locations to the building's coordinate
        grocery_locations = location_finder.find_closest_grocery_locations(origin)
        
        # Initialize a PreferredLocations object using the combined lists of locations
        preferred_locations = PreferredLocations(locations + grocery_locations)
        preferred_locations.origin = origin
        
        # Assign the generated PreferredLocations object to the building's attribute
        building.preferred_locations = preferred_locations

    logger.info(f"Preferred locations computed for {len(Building.instances)} buildings.")

def assign_workers_to_buildings():
    """ 
    Assigns workers to buildings based on their primary status.
    """
    # Add workers to buildings
    for building in Building.instances:
        houses = building.houses
        for house in houses:
            household = house.household
            for member in household.members:
                # Workers is a list of workers with primary_status = "WORK"
                if member.primary_status == "WORK":
                    building.workers += 1
                    building.worker_list.append(member)