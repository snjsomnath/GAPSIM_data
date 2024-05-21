
# -----------------------------------------------------
# od.py
# -----------------------------------------------------
# Description: Contains the helper functions used to sort out OD assignment.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.od"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import logging
import matplotlib.pyplot as plt
import geopandas as gpd
import psycopg2
import uuid
import logging
import re
import shapely
import random
import pandas as pd

# Importing classes

from tripsender.household import Household
from tripsender.building import Building
from tripsender.house import House
from tripsender.person import Person

#from tripsender.household import PreferredDestination
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)


def get_pgcon(local=True):
    """
    A helper function to read the `.pgpass` file. This is the authentication file for the `PostGIS` server. 
    The function reads the file and returns a connection object to the `PostGIS` server.
    The server contains the spatial data for the region, including the road network and building footprints.

    :param local: (bool: Optional) If ``True``, the function will connect to the local server.
    :return: (psycopg2 connection) A connection object to the `PostGIS` server.

    
    """
    host, port, database, user, password = '','','','',''
    f = open("secrets/.pgpass", "r")
    for i, line in enumerate(f):
        if i == local:
            host, port, database, user, password = line.split(':')
    # Create connection string
    con = psycopg2.connect(database=database.strip(),
                           user=user.strip(),
                           password=password.strip(),
                           host=host.strip())
    return con

def get_gdf(area,   #TODO Fix area naming convention to match PXWEB naming
            feature,
            KEY='',
            VAL='',
            title='Data from Server',
            filter=False,
            local=True,
            web=True,
            plot=False,
            save=True):
    """
    A function to fetch spatial geometry from the ``PostGIS`` server as ``GeoPandas GeoDataFrame``.
    This function is simply a wrapper for the ``psycopg2`` module.
    It constructs an ``SQL`` query based on the the params provided and requests data from the server.

    :param area: (str) The name of the Primary area in the Model naming format.
    :param feature: (str) The feature name to select from the ``PostGIS`` database. (Refer to PostGIS naming convention)
    :param KEY: (str: Optional) An optional attribute to filter from the data at the server level
    :param VAL: (str: Optional) An optional value for a given key to match from the data at the server level
    :param title: (str: Optional) An optional title for the plot
    :param filter: (bool: Optional) An optional input to specify if the data must be filtered at the server level
    :param local: (bool: Optional) Which server to fetch the data from
    :param web: (bool: Optional) If ``True``, the result will be reprojected
    :param plot: (bool: Optional) If ``True``, the result will be plotted
    :return: (GeoPandas GeoDataFrame, plot(Optional)) The resulting `GeoPandas GeoDataFrame`.
    """
    # If area has numbers in it, use  re.search(r"\d+\s*(.*)", area).group(1)
    # Clean area name
    if any(char.isdigit() for char in area):
        area = re.search(r"\d+\s*(.*)", area).group(1)
    else:
        pass

    layer = ''
    if feature == 0:
        layer = "se_got_trs_roads_tv"
    elif feature == 1:
        layer = "se_got_phy_buildingfootprint_lm"
    selectInString = 'SELECT ' + layer + '.geom'

    for k in KEY:
        selectOutString = selectInString + ' , ' + layer + '.' + k
        selectInString = selectOutString
    SELECT = selectInString

    # Adding conditionals
    if filter:
        COND = '\' AND ' + layer + '.' + KEY[0] + ' = \'' + VAL
    else:
        COND = ''

    # Building SQL
    FROM = ' FROM ' + layer
    JOIN = ' JOIN se_got_bnd_admin10_osm ON '
    CONTAIN = 'ST_Contains(se_got_bnd_admin10_osm.geom,' + layer + '.geom)'
    WHERE = ' WHERE se_got_bnd_admin10_osm.name =\'' + area + COND + '\';'
    sql = SELECT + FROM + JOIN + CONTAIN + WHERE
    ####logger.info(sql)

    # Get secrets
    con = get_pgcon(local=local)
    gdf = gpd.read_postgis(sql=sql,
                           con=con,
                           geom_col='geom')
    # Switching database
    if web:
        result = gdf.to_crs("EPSG:4326")
    else:
        result = gdf

    # Plotting
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(right=0.7)
        col = result.columns[1]
        title = area + ' ' + title
        ax.set_title(title)

        result.plot(column=col,
                    legend=True,
                    legend_kwds=dict(loc='upper left', bbox_to_anchor=(1, 1)), ax=ax)
        if save:
            plt.savefig('static/' + area + '.png')

    return result

def get_road(local=True, web=True, ped=True, clip=None, buffer_distance = 1000):
    """
    A function to fetch road data as a GeoPandas GeoDataFrame, with an option to clip it using a provided geometry and buffer the clip.

    :param local: If True, data will be fetched from the local server.
    :param web: If True, result will be reprojected to EPSG:4326.
    :param ped: If True, pedestrian roads will be included in the result.
    :param clip: Geometry to clip the road data. Expected to be a Shapely Polygon.
    :param buffer_distance: The distance to buffer around the clip geometry, default is 1000 meters.
    :return: A GeoDataFrame containing the road data.
    """
       
    con = get_pgcon(local=local)
    
    # Initialize a list to hold individual queries
    queries = []

    # Base SQL query for roads
    road_query = 'SELECT geom FROM se_got_trs_roads_tv'
    queries.append(road_query)
    
    # Add pedestrian roads if needed
    if ped:
        ped_query = 'SELECT geom FROM se_got_trs_path_tv'
        queries.append(ped_query)
    
    # Modify individual queries to clip with provided polygon
    if clip is not None:
        wkt = clip.buffer(buffer_distance).wkt  # Buffer 1km and convert Shapely Polygon to WKT
        # Apply the WHERE clause to each query
        queries = [q + f" WHERE ST_Intersects(geom, ST_GeomFromText('{wkt}', 3006))" for q in queries]

    # Combine the queries with UNION ALL
    sql = ' UNION ALL '.join(queries)

    #print("Final SQL Query:", sql)  # Debugging line

    gdf = gpd.read_postgis(sql=sql, con=con, geom_col='geom')

    if web:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf

def get_landuse(local=False, web=False, clip=None, buffer_distance=1000):
    """
    A function to fetch land use data as a GeoPandas GeoDataFrame, 
    with an option to clip it using a provided geometry and buffer the clip.
    
    :param local: If True, data will be fetched from the local server.
    :param web: If True, result will be reprojected to EPSG:4326.
    :param clip: Geometry to clip the natural features. Expected to be a Shapely Polygon.
    :param buffer_distance: The distance to buffer around the clip geometry, default is 1000 meters.
    :return: A GeoDataFrame containing the land use data.
    """
    con = get_pgcon(local=local)
    
    # Start with base SQL query to select all columns
    sql = 'SELECT * FROM se_got_phy_naturalfeatures_lm'
    
    # If a clipping polygon is provided, modify the SQL query to include a WHERE clause
    if clip is not None:
        # Validate the clip object is a Polygon and buffer it
        if isinstance(clip, shapely.geometry.Polygon):
            wkt = clip.buffer(buffer_distance).wkt
            sql += f' WHERE ST_Intersects(geom, ST_GeomFromText(\'{wkt}\', 3006))'
        else:
            raise ValueError('The clip parameter must be a Shapely Polygon.')
    
    # Fetch the data from the database
    try:
        gdf = gpd.read_postgis(sql=sql, con=con, geom_col='geom')
    except Exception as e:
        raise ConnectionError(f"An error occurred while connecting to the database: {e}")
    
    # Reproject if web is True
    if web:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def get_pg_query(sql, local=True, web=True):
    """
    A utility function that fetches any data using an ``SQL`` query from the ``PostGIS`` Server
    This function is a basic wrapper for the ``psycopg2`` module with secrets included.

    :param sql: (str) An SQL query string
    :param local: (bool: Optional) If ``True``, data will be fetched from the local server
    :param web: (bool: Optional) If ``True``, result will be reprojected to ``EPSG:4326``
    :return:
    """
    con = get_pgcon(local=local)
    ####logger.info(sql)
    gdf = gpd.read_postgis(sql=sql,
                           con=con,
                           geom_col='geom')
    if web:
        result = gdf.to_crs("EPSG:4326")
    else:
        result = gdf
    return result

def process_residential_buildings(gdf_building, area_per_person=36):

    """
    This function processes the residential buildings to calculate the number of people living in each building.
    The function filters the buildings based on the area_per_person parameter and calculates the number of people living in each building.
    The function also calculates the number of floors in each building and the total Built-up Area (BTA) of each building.
    The function also assigns a UUID to each building.

    The data for the area_per_person parameter is based on the average living area per person in Sweden. 
    Reference: https://www.scb.se/en/finding-statistics/statistics-by-subject-area/household-finances/income-and-income-distribution/households-housing/pong/statistical-news/households-housing-2019/

    A service area factor is applied to the area of the building based on the building type.
    The service area factor is 0.85 for 'Flerfamiljshus' and 0.9 for all other building types.

    :param gdf_building: (GeoPandas GeoDataFrame) A GeoDataFrame containing the building footprints.
    :param area_per_person: (int: Optional) The area per person in square meters. Default is 36 m2.
    :return: (GeoPandas GeoDataFrame) A GeoDataFrame containing the processed residential buildings.
    """
    
    # Average living area per person
    # https://www.scb.se/en/finding-statistics/statistics-by-subject-area/household-finances/income-and-income-distribution/households-housing/pong/statistical-news/households-housing-2019/
    # Split andamal_1t by ';' and create a new column for each value, new columns are byggnadstyp and byggnadsundergrupp
    gdf_building['byggnadstyp'] = gdf_building['andamal_1t'].str.split(';').str[0]
    gdf_building['byggnadsundergrupp'] = gdf_building['andamal_1t'].str.split('; ').str[1]

    # Filter Bostad only
    gdf_residential = gdf_building[gdf_building['byggnadstyp'] == 'Bostad']

    # Remove buildings with area less than area_per_person m2
    gdf_residential = gdf_residential[gdf_residential['geom'].area > area_per_person]

    # Plot the counts of each byggnadsundergrupp in gdf_residential
    #gdf_residential['byggnadsundergrupp'].value_counts().plot(kind='bar')

    # Get number of floors from height where each floor is 3m high and round up such that minimum floors is 1
    gdf_residential['floors'] = gdf_residential['height'].apply(lambda x: round(x/3.5,0)).apply(lambda x: max(x,1))

    # Get BTA by calculating area of the shape and multiply by number of floors and round down
    gdf_residential['area'] = gdf_residential['geom'].area

    # Histo gram of areas for each byggnadsundergrupp
    #gdf_residential.hist(column='area', by='byggnadsundergrupp', bins=50, figsize=(20,15))

    # If house is 'Flerfamiljshus', multiply by 0.85 to account for service areas
    gdf_residential.loc[gdf_residential['byggnadsundergrupp'] == 'Flerfamiljshus', 'area'] = gdf_residential['area'] * 0.7

    # For all other houses, multiply by 0.9 to account for service areas
    gdf_residential.loc[gdf_residential['byggnadsundergrupp'] != 'Flerfamiljshus', 'area'] = gdf_residential['area'] * 0.8

    # Get BTA by calculating area of the shape and multiply by number of floors and round down
    gdf_residential['BTA'] = gdf_residential['area'] * gdf_residential['floors'].apply(lambda x: round(x))

    # Add a column for number of people living in each building level (footprint_area/area_per_person) and round down
    gdf_residential['population_per_floor'] = gdf_residential['area'].apply(lambda x: round(x/area_per_person))

    # Add a column for number of people living in each building (people_per_floor * number of floors)
    gdf_residential['population_total'] = gdf_residential['population_per_floor'] * gdf_residential['floors']

    # Add a uuid for each building using uuid4
    # Added automatically on creating he class
    #gdf_residential['uuid'] = gdf_residential.apply(lambda x: uuid.uuid4(), axis=1)
    #gdf_residential.head()
    return gdf_residential

def sort_households_by_type_and_count():
    """
    This function sorts the households by type and population into single family and multi family households
    Households are sorted by population in descending order.
    Returns:
    households_in_single_family_house (list): A list of single family households sorted by population
    households_in_multi_family_house (list): A list of multi family households sorted by population
    """

    households_in_single_family_house = [
        household for household in Household.instances if household.house_type == 'Villa']
    households_in_single_family_house.sort(key=lambda x: len(x.members), reverse=True)

    households_in_multi_family_house = [
        household for household in Household.instances if household.house_type in ['Apartment', 'Other']]
    households_in_multi_family_house.sort(key=lambda x: len(x.members), reverse=True)

    return households_in_single_family_house, households_in_multi_family_house

def sort_buildings_by_type_and_population():
    """
    This function sorts the buildings by type and population into single family and multi family buildings
    Buildings are sorted by population in descending order.
    Note: Population is calculated as the number of people living in the building based on the area_per_person parameter.

    Returns:
    single_family_buildings (list): A list of single family buildings sorted by population
    multi_family_buildings (list): A list of multi family buildings sorted by population
    """

    single_family_house = ['Småhus friliggande', 'Småhus kedjehus', 'Småhus radhus', 'Småhus med flera lägenheter']
    multi_family_house = ['Flerfamiljshus']

    single_family_buildings = [
        building for building in Building.instances if building.type in single_family_house]
    single_family_buildings.sort(key=lambda x: x.population_total, reverse=True)

    multi_family_buildings = [
        building for building in Building.instances if building.type in multi_family_house]
    multi_family_buildings.sort(key=lambda x: x.population_total, reverse=True)
    if len(multi_family_buildings) == 0:
        ValueError("No multi family buildings found")
    if len(single_family_buildings) == 0:
        ValueError("No single family buildings found")
    return single_family_buildings, multi_family_buildings

def assign_to_single_family_buildings(households, buildings):
    """
    This function assigns the households to single family buildings.
    """
    if len(households) < len(buildings):
        # If there are more buildings than households, assign one household to each building
        # Every building will have atleast 1 single family house
        buildings = buildings[:len(households)]
        for i, household in enumerate(households):
            building = buildings[i]
            house = House(household, building)
            house.area = building.area
        logger.info(f"assign_to_single_family_buildings: Assgned {len(households)} single family households to {len(buildings)} buildings")
    else:
        pass

def assign_to_multi_family_buildings(households, buildings, area_per_person):
    """
    This function assigns the households to multi family buildings.
    """
    num_buildings = len(buildings)
    if num_buildings == 0:
        logger.info("assign_to_multi_family_buildings: No buildings left to assign multi family households to")
        return

    num_households = len(households)
    cycles = num_households // num_buildings
    remaining_households = num_households % num_buildings

    for cycle in range(cycles):
        for building in buildings:
            household = households.pop(0)
            remaining_capacity = area_per_person * building.built_up_area - building.population_total
            if remaining_capacity > len(household.members):
                house = House(household, building)
                remaining_capacity -= len(household.members)
            else:
                households.insert(0, household)

    for i in range(remaining_households):
        building = buildings[i]
        household = households.pop(0)
        remaining_capacity = area_per_person * building.built_up_area - building.population_total
        if remaining_capacity > len(household.members):
            house = House(household, building)
            remaining_capacity -= len(household.members)
        else:
            households.insert(0, household)

def mark_empty_buildings():
    """
    This function marks the buildings as empty if the population_total is 0.
    """
    for building in Building.instances:
        if building.population_total == 0:
            building.isEmpty = True

def assign_households_to_buildings(gdf_residential, area_per_person=32):
    """
    This function assigns the households to the buildings based on the area_per_person parameter.
    """

    House.clear_instances()
    logger.info("Processing residential buildings")
    logger.info("Fetching single and multi family housesholds")
    households_in_single_family_house, households_in_multi_family_house = sort_households_by_type_and_count()
    logger.info("Fetching single and multi family buildings")
    single_family_buildings, multi_family_buildings = sort_buildings_by_type_and_population()
    logger.info(f"Number of single family buildings: {len(single_family_buildings)} for {len(households_in_single_family_house)} households")
    logger.info(f"Number of multi family buildings: {len(multi_family_buildings)} for {len(households_in_multi_family_house)} households")
    
    # If no buildings are found in single and multi family houses, raise an error
    if len(single_family_buildings) == 0 and len(households_in_single_family_house) !=0:
        raise Exception("No buildings found in single family houses while there are households in single family houses")
    
    if len(multi_family_buildings) == 0 and len(households_in_multi_family_house) !=0:
        raise Exception("No buildings found in multi family houses while there are households in multi family houses")
    
    if single_family_buildings and households_in_single_family_house:
        logger.info("Processing single family houses")
        assign_to_single_family_buildings(households_in_single_family_house, single_family_buildings)
    if multi_family_buildings and households_in_multi_family_house:
        logger.info("Processing multi family houses")
        assign_to_multi_family_buildings(households_in_multi_family_house, multi_family_buildings, area_per_person)
    
    logger.info("Marking empty buildings")

    # Update the house_type at the person level to match the house_type at the Household level
    for person in Person.instances:
        person.house_type = person.household.house_type
    mark_empty_buildings()

class ODMatrix():
    def __init__(self,df_matrix):
        self.matrix = df_matrix
        self.activity_sequences = None
    
    def compute_routes(self,nrc_drive,nrc_bike,nrc_walk,nrc_transit):
        """
        This function computes the routes and durations for the OD matrix based on the mode of transport.
        The function iterates over the rows of the OD matrix and computes the route and duration based on the mode of transport.
        It user the NetworkRoutingComputer objects for each mode of transport to compute the route and duration.

        :param nrc_drive: (NetworkRoutingComputer) A NetworkRoutingComputer object for driving
        :param nrc_bike: (NetworkRoutingComputer) A NetworkRoutingComputer object for biking
        :param nrc_walk: (NetworkRoutingComputer) A NetworkRoutingComputer object for walking
        :param nrc_transit: (NetworkRoutingComputer) A NetworkRoutingComputer object for transit
                
        """

        for index, row in self.matrix.iterrows():
            mode = row['mode']
            origin = row['O']
            destination = row['D']
            activity_sequence = row['activity_sequence']
            transit_activity = row['transit_activity']
            # Initialize route and duration variables
            route, duration = None, None
            # Compute route and duration based on the mode
            if mode in ['Car', 'Taxi', 'Moped', 'Other', 'Transportation service', 'Flight']:
                route, duration = nrc_drive.compute_route(origin, destination)
                #route = "This came from drive"
                #print("Route:", route) 
            elif mode in ["Bicycle/E-bike"]:
                route, duration = nrc_bike.compute_route(origin, destination)
                #route = "This came from bike"
                #print("Route:", route) 
            elif mode in ['Walking']:
                route, duration = nrc_walk.compute_route(origin, destination)
                #route = "This came from walk"
                #print("Route:", route) 
            elif mode in ['Train/Tram', 'Bus', 'Boat']:
                route, duration = nrc_transit.compute_route(origin, destination)
                #route = "This came from transit"
                #print("Route:", route)  
            
            # Here if the route and duration is None it is because either the origin or destination is same
            # or the purpose was travel and there is no specific location for travel
            # In such cases, we will assign the origin and destination as the route and duration as 0

            transit_activity.route = route
            transit_activity.calculated_duration = duration
            transit_activity.origin_coordinates = origin
            transit_activity.destination_coordinates = destination



def generate_od_matrix(num_samples=None, random_location = False):
    """ 
    This function generates an OD matrix based on the activity sequences of the persons.
    The process of generating the OD matrix is as follows:
    1 - Identify all the adults in the households
    2 - For each adult, identify the activity sequence
    3 - Get the location_mapping dictionary for the person
    4 - For each activity, identify the origin, destination and mode of transport
        4.1 - First origin is the home location
        4.2 - Last destination is the home location
        4.3 - Mode of transport is the mode of transport for the activity
    5 - Append the origin, destination, mode, person and activity sequence to the OD_pairs list
    6 - Create a dataframe from the OD_pairs list
    7 - Create an ODMatrix object from the dataframe
    8 - Return the ODMatrix object

    Some caveats:
    - If the destination purpose is Travel, the destination is set to None. This is done by the create_location_mapping function
    """
    logger.info("Generating OD Matrix...") 
    activity_sequences = []
    OD_pairs = []
    person_list = []
    activity_sequence_list = []

    if num_samples is None:
        num_samples = len(Person.instances)

    buildings = Building.instances
    houses = [house for building in buildings for house in building.houses]
    households = [house.household for house in houses]
    persons = [person for household in households for person in household.members]
    adults = [person for person in persons if person.age >= 18][0:num_samples]

    for person in adults:
        activity_sequence = person.activity_sequence
        activity_sequences.append(activity_sequence)

        if activity_sequence is not None:
            location_mapping = create_location_mapping(person, random_location = random_location)
            activities = activity_sequence.activities

            if activities:
                last_non_none_location = person.origin  # Initialize with home location

                for i in range(0, len(activities)-2, 2):
                    origin_activity = activities[i]
                    mode_activity = activities[i + 1]
                    destination_activity = activities[i + 2] if i + 2 < len(activities) else origin_activity

                    # Assigning locations
                    origin = location_mapping.get(origin_activity.purpose, last_non_none_location)
                    destination = location_mapping.get(destination_activity.purpose)

                    # If destination is None due to "Travel", keep last_non_none_location for next origin
                    if destination is not None:
                        last_non_none_location = destination

                    mode = mode_activity.mode

                    # Append to OD pairs
                    OD_pairs.append([origin, destination, mode, mode_activity, origin_activity.purpose, destination_activity.purpose])
                    person_list.append(person)
                    activity_sequence_list.append(activity_sequence)

    df = pd.DataFrame(OD_pairs, columns=['O', 'D', 'mode', "transit_activity" ,'O_purpose', 'D_purpose'])
    df['person'] = person_list
    df['activity_sequence'] = activity_sequence_list

    od_matrix = ODMatrix(df)
    od_matrix.activity_sequences = activity_sequences
    logger.info("OD Matrix successfully created...")
    return od_matrix


def create_location_mapping(person, random_location =False):
    """Helper function to create a mapping of activity purposes to locations.
    It is possible to select random locations for each activity purpose by setting the random_location flag to True.
    If random_location is False, the first location from the preferred locations is selected for each activity purpose.
    """
    preferred_locations = person.household.house.building.preferred_locations
    # Location assignments
    home_location = person.origin
    work_location = person.work_location.location_coordinates if person.work_location else None
    
    # Selecting locations based on the random flag
    if random_location:
        education_location = random.choice(preferred_locations.EDUCATION).location_coordinates
        shopping_location = random.choice(preferred_locations.SHOPPING_OTHER).location_coordinates
        shopping_groceries_location = random.choice(preferred_locations.SHOPPING_GROCERY).location_coordinates
        leisure_location = random.choice(preferred_locations.LEISURE).location_coordinates
        healthcare_location = preferred_locations.HEALTHCARE[0].location_coordinates
        #healthcare_location = random.choice(preferred_locations.HEALTHCARE).location_coordinates
    else:
        education_location = preferred_locations.EDUCATION[0].location_coordinates
        shopping_location = preferred_locations.SHOPPING_OTHER[0].location_coordinates
        shopping_groceries_location = preferred_locations.SHOPPING_GROCERY[0].location_coordinates
        leisure_location = preferred_locations.LEISURE[0].location_coordinates
        healthcare_location = preferred_locations.HEALTHCARE[0].location_coordinates

    # Handling other random location which is common between random and non-random selection
    other_location = preferred_locations.random_location()
    travel_location = person.origin  # Assuming there's no specific location for 'Travel'
    children_location = preferred_locations.EDUCATION_förskola.location_coordinates

    location_mapping = {
        "Home": home_location,
        "Work": work_location,
        "Education": education_location,
        "Shopping": shopping_location,
        "Grocery": shopping_groceries_location,
        "Leisure": leisure_location,
        "Other": other_location,
        "Healthcare": healthcare_location,
        "Travel": travel_location,
        "Pickup/Dropoff child": children_location
    }
    person.location_mapping = location_mapping
    return location_mapping
