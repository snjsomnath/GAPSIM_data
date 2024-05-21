
# -----------------------------------------------------
# io.py
# -----------------------------------------------------
# Description: Contains the functions for loading files.
# Author: Sanjay Somanath
# Last Modified: 2023-11-03
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.io"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

import osmnx as ox
import igraph as ig
from pyrosm import OSM
import sqlite3
from tripsender.household import Household
from tripsender.person import Person
from tripsender.activity import Activity
from tripsender.building import Building
from tripsender.house import House
import datetime
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)


# Constants
GRAPH_PATHS = {
    'drive': "data/osm/GOT_OSM_DRIVE.graphml",
    'walk': "data/osm/GOT_OSM_WALK.graphml",
    'bike': "data/osm/GOT_OSM_BIKE.graphml",
}

def fetch_igraph(graph_type):
    """
    Fetches an igraph graph based on the type specified (drive, walk, bike).

    Parameters:
    graph_type (str): The type of graph to fetch.

    Returns:
    ig.Graph: The igraph graph object.
    """
    graph_path = GRAPH_PATHS.get(graph_type)
    if graph_path is None:
        raise ValueError(f"Graph type '{graph_type}' is not supported.")
    
    logger.info(f"Loading {graph_type} networkx graph...")
    G_nx = ox.load_graphml(graph_path)
    logger.info(f"Converting {graph_type} networkx graph to igraph...")
    G_ig = ig.Graph.from_networkx(G_nx)
    return G_ig


def fetch_osm_graph(graph_type):
    """
    Fetches an igraph graph based on the type specified (drive, walk, bike).

    Attributes:
    graph_type (str): The type of graph to fetch.

    Returns:
    ig.Graph: The igraph graph object.
    """
    path_to_pbf = "data\osm\GOT_OSM.pbf"
    osm = OSM(path_to_pbf)
    nodes, edges = osm.get_network(nodes=True, network_type=graph_type)
    G_ig = osm.to_graph(nodes, edges)
    return G_ig




def write_to_database(area,year,od_matrix):
    """
    Writes the data to a database.

    Attributes:
    area (str): The area for which the data is being written.
    year (int): The year for which the data is being written.
    od_matrix (ODMatrix): The ODMatrix object containing the data to be written.
    """    

    date_today = datetime.date.today()
    # Format yyyymmdd
    date_today = date_today.strftime("%Y%m%d")

    db_name = str.format('data/processed/{}_tripsender_{}_{}.db', date_today,year,area)
    person_df = Person.return_dataframe()
    household_df = Household.return_dataframe()
    od_matrix_df = od_matrix.matrix
    house_df =  House.return_dataframe()
    building_df = Building.return_gdf().drop(columns=['footprint'])
    # Now lets define the database schema and setup the primary keys

    # Create a connection to the database
    conn = sqlite3.connect(db_name)

    # Create a cursor object
    cursor = conn.cursor()

    # Create the person table
    cursor.execute('''CREATE TABLE IF NOT EXISTS person (
        uuid TEXT PRIMARY KEY,
        uuid_household TEXT,
        uuid_parent TEXT,
        age INTEGER,
        sex TEXT,
        type_household TEXT,
        household TEXT,
        has_car BOOLEAN,
        child_count INTEGER,
        is_head BOOLEAN,
        is_child BOOLEAN,
        origin TEXT,
        activity_sequence TEXT,
        primary_status TEXT,
        age_group TEXT,
        location_work TEXT,
        type_house TEXT,
        has_child BOOLEAN,
        location_mapping TEXT,
        FOREIGN KEY (uuid_household) REFERENCES household(uuid),
        FOREIGN KEY (uuid_parent) REFERENCES person(uuid));''')

    # Create the household table
    cursor.execute('''CREATE TABLE IF NOT EXISTS household (
        uuid TEXT PRIMARY KEY,
        name_category TEXT,
        count_children INTEGER,
        bool_children BOOLEAN,
        count_adults INTEGER,
        count_members INTEGER,
        uuid_members TEXT,             -- Assuming this stores a list, consider normalization if possible
        type_house TEXT,
        uuid_house TEXT,               -- Foreign key linking to the house table
        count_cars INTEGER,
        head_of_household TEXT,        -- Foreign key linking to the person table
        FOREIGN KEY (uuid_house) REFERENCES house(uuid),
        FOREIGN KEY (head_of_household) REFERENCES person(uuid)
    );''')

    # Create the od_matrix table
    cursor.execute('''CREATE TABLE IF NOT EXISTS od_matrix (
        uuid TEXT PRIMARY KEY,
        origin TEXT,
        destination TEXT,
        mode TEXT,
        distance REAL,
        duration REAL,
        FOREIGN KEY (origin) REFERENCES building(uuid),
        FOREIGN KEY (destination) REFERENCES building(uuid));''')

    # Create the house table
    cursor.execute('''CREATE TABLE IF NOT EXISTS od_matrix (
        origin TEXT,
        destination TEXT,
        mode TEXT,
        transit_activity TEXT,
        origin_purpose TEXT,
        destination_purpose TEXT,
        activity_sequence TEXT,
        uuid_person TEXT,               -- Foreign key linking to the person table
        PRIMARY KEY (origin, destination, uuid_person),  -- Composite primary key, adjust as needed
        FOREIGN KEY (uuid_person) REFERENCES person(uuid)
    );''')

    # Create the building table
    cursor.execute('''CREATE TABLE IF NOT EXISTS building (
        uuid TEXT PRIMARY KEY,
        type_building TEXT,
        area_square_meters REAL,
        height_meters REAL,
        count_floors INTEGER,
        population_per_floor INTEGER,
        population_total INTEGER,
        built_up_area REAL,
        count_workers INTEGER,
        is_empty BOOLEAN,
        building TEXT,  -- This column's purpose seems redundant given the table context, consider its necessity
        coord TEXT,  -- Assuming this stores coordinates; consider storing as separate latitude and longitude columns if applicable
        preferred_locations TEXT  -- Assuming this stores a list or complex data; consider normalization if it represents relationships
    );''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # Create a connection to the database
    conn = sqlite3.connect(db_name)


    # We need to convert the datatypes to the correct ones

    person_df['uuid'] = person_df['uuid'].astype(str)
    person_df['household_uuid'] = person_df['household_uuid'].astype(str)
    person_df['parent_uuid'] = person_df['parent_uuid'].astype(str)
    person_df['age'] = person_df['age'].astype(int)
    person_df['sex'] = person_df['sex'].astype(str)
    person_df['household_type'] = person_df['household_type'].astype(str)
    person_df['household'] = person_df['household'].astype(str)
    person_df['has_car'] = person_df['has_car'].astype(bool)
    person_df['child_count'] = person_df['child_count'].astype(int)
    person_df['is_head'] = person_df['is_head'].astype(bool)
    person_df['is_child'] = person_df['is_child'].astype(bool)
    person_df['origin'] = person_df['origin'].astype(str)
    person_df['activity_sequence'] = person_df['activity_sequence'].astype(str)
    person_df['primary_status'] = person_df['primary_status'].astype(str)
    person_df['age_group'] = person_df['age_group'].astype(str)
    person_df['work_location'] = person_df['work_location'].astype(str)
    person_df['house_type'] = person_df['house_type'].astype(str)
    person_df['has_child'] = person_df['has_child'].astype(bool)
    person_df['location_mapping'] = person_df['location_mapping'].astype(str)

    # Lets now rename all the columns
    renaming_dict = {
        'uuid': 'uuid',                             # Primary key
        'household_uuid': 'uuid_household',         # Links to the household table
        'parent_uuid': 'uuid_parent',               # Links to another person in the same table
        'age': 'age',
        'sex': 'sex',
        'household_type': 'type_household',
        'household': 'household',
        'has_car': 'has_car',
        'child_count': 'child_count',
        'is_head': 'is_head',
        'is_child': 'is_child',
        'origin': 'origin',
        'activity_sequence': 'activity_sequence',
        'primary_status': 'primary_status',
        'age_group': 'age_group',
        'work_location': 'location_work',
        'house_type': 'type_house',
        'has_child': 'has_child',
        'location_mapping': 'location_mapping'
    }

    person_df.rename(columns=renaming_dict, inplace=True)

    # Writing the person_df to database
    person_df.to_sql('person', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()
    # Next households

    # Create a connection to the database
    conn = sqlite3.connect(db_name)

    # We need to convert the datatypes to the correct ones

    household_df['uuid_household'] = household_df['uuid_household'].astype(str)
    household_df['name_category'] = household_df['name_category'].astype(str)
    household_df['count_children'] = household_df['count_children'].astype(int)
    household_df['bool_children'] = household_df['bool_children'].astype(bool)
    household_df['count_adults'] = household_df['count_adults'].astype(int)
    household_df['count_members'] = household_df['count_members'].astype(int)
    household_df['uuid_members'] = household_df['uuid_members'].astype(str)
    household_df['type_house'] = household_df['type_house'].astype(str)
    household_df['uuid_house'] = household_df['uuid_house'].astype(str)
    household_df['count_cars'] = household_df['count_cars'].astype(int)
    household_df['head_of_household'] = household_df['head_of_household'].astype(str) # UUID

    # Lets now rename all the columns

    renaming_dict = {
        'uuid_household': 'uuid',                   # Primary key
        'name_category': 'name_category',
        'count_children': 'count_children',
        'bool_children': 'bool_children',
        'count_adults': 'count_adults',
        'count_members': 'count_members',
        'uuid_members': 'uuid_members',             # List of UUIDs of the members
        'type_house': 'type_house',
        'uuid_house': 'uuid_house',                 # Links to the house table
        'count_cars': 'count_cars',
        'head_of_household': 'head_of_household'    # Links to the person table
    }

    household_df.rename(columns=renaming_dict, inplace=True)

    # Writing the person_df to database
    household_df.to_sql('household', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    # Next the od_matrix_df

    # The person column contains a person object, we need to create a  new column for the uuid of the person

    od_matrix_df['person_uuid'] = od_matrix_df['person'].apply(lambda x: x.uuid)

    # Create a connection to the database
    conn = sqlite3.connect(db_name)

    # We need to convert the datatypes to the correct ones

    #Attributes from transit_activity
    #self.start_time = parsed_datetime.time() if parsed_datetime else None
    #self.duration_minutes = duration_minutes
    #self.duration_timedelta = self.duration()
    #self.end_time = (datetime.combine(datetime.today(), self.start_time) + timedelta(minutes=duration_minutes)).time()
    #self.purpose = purpose
    #self.mode = mode
    #self.destination = None
    #self.destination_coordinates = None
    #self.origin = None
    #self.origin_coordinates = None
    #self.calculated_duration = None
    #self.route = None
    # Create a column for route from transit_activity.route

    od_matrix_df['route'] = od_matrix_df['transit_activity'].apply(lambda x: x.route)
    od_matrix_df['calculated_duration'] = od_matrix_df['transit_activity'].apply(lambda x: x.calculated_duration)
    od_matrix_df['duration_minutes'] = od_matrix_df['transit_activity'].apply(lambda x: x.duration_minutes)

    od_matrix_df['O'] = od_matrix_df['O'].astype(str)
    od_matrix_df['D'] = od_matrix_df['D'].astype(str)
    od_matrix_df['mode'] = od_matrix_df['mode'].astype(str)
    od_matrix_df['transit_activity'] = od_matrix_df['transit_activity'].astype(str)
    od_matrix_df['route'] = od_matrix_df['route'].astype(str)
    
    od_matrix_df['calculated_duration'] = od_matrix_df['calculated_duration'].astype(float)
    od_matrix_df['duration_minutes'] = od_matrix_df['duration_minutes'].astype(float)

    od_matrix_df['O_purpose'] = od_matrix_df['O_purpose'].astype(str)
    od_matrix_df['D_purpose'] = od_matrix_df['D_purpose'].astype(str)
    od_matrix_df['person'] = od_matrix_df['person'].astype(str)
    od_matrix_df['activity_sequence'] = od_matrix_df['activity_sequence'].astype(str)
    od_matrix_df['person_uuid'] = od_matrix_df['person_uuid'].astype(str)

    # Lets now rename all the columns
        


    renaming_dict = {
        'O': 'origin',                              # Primary key
        'D': 'destination',
        'mode': 'mode',
        'transit_activity': 'transit_activity',
        'route': 'route',
        'calculated_duration': 'calculated_duration',
        'duration_minutes': 'sampled_duration',
        'O_purpose': 'origin_purpose',
        'D_purpose': 'destination_purpose',
        'activity_sequence': 'activity_sequence',
        'person_uuid': 'uuid_person'                 # Links to the person table
    }

    od_matrix_df.rename(columns=renaming_dict, inplace=True)

    # Writing the person_df to database
    od_matrix_df.to_sql('od_matrix', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    # Next the house_df

    # Create a connection to the database
    conn = sqlite3.connect(db_name)

    # We need to convert the datatypes to the correct ones

    house_df['House UUID'] = house_df['House UUID'].astype(str)
    house_df['Household UUID'] = house_df['Household UUID'].astype(str)
    house_df['Building UUID'] = house_df['Building UUID'].astype(str)
    house_df['Members in house'] = house_df['Members in house'].astype(int)
    house_df['Adults in house'] = house_df['Adults in house'].astype(int)
    house_df['Children in house'] = house_df['Children in house'].astype(int)
    house_df['Cars in the household'] = house_df['Cars in the household'].astype(int)
    house_df['Area'] = house_df['Area'].astype(str)

    # Lets now rename all the columns

    renaming_dict = {
        'House UUID': 'uuid',                   # Primary key
        'Household UUID': 'uuid_household',     # Links to the household table
        'Building UUID': 'uuid_building',       # Links to the building table
        'Members in house': 'count_members',
        'Adults in house': 'count_adults',
        'Children in house': 'count_children',
        'Cars in the household': 'count_cars',
        'Area': 'area_square_meters'
    }

    house_df.rename(columns=renaming_dict, inplace=True)

    # Writing the person_df to database
    house_df.to_sql('house', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()
    # Finally the building_df

    # Create a connection to the database
    conn = sqlite3.connect(db_name)

    # We need to convert the datatypes to the correct ones
    #Index(['uuid', 'type', 'area', 'height', 'floors', 'population_per_floor',
    ##       'population_total', 'built_up_area', 'workers', 'is_empty', 'building',
    #      'coord', 'preferred_locations'],
    #     dtype='object')

    building_df['uuid'] = building_df['uuid'].astype(str)
    building_df['type'] = building_df['type'].astype(str)
    building_df['area'] = building_df['area'].astype(float)
    building_df['height'] = building_df['height'].astype(float)
    building_df['floors'] = building_df['floors'].astype(int)
    building_df['population_per_floor'] = building_df['population_per_floor'].astype(int)
    building_df['population_total'] = building_df['population_total'].astype(int)
    building_df['built_up_area'] = building_df['built_up_area'].astype(float)
    building_df['workers'] = building_df['workers'].astype(int)
    building_df['is_empty'] = building_df['is_empty'].astype(bool)
    building_df['building'] = building_df['building'].astype(str)
    building_df['coord'] = building_df['coord'].astype(str)
    building_df['preferred_locations'] = building_df['preferred_locations'].astype(str)

    # Lets now rename all the columns

    renaming_dict = {
        'uuid': 'uuid',                   # Primary key
        'type': 'type_building',
        'area': 'area_square_meters',
        'height': 'height_meters',
        'floors': 'count_floors',
        'population_per_floor': 'population_per_floor',
        'population_total': 'population_total',
        'built_up_area': 'built_up_area',
        'workers': 'count_workers',
        'is_empty': 'is_empty',
        'building': 'building',
        'coord': 'coord',
        'preferred_locations': 'preferred_locations'
        }

    building_df.rename(columns=renaming_dict, inplace=True)

    # Writing the person_df to database
    building_df.to_sql('building', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()