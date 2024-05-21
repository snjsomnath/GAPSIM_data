
# -----------------------------------------------------
# routing.py
# -----------------------------------------------------
# Description: Contains the functions to calculate shortest paths between OD pairs.
# Author: Sanjay Somanath
# Last Modified: 2023-10-29
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.routing"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import logging
import networkx as nx
import igraph as ig
from shapely.geometry import Point,MultiLineString, LineString
from scipy.spatial import cKDTree
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from tripsender.building import Building
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import tripsender.od as od
from rasterio.features import rasterize
import numpy as np
import tripsender.io as io
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from tripsender.activity import Route
import h5py
from scipy.spatial import KDTree
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

hdf5_path = "data/processed/od_matrix.h5"
origins_gdf = gpd.read_file("data/processed/origins.gpkg")

# Constants
COEF_LENGTH = 1/3
COEF_NATURE = 1/3
COEF_SLOPE = 1/3
dem_raster_path = "data/raw/dem/GOT_DEM.tif"
def get_edge_lengths(G_ig):
    """
    Compute the lengths of geometries for each edge in the graph.

    Args:
        G_ig (igraph.Graph): The input graph.

    Returns:
        list: A list of edge lengths.
    """
    logger.info("Computing lengths of geometries...")
    edge_lengths = [edge['length'] for edge in G_ig.es]
    return edge_lengths

def get_edge_travel_time(G_ig):
    """
    Compute the travel times for each edge in the graph.

    Args:
        G_ig (igraph.Graph): The input graph.

    Returns:
        list: A list of travel times.
    """
    logger.info("Computing lengths of geometries...")
    travel_time = [edge['travel_time'] for edge in G_ig.es]
    return travel_time

def get_vertex_coords(G_ig):
    """
    Get the coordinates of vertices in the graph.

    Args:
        G_ig (igraph.Graph): The input graph.

    Returns:
        list: A list of tuples representing vertex coordinates.
    """
    v_coords = [(vertex['x'], vertex['y']) for vertex in G_ig.vs]
    return v_coords

def get_edge_coords(G_ig):
    """
    Get the coordinates of edges in the graph.

    Args:
        G_ig (igraph.Graph): The input graph.

    Returns:
        list: A list of tuples representing edge coordinates.
    """
    edge_coords = []
    for edge in G_ig.es:
        # Get the source and target vertex indices of the edge
        source_vertex_id = edge.source
        target_vertex_id = edge.target
        
        # Retrieve the coordinates of the source vertex
        source_x = G_ig.vs[source_vertex_id]['x']
        source_y = G_ig.vs[source_vertex_id]['y']
        
        # Retrieve the coordinates of the target vertex
        target_x = G_ig.vs[target_vertex_id]['x']
        target_y = G_ig.vs[target_vertex_id]['y']
        
        # Append the coordinates as a tuple of tuples
        edge_coords.append(((source_x, source_y), (target_x, target_y)))

    return edge_coords

def find_closest_source_target(tree, G_ig, source_coord, targets_coords):
    """
    Find the closest source and target nodes in the graph using a KDTree.

    Args:
        tree (cKDTree): The KDTree for spatial indexing.
        G_ig (igraph.Graph): The input graph.
        source_coord (Point): The source coordinate.
        targets_coords (list): The target coordinates.

    Returns:
        tuple: Closest source and target nodes.
    """
    # Convert the POINT object to a list of coordinates for the source.
    source = [source_coord.x, source_coord.y]
    # Query the KDTree to find the index of the closest node to the source.
    _, source_index = tree.query([source], k=1)
    
    # Convert the list of POINT objects to a list of coordinate lists for the targets.
    targets = [[coord.x, coord.y] for coord in targets_coords]
    # Query the KDTree to find the indices of the closest nodes to the targets.
    _, target_indices = tree.query(targets, k=1)

    # Retrieve the closest source and target nodes from the graph using indices
    closest_source = G_ig.vs[source_index[0]]["name"]
    closest_targets = [G_ig.vs[index]["name"] for index in target_indices]

    return closest_source, closest_targets

##################### Raster weight helpers ##################### 
def read_and_clip_raster(dem_raster_path, convex_hull):
    """
    Read and clip a DEM raster using a convex hull.

    Args:
        dem_raster_path (str): Path to the DEM raster file.
        convex_hull (Polygon): The convex hull for clipping.

    Returns:
        tuple: Clipped raster image, transform, and metadata.
    """
    with rasterio.open(dem_raster_path) as dem_raster:  # Step 1
        bbox = box(*convex_hull.bounds)  # Step 2
        window = rasterio.windows.from_bounds(*bbox.bounds, dem_raster.transform)  # Step 3
        out_img = dem_raster.read(window=window, masked=True)  # Step 4
        out_transform = dem_raster.window_transform(window)  # Step 5
        out_meta = dem_raster.meta.copy()  # Step 6
        out_meta.update({
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform
        })
        
    return out_img, out_transform, out_meta

def process_landuse_data(landuse_vector, convex_hull, dem_raster_crs='EPSG:3006'):
    """
    Process land use data by clipping and dissolving geometries.

    Args:
        landuse_vector (GeoDataFrame): The land use vector data.
        convex_hull (Polygon): The convex hull for clipping.
        dem_raster_crs (str): The CRS of the DEM raster. Defaults to 'EPSG:3006'.

    Returns:
        GeoDataFrame: Processed and clipped land use vector data.
    """
    # Filter to include only specified land use types, assuming this is less computationally expensive than clipping
    landuse_vector = landuse_vector[landuse_vector['detaljtyp'].isin(['SKOGBARR', 'ÖPMARK', 'VATTEN', 'SKOGLÖV', 'ODLÅKER'])]

    # Reproject the GeoDataFrame if its CRS differs from the DEM raster's CRS
    if landuse_vector.crs != dem_raster_crs:
        landuse_vector = landuse_vector.to_crs(dem_raster_crs)

    # Clip the vector data using the bounding box of the convex hull
    bounding_box = box(*convex_hull.bounds)
    clipped_vector_landuse = gpd.clip(landuse_vector, bounding_box)

    # Dissolve the clipped geometries by land use type to merge geometries with the same 'detaljtyp'
    clipped_vector_landuse = clipped_vector_landuse.dissolve(by='detaljtyp')

    return clipped_vector_landuse

def create_gradient(clipped_vector_landuse, out_shape, out_transform, max_distance=200.0):
    """
    Create a gradient effect of natural features.

    Args:
        clipped_vector_landuse (GeoDataFrame): Clipped land use vector data.
        out_shape (tuple): Shape of the output raster.
        out_transform (Affine): Transform of the output raster.
        max_distance (float): Maximum distance for gradient scaling. Defaults to 200.0.

    Returns:
        ndarray: Gradient raster.
    """
    # Rasterize the vector data, simplifying geometries to improve performance
    rasterized_vector = rasterize(
        [(geom, 1) for geom in clipped_vector_landuse.geometry.simplify(tolerance=0.1)],
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype=np.uint8
    )

    # Calculate the distance from each '0' pixel (non-natural feature) to the nearest '1' pixel (natural feature)
    distance_from_nature = distance_transform_edt(rasterized_vector == 0)

    # Scale distances to meters in-place, assuming the CRS unit is meters
    pixel_size = out_transform[0]
    np.multiply(distance_from_nature, pixel_size, out=distance_from_nature)

    # Clip and scale distances to a 0-1 range in-place, within the specified max_distance
    np.clip(distance_from_nature / max_distance, 0, 1, out=distance_from_nature)

    # Invert the scaled distance in-place to create the gradient effect (decay from natural features)
    np.subtract(1, distance_from_nature, out=distance_from_nature)

    # Ensure that natural feature pixels retain their original value ('1')
    gradient = np.where(rasterized_vector == 1, 1, distance_from_nature)

    return gradient

def plot_gradient(gradient, transform, title='Gradient Effect of Natural Features', cmap='viridis', figsize=(10, 5)):
    """
    Plot the gradient effect of natural features.

    Args:
        gradient (ndarray): Gradient raster.
        transform (Affine): Transform of the raster.
        title (str): Title of the plot. Defaults to 'Gradient Effect of Natural Features'.
        cmap (str): Colormap for the plot. Defaults to 'viridis'.
        figsize (tuple): Figure size. Defaults to (10, 5).
    """
    # Calculate the extent of the raster based on the transform
    # This defines the min and max of the x and y axes
    height, width = gradient.shape
    extent = (
        transform[2], transform[2] + transform[0] * width,
        transform[5] + transform[4] * height, transform[5]
    )
    
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the image display of the gradient array using the specified colormap
    # and include the extent to map to geographic coordinates
    im = ax.imshow(gradient, cmap=cmap, extent=extent)
    
    # Add a colorbar to the plot, with a label indicating it shows normalized gradient values
    plt.colorbar(im, ax=ax, label='Normalized Gradient')
    
    # Set the title of the plot to the specified title
    ax.set_title(title)
    
    # Display the plot
    plt.show()

def rowcol(transform, x, y):
    """
    Convert real-world coordinates to pixel coordinates.

    Args:
        transform (Affine): Affine transform.
        x (float): X-coordinate.
        y (float): Y-coordinate.

    Returns:
        tuple: Row and column indices.
    """
    # Convert real-world coordinates to pixel coordinates using the affine transform
    col, row = ~transform * (x, y)  # Use the inverse transform to map (x, y) to (row, col)
    return int(row), int(col)

def raster_value_at_line_ends(raster, transform, start_coord, end_coord, value_type='average'):
    """
    Retrieve raster values at the ends of a line segment.

    Args:
        raster (ndarray): The raster data.
        transform (Affine): Affine transform of the raster.
        start_coord (tuple): Start coordinate.
        end_coord (tuple): End coordinate.
        value_type (str): Type of value to return ('average' or 'absolute_difference').

    Returns:
        float: The calculated value.
    """
    if value_type not in ('average', 'absolute_difference'):
        raise ValueError("value_type must be 'average' or 'absolute_difference'.")
    # Convert real-world coordinates to row and column indices
    row_start, col_start = rowcol(transform, *start_coord)
    row_end, col_end = rowcol(transform, *end_coord)

    # Retrieve raster values at start and end points, safely handling out-of-bounds indices
    start_value = raster[row_start, col_start] if 0 <= row_start < raster.shape[0] and 0 <= col_start < raster.shape[1] else np.nan
    end_value = raster[row_end, col_end] if 0 <= row_end < raster.shape[0] and 0 <= col_end < raster.shape[1] else np.nan

    # Calculate the desired value
    if value_type == 'average':
        # Only average non-nan values
        values = [v for v in [start_value, end_value] if not np.isnan(v)]
        val = np.mean(values) if values else np.nan  # Returns nan if both are nan or empty
    elif value_type == 'absolute_difference':
        val = abs(start_value - end_value) if not np.isnan(start_value) and not np.isnan(end_value) else np.nan

    return val

def normalise_list(input_list):
    """
    Normalize a list of values to a 0-1 range.

    Args:
        input_list (list): List of values.

    Returns:
        list: Normalized list of values.
    """
    list_min = min(input_list)
    list_max = max(input_list)
    
    return [(x-list_min)/(list_max-list_min) for x in input_list]


def compute_edge_weights(coords, edge_length, dem_raster_path , landuse_vector, convex_hull):
    """
    Compute edge weights for the graph based on edge length, slope, and proximity to natural features.

    Args:
        coords (list): List of coordinates for each edge.
        edge_length (list): List of edge lengths.
        dem_raster_path (str): Path to the DEM raster file.
        landuse_vector (GeoDataFrame): Land use vector data.
        convex_hull (Polygon): Convex hull for clipping.

    Returns:
        list: Computed edge weights.
    """
    # Clip the DEM raster to the area of interest defined by the convex hull
    clipped_dem, out_transform, out_meta = read_and_clip_raster(dem_raster_path, convex_hull)

    # Process the land use vector data, clipping it to the same area and creating a gradient
    processed_landuse = process_landuse_data(landuse_vector, convex_hull, out_meta['crs'])
    normalize_nature_raster = create_gradient(processed_landuse, clipped_dem.shape[1:], out_transform)

    average_nature_values = []
    height_difference = []

    # Iterate over each street segment to compute the average gradient and slope
    for line_coords in coords:
        if len(line_coords) > 1:
            # Get the raster values at the start and end of the street segment
            #print("Calculating dem absolute difference")
            absolute_difference = raster_value_at_line_ends(clipped_dem[0], out_transform, line_coords[0], line_coords[-1], value_type='absolute_difference')
            #print("Calculating landuse average")
            average_value = raster_value_at_line_ends(normalize_nature_raster, out_transform, line_coords[0], line_coords[-1], value_type='average')


            average_nature_values.append(average_value)
            height_difference.append(absolute_difference)


    slope = [x/y for x,y in zip(height_difference, edge_length)]

    norm_edge_length = normalise_list(edge_length)
    norm_distance_nature = normalise_list(average_nature_values)
    norm_slope = normalise_list(slope)
    
    weights = [COEF_LENGTH * norm_edge_length[i] + COEF_NATURE * norm_distance_nature[i] + COEF_SLOPE * norm_slope[i] for i in range(len(edge_length))]

    # Return the edge lengths and computed weight factors for further processing
    return weights

#####################  End of raster weight helpers #####################

def path_to_linestring(path_indices, G_ig,s,t):
    """
    Convert a path in the graph to a LineString.

    Args:
        path_indices (list): List of vertex indices representing the path.
        G_ig (igraph.Graph): The input graph.
        s (str): Source node identifier.
        t (str): Target node identifier.

    Returns:
        LineString or Point: The LineString representing the path or a Point if the path is a single point.
    """
    # End the script if the path is empty or has only one point.
    if not path_indices :
        raise ValueError("path_to_linestring: The routing has returned an empty path or a single point, which is not sufficient for creating a LineString")

    # Extract the coordinates of the path
    vertices = G_ig.vs[path_indices]
    x = vertices["x"]
    y = vertices["y"]
    
    if len(x) < 2:
        # Check if source and destination are the same
        if s == t:
            return Point(x[0], y[0])
        else:
            raise ValueError(f"path_to_linestring: The routing has returned a path with only one point. {s,t}")
    # Create the LineString for the path
    line = LineString(zip(x, y))

    return line


def process_building(building, tree, A, G_ig, route_cache, mode):
    """
    Process a single building to compute routes to preferred locations.

    Args:
        building (Building): The building to process.
        tree (cKDTree): KDTree for spatial indexing.
        A (ndarray): Array of vertex coordinates.
        G_ig (igraph.Graph): The input graph.
        route_cache (dict): Cache to store computed routes.
        mode (str): Mode of transportation ('walk', 'car', 'bike').

    Returns:
        Building: The updated building with computed routes.
    """
    source_coord = building.coord
    preferred_locations = building.preferred_locations
    
    for category, locations in preferred_locations.__dict__.items():

        if locations and isinstance(locations, (list, tuple)):
            targets_coords = [loc.location_coordinates for loc in locations]  # List of coordinate pairs

            # Call find_closest_source_target with correct formats
            closest_source, closest_targets = find_closest_source_target(tree, G_ig, source_coord, targets_coords)

            for idx, location in enumerate(locations):
                s = closest_source
                t = closest_targets[idx]

                # Use tuples directly as keys
                route_key = (s, t)
                if route_key in route_cache:
                    linestring = route_cache[route_key]
                else:
                    path_indices = G_ig.get_shortest_path(s, t, weights="weight", mode="out", output="vpath")
                    linestring = path_to_linestring(path_indices, G_ig,s,t)
                    route_cache[route_key] = linestring

                if mode == 'walk':
                    route = Route(mode, linestring, 4)
                    location.route_walk = route
                elif mode == 'car':
                    route = Route(mode, linestring, 45)
                    location.route_car = route
                elif mode == 'bike':
                    route = Route(mode, linestring, 10)
                    location.route_bike = route

    # Buildings are updated on the fly without parallel
    # However, in the case of a batch, we need to return the updated building
    # so they can be processed in multiple processes and then replace the Building.instances
    return building

def process_building_batch(building_batch, tree, A, G_ig, route_cache,mode):
    """
    Process a batch of buildings to compute routes in parallel.

    Args:
        building_batch (list): List of buildings to process.
        tree (cKDTree): KDTree for spatial indexing.
        A (ndarray): Array of vertex coordinates.
        G_ig (igraph.Graph): The input graph.
        route_cache (dict): Cache to store computed routes.
        mode (str): Mode of transportation ('walk', 'car', 'bike').

    Returns:
        list: List of updated buildings with computed routes.
    """
    logger.info("Processing building batch...")
    for building in building_batch:
        process_building(building, tree, A, G_ig, route_cache, mode)
    
    # For parallel processing, return the updated building batch
    return building_batch

def compute_routes_for_all_buildings(mode = 'walk', parallel=False): # landuse_vector, convex_hull, 
    """
    Compute routes for all buildings in the dataset.

    Args:
        mode (str): Mode of transportation ('walk', 'bike', 'drive'). Defaults to 'walk'.
        parallel (bool): Whether to process buildings in parallel. Defaults to False.
    """
    #Fetch the iGraph
    G_ig = io.fetch_igraph(mode)

    # Build a KDTree for spatial indexing
    logger.info("Building KDTree for spatial indexing...")
    v_coordinates = get_vertex_coords(G_ig)
    A = np.array(v_coordinates)
    tree = cKDTree(A)

    # Set names for vertices in the form "(x,y)"
    names = [f"({x},{y})" for x, y in v_coordinates]
    G_ig.vs["name"] = names

    # Set weights based on mode
    if mode in ['walk','bike']:
        #e_coordinates = get_edge_coords(G_ig)
        edge_length = get_edge_lengths(G_ig)
        weights = edge_length #compute_edge_weights(e_coordinates, edge_length, dem_raster_path, landuse_vector, convex_hull)
    elif mode == 'drive':
        edge_travel_time = get_edge_travel_time(G_ig)
        weights = edge_travel_time
    else:
        raise ValueError(f"Mode must be one of 'walk', 'bike' or 'drive', got {mode} instead.")
    G_ig.es["weight"] = weights

    # Create a cache to store computed routes
    route_cache = {}

    # Parallel processing for each building
    logger.info("Processing buildings...")
    batch_size = 10             # Number of buildings to process in each batch
    building_batches = [Building.instances[i:i+batch_size] for i in range(0, len(Building.instances), batch_size)]
    
    if parallel:
        # Capturing the list of lists
        processed_building_batches = Parallel(n_jobs=-1)(delayed(process_building_batch)(building_batch, tree, A, G_ig, route_cache, mode=mode) for building_batch in building_batches)
        
        # Flatten the list of lists
        processed_buildings = [building for batch in processed_building_batches for building in batch]
        
        # Replace Building.instances with the updated buildings
        Building.instances = processed_buildings
    else:
        for building_batch in building_batches:
            process_building_batch(building_batch, tree, A, G_ig, route_cache, mode=mode)

    logger.info("Finished processing computing routes.")



class TransitMatrixComputer:
    def __init__(self, hdf5_path = hdf5_path,origins_gdf = origins_gdf):
        """
        A class to compute and manage a transit matrix, providing travel times between origin-destination pairs.

        This class preprocesses data from an HDF5 file and uses a KDTree for fast lookup of travel times.
        It also provides methods to query travel times and compute routes between coordinates.

        Attributes:
            hdf5_path (str): Path to the HDF5 file containing the transit data.
            travel_times_dict (dict): Dictionary to store travel times between origin-destination pairs.
            origins_gdf (GeoDataFrame): GeoDataFrame containing origin points.
            points (ndarray): Array of coordinates extracted from the origins_gdf.
            tree (KDTree): KDTree for spatial indexing of origin points.
            ids (ndarray): Array of IDs corresponding to the origin points.

        Methods:
            preprocess_data():
                Preprocess the data from the HDF5 file and store it in a dictionary for fast lookups.
            get_closest_id_tree(lat, lon):
                Get the closest origin ID to the given latitude and longitude using the KDTree.
            query_travel_time(o, d):
                Query the travel time between two coordinates.
            compute_route(source_coord, target_coord):
                Compute the route and travel time between source and target coordinates.
        """
        self.hdf5_path = hdf5_path
        self.travel_times_dict = {}
        self.origins_gdf = origins_gdf
        self.points = np.array(self.origins_gdf[['geometry']].apply(lambda x: [x[0].x, x[0].y], axis=1).tolist())
        self.tree = KDTree(self.points)
        self.ids = self.origins_gdf['id'].to_numpy()
        self.preprocess_data()


    def preprocess_data(self):
        """Preprocess the data from the HDF5 file and store it in a dictionary for fast lookups."""
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:


            # Proceed with the existing logic if datasets are present
            dataset_size = hdf5_file['from_id'].shape[0]
            batch_size = 10000  # Adjust based on your system's memory capacity

            for i in range(0, dataset_size, batch_size):
                from_ids = hdf5_file['from_id'][i:i+batch_size]
                to_ids = hdf5_file['to_id'][i:i+batch_size]
                travel_times = hdf5_file['travel_time'][i:i+batch_size]
                
                for from_id, to_id, travel_time in zip(from_ids, to_ids, travel_times):
                    self.travel_times_dict[(from_id, to_id)] = travel_time
    
    def get_closest_id_tree(self, lat, lon):
        dist, closest_idx = self.tree.query([[lon, lat]], k=1)  # Ensure k=1 for clarity
        # Since closest_idx is a 1D array with a single element, access it directly
        closest_id = self.ids[closest_idx][0]  # Access the first element since closest_idx is 1D
        return closest_id
    
    def query_travel_time(self,o,d):
        # Get closest id
        origin_closest_id = self.get_closest_id_tree(o.y, o.x)
        # Get closest id
        destination_closest_id = self.get_closest_id_tree(d.y, d.x)
        # Query the travel time for a given origin and destination ID pair.
        travel_time = self.travel_times_dict.get((origin_closest_id, destination_closest_id), None)
        return travel_time

    
    def compute_route(self, source_coord, target_coord):
        linestring, travel_time = None, None  # Default values in case of early return

        # Correctly check if any of the coordinates is None
        if source_coord is None or target_coord is None:
            #logger.info("One of the coordinates is None. Returning None for linestring and travel time.")
            return linestring, travel_time

        # Check if source and target coordinates are the same
        if source_coord == target_coord:
            dummy_target = Point(target_coord.x + 0.0001, target_coord.y + 0.0001)
            linestring = LineString([source_coord, dummy_target])
            travel_time = 0
            #logger.info("Source and target coordinates are the same. Returning a point for linestring and 0 for travel time.")
            return linestring, travel_time

        # Main route computation block
        try:
            # Return a straight line between the source and target coordinates
            linestring = LineString([source_coord, target_coord])
            # Complete the linestring if necessary (this function should handle its own errors or be wrapped in try-except if it can raise exceptions)
            linestring = complete_linestring(linestring, source_coord, target_coord)
            # Query travel time (this function should also handle its own errors or be wrapped in try-except)
            travel_time = self.query_travel_time(source_coord, target_coord)

        except Exception as e:
            logger.error(f"Error computing route: {e}")
            # Optionally, set linestring and travel_time to None or a default value here

        return linestring, travel_time



class NetworkRoutingComputer:
    """
    A class to compute network routes based on a specified mode of transportation.

    This class provides methods to find the closest source and target nodes, compute routes, and manage the graph structure for routing.

    Attributes:
        mode (str): The mode of transportation ('walk', 'bike', 'drive').
        G_ig (igraph.Graph): The graph representing the transportation network.
        v_coordinates (list): List of vertex coordinates in the graph.
        A (ndarray): Array of vertex coordinates.
        tree (cKDTree): KDTree for spatial indexing of vertex coordinates.
        names (list): List of vertex names.
        weights (list): List of edge weights in the graph.
        route_cache (dict): Cache to store computed routes.
        speed_factor (int): Factor to adjust travel speed based on the mode of transportation.

    Methods:
        get_closest_source_target(source_coord, target_coord):
            Find the closest source and target nodes in the graph for the given coordinates.
        path_to_linestring(path_indices, G_ig, s, t):
            Convert a path in the graph to a LineString.
        compute_route(source_coord, target_coord, mode='walk'):
            Compute the route and travel time between source and target coordinates.
    """
    def __init__(self,mode,speed_factor = 4):
        self.mode = mode
        self.G_ig = io.fetch_igraph(mode)
        self.v_coordinates = get_vertex_coords(self.G_ig)
        self.A = np.array(self.v_coordinates)
        self.tree = cKDTree(self.A)
        self.names = [f"({x},{y})" for x, y in self.v_coordinates]
        self.G_ig.vs["name"] = self.names
        if mode in ['active','walk','bike']:
            self.edge_length = get_edge_lengths(self.G_ig)
            self.weights = self.edge_length
        elif mode == 'drive':
            self.edge_travel_time = get_edge_travel_time(self.G_ig)
            self.weights = self.edge_travel_time
        else:
            raise ValueError(f"Mode must be one of 'walk', 'bike' or 'drive', got {mode} instead.")
        self.G_ig.es["weight"] = self.weights
        self.route_cache = {}
        self.speed_factor = speed_factor


    
    def get_closest_source_target(self,source_coord, target_coord):
        # Convert the POINT object to a list of coordinates for the source.
        source = [source_coord.x, source_coord.y]
        # Query the KDTree to find the index of the closest node to the source.
        _, source_index = self.tree.query([source], k=1)
        
        
        target = [target_coord.x, target_coord.y]
        # Query the KDTree to find the indices of the closest nodes to the targets.
        _, target_index = self.tree.query([target], k=1)

        # Retrieve the closest source and target nodes from the graph using indices
        closest_source = self.G_ig.vs[source_index[0]]["name"]
        closest_target = self.G_ig.vs[target_index[0]]["name"]

        return closest_source, closest_target
    
    def path_to_linestring(self,path_indices, G_ig,s,t):
        # End the script if the path is empty or has only one point.
        if not path_indices :
            raise ValueError("path_to_linestring: The routing has returned an empty path or a single point, which is not sufficient for creating a LineString")

        # Extract the coordinates of the path
        vertices = G_ig.vs[path_indices]
        x = vertices["x"]
        y = vertices["y"]
        
        if len(x) < 2:
            # Check if source and destination are the same
            if s == t:
                return Point(x[0], y[0])
            else:
                raise ValueError(f"path_to_linestring: The routing has returned a path with only one point. {s,t}")
        # Create the LineString for the path
        line = LineString(zip(x, y))

        return line
    

    def compute_route(self, source_coord, target_coord, mode='walk'):
        linestring, travel_time = None, None  # Default values in case of early return

        # Check if any of the coordinates is None
        if source_coord is None or target_coord is None:
            #logger.info(f"One of the coordinates is None. Returning None for linestring and travel time. Got {source_coord} and {target_coord} instead.")
            return linestring, travel_time

        # Check if source and target coordinates are the same
        if source_coord == target_coord:
            # Create a linestring with 10m length and 0 travel time
            dummy_target = Point(target_coord.x + 0.0001, target_coord.y + 0.0001)
            linestring = LineString([source_coord, dummy_target])
            travel_time = 0
            #logger.info(f"Source and target coordinates are the same. Returning a point for linestring and 0 for travel time. Got {source_coord} and {target_coord} instead.")
            return linestring, travel_time

        # Check if source and target are of type Point
        if not isinstance(source_coord, Point) or not isinstance(target_coord, Point):
            #logger.info(f"Source and target coordinates must be of type Point. Returning None for linestring and travel time. Got {source_coord} and {target_coord} instead.")
            return linestring, travel_time

        # Main route computation block
        try:
            closest_source, closest_target = self.get_closest_source_target(source_coord, target_coord)
            path_indices = self.G_ig.get_shortest_path(closest_source, closest_target, weights="weight", mode="out", output="vpath")
            linestring = self.path_to_linestring(path_indices, self.G_ig, closest_source, closest_target)
            linestring = complete_linestring(linestring, source_coord, target_coord)
            distance = linestring.length

            travel_time = ((distance/1000)/(self.speed_factor))*60
            #travel_time = speed_factor * distance / 1000


        except Exception as e:
            logger.error(f"Error computing route: {e} between {source_coord} and {target_coord}")
            # Optionally, set linestring and travel_time to None or a default value here

        return linestring, travel_time

def complete_linestring(linestring,source_coord,target_coord):
    # Sometimes the routing algorithm returns a path that does not snap to the source and target coordinates.
    # This function completes the route by adding the source and target coordinates to the beginning and end of the route.

    # Get the coordinates of the first and last points in the linestring
    first_point = linestring.coords[0]
    last_point = linestring.coords[-1]

    # Check if the first point is the same as the source coordinate
    if first_point != (source_coord.x, source_coord.y):
        # If not, add the source coordinate to the beginning of the linestring
        linestring = LineString([source_coord, *linestring.coords])
    # Check if the last point is the same as the target coordinate
    if last_point != (target_coord.x, target_coord.y):
        # If not, add the target coordinate to the end of the linestring
        linestring = LineString([*linestring.coords, target_coord])

    return linestring