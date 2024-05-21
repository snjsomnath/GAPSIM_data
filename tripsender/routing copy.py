
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

from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# Constants
COEF_LENGTH = 0.5
COEF_NATURE = 0.5
COEF_SLOPE = 0.5

from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)
def get_coords(geom):
    # If the geometry is a simple LineString, wrap it in a list for consistent processing
    geoms = [geom] if geom.geom_type == "LineString" else geom.geoms
    
    # Use a list comprehension to extract start and end coordinates for each LineString
    return [(line.coords[0], line.coords[-1]) for line in geoms]

def gdf_to_networkx(gdf):
    # Create a new empty undirected graph
    logger.info("Creating empty networkx graph...")
    G = nx.Graph()

    # Apply the get_coords function to every row of the GeoDataFrame to get the start and end coordinates
    # of each road segment
    coords_list = gdf['geom'].apply(get_coords)
    
    # For each set of start and end coordinates, add an edge to the graph
    logger.info("Adding edges to graph...")
    for coords in coords_list:
        for start, end in coords:
            G.add_edge(start, end)

    # Return the resulting graph
    logger.info("Returning networkx graph...")
    return G

def networkx_to_igraph(G_nx):

    # Generate a mapping between the node objects of the NetworkX graph and integer indices for iGraph.
    # This is done by enumerating the nodes of G_nx. The resultant dictionary will have node objects as keys
    # and their respective integer indices as values.
    logger.info("Converting networkx graph to igraph...")
    G_ig = ig.Graph.from_networkx(G_nx)
    return G_ig

def compute_length(geom):
    if geom.geom_type == "MultiLineString":
        return sum(line.length for line in geom.geoms)
    return geom.length

def get_edge_length(G_ig):
    logger.info("Computing lengths of geometries...")
    edge_lengths = [edge['length'] for edge in G_ig.es]

    return edge_lengths

def get_edge_travel_time(G_ig):
    logger.info("Computing lengths of geometries...")
    travel_time = [edge['travel_time'] for edge in G_ig.es]

    return travel_time

def find_closest_source_target(tree, G_ig, source, targets):
    #logger.info("Debug: Finding closest source and target nodes...")

    # Ensure the source is a single coordinate tuple, not a list
    if isinstance(source, list) and len(source) == 1:
        source = source[0]

    # Check if source is already a tuple of coordinates, otherwise extract from source.coords
    source_coord = source if isinstance(source, tuple) else source.coords[0]
    source_coords = np.array([source_coord])  # Convert to 2D array for KDTree

    # Check if the targets are already a list of coordinate tuples, otherwise extract from target.coords
    target_coords = [t if isinstance(t, tuple) else t.coords[0] for t in targets]
    target_coords = np.array(target_coords)  # Ensure this is a 2D array for KDTree

    # Query the KDTree to find the indices of the closest nodes.
    _, source_index = tree.query(source_coords)
    _, target_indices = tree.query(target_coords)

    # Use indices to retrieve the closest nodes from the graph
    closest_source = G_ig.vs[source_index]["name"] if source_index.size > 0 else None
    closest_targets = [G_ig.vs[index]["name"] for index in target_indices]

    return closest_source, closest_targets





def read_and_clip_raster(dem_raster_path, convex_hull):
    """
    Reads and clips a digital elevation model (DEM) raster based on a convex hull boundary.

    This function is designed to extract a subset of a raster file, minimizing the memory usage by reading only the 
    portion within a given convex hull boundary. It uses rasterio's windowed reading functionality, which is efficient 
    for large raster files, allowing for operations on just the required subset without loading the entire raster into memory.

    Workflow:
    1. Open the DEM raster file using rasterio.
    2. Create a bounding box (bbox) based on the convex hull's boundaries.
    3. Calculate the window of interest from the bounding box and the raster's affine transform.
    4. Read the raster data for the window, ensuring that the operation is masked to handle no data values.
    5. Obtain the affine transform of the windowed read for geographic reference.
    6. Copy and update the raster's metadata to reflect the new clipped size and transform.

    Parameters:
    - dem_raster_path (str): File path to the DEM raster that needs to be read and clipped.
    - convex_hull (Polygon): A shapely Polygon representing the convex hull that defines the area to clip the raster.

    Returns:
    Tuple[numpy.ndarray, affine.Affine, dict]: A tuple containing:
    - The clipped raster data as a 2D numpy array.
    - The affine transform for the clipped raster data.
    - The metadata dictionary for the clipped raster data, updated with the new dimensions and transform.

    Note:
    - It is assumed that the convex hull and the raster data are in the same coordinate reference system (CRS).
    - The returned affine transform and metadata can be used for writing the clipped raster back to a new file, if needed.
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
    Processes land use vector data by filtering, reprojecting, clipping, and dissolving based on specified attributes.

    The function narrows down the land use data to relevant categories, reprojects the data to match the coordinate 
    reference system (CRS) of a DEM raster, and clips the data to the extent of a convex hull. After clipping, 
    it dissolves the geometries by a 'detaljtyp' attribute to aggregate geometries with the same land use type. 

    Workflow:
    1. Filter the GeoDataFrame to include only the rows of interest based on 'detaljtyp'.
    2. Reproject the GeoDataFrame to the CRS of the DEM raster if they differ.
    3. Clip the land use data to the area within the convex hull's bounding box.
    4. Dissolve the clipped geometries by the 'detaljtyp' attribute to combine geometries of the same type.

    Parameters:
    - landuse_vector (geopandas.GeoDataFrame): A GeoDataFrame containing land use vector data with a 'detaljtyp' attribute.
    - convex_hull (shapely.geometry.Polygon): A shapely Polygon representing the convex hull for clipping the land use data.
    - dem_raster_crs (str, optional): The EPSG code for the CRS of the DEM raster. Defaults to 'EPSG:3006'.

    Returns:
    - clipped_vector_landuse (geopandas.GeoDataFrame): The processed GeoDataFrame containing only relevant land use data,
      reprojected and clipped to the convex hull, with geometries dissolved by land use type.

    Note:
    - The function assumes that the 'detaljtyp' column exists and contains categorical data indicating the type of land use.
    - It is critical to filter before reprojecting or clipping to avoid unnecessary computations on irrelevant data.
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
    Generates a gradient effect in a raster format based on distance from rasterized natural features.

    The function first rasterizes given vector data, then calculates the distance from each non-feature pixel to the nearest feature pixel.
    It scales the distance to a 0-1 range and inverses the values to create a gradient that decays from natural features.

    Optimizations include:
    - Simplification of geometries before rasterization to speed up the process.
    - In-place operations for distance scaling and clipping to reduce memory usage.
    - Avoidance of unnecessary array creation.

    Parameters:
    - clipped_vector_landuse (GeoDataFrame): GeoDataFrame with vector data of landuse to rasterize.
    - out_shape (tuple of int): The shape (height, width) of the output raster.
    - out_transform (Affine): The affine transformation associated with the output raster.
    - max_distance (float): The maximum distance for gradient influence from natural features.

    Returns:
    - gradient (ndarray): A NumPy array representing the gradient effect raster.
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
    Plot the gradient effect of natural features on a 2D array with geographic coordinates.

    This function uses matplotlib to visualize a 2D gradient array. It displays the array as an image
    with a specific colormap and includes a colorbar for reference. The title of the plot and the figure
    size can be customized. Geographic coordinates are derived from a provided affine transform.

    Parameters:
    - gradient (numpy.ndarray): The 2D array representing the normalized gradient values.
    - transform (Affine): An affine transform relating pixel coordinates to the real-world coordinates.
    - title (str, optional): The title of the plot. Defaults to 'Gradient Effect of Natural Features'.
    - cmap (str, optional): The colormap used for plotting. Defaults to 'viridis'.
    - figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
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
    # Convert real-world coordinates to pixel coordinates using the affine transform
    col, row = ~transform * (x, y)  # Use the inverse transform to map (x, y) to (row, col)
    return int(row), int(col)

def raster_value_at_line_ends(raster, transform, start_coord, end_coord, value_type='average'): #TODO Not working, too many nulls
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
    list_min = min(input_list)
    list_max = max(input_list)
    
    return [(x-list_min)/(list_max-list_min) for x in input_list]


def compute_edge_weights(coords, edge_length, dem_raster_path , landuse_vector, convex_hull):

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


def path_to_multilinestring(path_indices, A):

    # End the script if the path is empty.
    if not path_indices:
        raise ValueError("path_to_multilinestring: The routing has returned an empty path, please check the script.")

    # Extract the coordinates of nodes in the path from the array 'A' using the provided indices.
    path_coords = A[path_indices]

    # For each consecutive pair of coordinates in the path, create a LineString segment.
    line_segments = [LineString(pair) for pair in zip(path_coords[:-1], path_coords[1:])]

    # Combine these segments to construct a MultiLineString that represents the entire path.
    return MultiLineString(line_segments)





def process_building(building, tree, A, G_ig, route_cache, mode):

    source_coord = building.coord  # Assuming this is a single coordinate pair (x, y)
    preferred_locations = building.preferred_locations
    
    for category, locations in preferred_locations.__dict__.items():

        if locations and isinstance(locations, (list, tuple)):
            targets_coords = [loc.location_coordinates for loc in locations]  # List of coordinate pairs

            # Call find_closest_source_target with correct formats
            closest_source, closest_targets = find_closest_source_target(tree, G_ig, [source_coord], targets_coords)

            for idx, location in enumerate(locations):
                s = closest_source[0]
                t = closest_targets[idx]

                # Use tuples directly as keys
                route_key = (s, t)
                if route_key in route_cache:
                    route = route_cache[route_key]
                else:
                    # Assuming your graph G_ig uses coordinate tuples as vertex identifiers:
                    path_indices = G_ig.get_shortest_path(s, t, weights="weight", mode="out", output="vpath")
                    route = path_to_multilinestring(path_indices, A)
                    if route is None:
                        # Log an error if no route is found
                        logger.error(f"No route found between {s} and {t}")
                    else:
                        # Add computed route to cache
                        route_cache[route_key] = route

                if mode == 'active':
                    location.route_active = route
                elif mode == 'car':
                    location.route_car = route
                elif mode == 'public':
                    location.route_public = route

    # Buildings are updated on the fly generally
    # However, in the case of a batch, we need to return the updated building
    # so they can be processed in multiple processes and then replace the Building.instances
    return building



def process_building_batch(building_batch, tree, A, G_ig, route_cache,mode):

    logger.info("Processing building batch...")
    for building in building_batch:
        process_building(building, tree, A, G_ig, route_cache, mode)
    
    # For parallel processing, return the updated building batch
    return building_batch

def get_edge_coords(G_ig):
    # Assuming G_ig is your igraph graph object and it has 'x' and 'y' attributes for vertices

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

def compute_routes_for_all_buildings(G_nx,dem_raster_path, landuse_vector, convex_hull, mode = 'active', parallel=False):

    # Step 1 & 2: Convert road networks to graphs
    logger.info("Converting road network to graphs...")
    #G_nx = gdf_to_networkx(street_gdf)
    G_ig = networkx_to_igraph(G_nx)
    
    # Step 3: Set up an array of nodes
    logger.info("Building KDTree for spatial indexing...")
    v_coordinates = [(vertex['x'], vertex['y']) for vertex in G_ig.vs]
    e_coordinates = get_edge_coords(G_ig)
    A = np.array(v_coordinates)
    #A = np.array(list(G_nx))
    # Build KDTree once
    tree = cKDTree(A)

    # Step 4: Compute edge lengths (weights)
    edge_length = get_edge_length(G_ig)
    weights = get_edge_travel_time(G_ig)
    if mode == 'active':
        weights = compute_edge_weights(e_coordinates,edge_length, dem_raster_path, landuse_vector, convex_hull)
    logger.info(f"Number of edges: {len(weights)}")
    # Step 5: Assign these weights to edges in the iGraph
    logger.info("Assigning edge lengths to iGraph...")
    names = [f"({vertex['x']},{vertex['y']})" for vertex in G_ig.vs]
    G_ig.vs["name"] = names

    G_ig.es["weight"] = weights #TODO add weights for slope and distance to green
    
    # Create a cache to store computed routes
    route_cache = {}

    # Step 6: Parallel processing for each building
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