# -----------------------------------------------------
# activity.py
# -----------------------------------------------------
# Description: Contains the Activity and ActivitySequence classes used to represent an individual's activity sequence in the simulation.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.activity"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

from datetime import datetime, timedelta, time
from shapely.geometry import Point, MultiLineString
from typing import TYPE_CHECKING, Optional, List, Union
from tripsender import sampler, nhts
import pandas as pd
import plotly.express as px
import logging
import random
import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go



from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

DURATION_DIST_PATH = "models/DURATION_DIST.JSON"
with open (DURATION_DIST_PATH, "r") as f:
    DURATION_DIST = json.load(f)

class Activity:
    """
    Represents an activity performed by an individual, based on a sampled and matched activity sequence from the National Household Travel Survey (NHTS).

    An activity object consists of:
        - Activity purpose
        - Start time
        - Duration
        - End time
        - Mode of travel

    Attributes:
        start_time (datetime.time): The starting time of the activity.
        duration_minutes (int): The duration of the activity in minutes.
        duration_timedelta (timedelta): The duration of the activity as a timedelta object.
        end_time (datetime.time): The ending time of the activity.
        purpose (str): The purpose of the activity (e.g., work, school, shopping).
        mode (str, optional): The mode of transportation used for the activity (e.g., walking, driving).
        destination (str, optional): The destination of the activity.
        destination_coordinates (tuple, optional): The coordinates of the destination.
        origin (str, optional): The origin of the activity.
        origin_coordinates (tuple, optional): The coordinates of the origin.
        calculated_duration (timedelta, optional): The calculated duration of the trip if different from the provided duration.
        route (list, optional): The route taken for the activity, if applicable.
    """
    def __init__(self, start_time, duration_minutes, purpose, mode=None):
        """
        Initializes an Activity object with the given parameters.

        Args:
            start_time (str or datetime): The start time of the activity. Can be a string or a datetime object.
            duration_minutes (int): The duration of the activity in minutes.
            purpose (str): The purpose of the activity.
            mode (str, optional): The mode of transportation. Defaults to None.
        """
        # Use _parse_time_input for consistent time parsing
        parsed_datetime = self._parse_time_input(start_time)
        self.start_time = parsed_datetime.time() if parsed_datetime else None
        self.duration_minutes = duration_minutes
        self.duration_timedelta = self.duration()
        self.end_time = (datetime.combine(datetime.today(), self.start_time) + timedelta(minutes=duration_minutes)).time()
        self.purpose = purpose
        self.mode = mode
        self.destination = None
        self.destination_coordinates = None
        self.origin = None
        self.origin_coordinates = None
        self.calculated_duration = None
        self.route = None




    def __repr__(self):
        return f"{self.start_time.strftime('%H:%M')} - {self.purpose} ({self.duration_minutes} mins)"

    def _parse_time_input(self, time_input):
        """ Parses time input into a datetime object.
        Args:
            time_input (str or datetime.time): The time input to be parsed.
        Returns:
            datetime.datetime: The parsed datetime object.
        
        example:
            _parse_time_input("1200") -> datetime.datetime(1900, 1, 1, 12, 0)
            _parse_time_input("12:00") -> datetime.datetime(1900, 1, 1, 12, 0)
            _parse_time_input(datetime.time(12, 0)) -> datetime.datetime(1900, 1, 1, 12, 0)
        """
        
        if isinstance(time_input, str):
            if ':' in time_input:
                return datetime.strptime(time_input, '%H:%M')
            else:
                return datetime.strptime(time_input, '%H%M')
        elif isinstance(time_input, time):
            return datetime.combine(datetime.today(), time_input)
        elif isinstance(time_input, datetime):
            return time_input
        else:
            logger.error("Time input must be in the format HH:MM or HHMM or datetime.time or datetime.datetime")
            return None

    def __str__(self):
        return f"Start Time: {self.start_time}, End Time: {self.end_time}, Purpose: {self.purpose}, Mode: {self.mode}"

    def duration(self):
        return timedelta(minutes=self.duration_minutes)


class ActivitySequence:
    instances: List['ActivitySequence'] = []
    samples: List['ActivitySequence'] = []
    def __init__(self):
        self.person = None
        self.activities = []
        self.disruptions = 0 # Number of disruptions in the sequence
        
    @classmethod
    def return_person_df(cls):
        list_of_person_dicts = []
        for activity_sequence in cls.samples:
            list_of_person_dicts.append(activity_sequence.sampled_person)
        
        df = pd.DataFrame(list_of_person_dicts)

        # Create a column called ActivitySequence that contains the ActivitySequence instance
        df['ActivitySequence'] = cls.samples
        return df
    
    @classmethod
    def clear_instances(cls):
        cls.instances = []

    @classmethod
    def clear_samples(cls):
        cls.samples = []
    
    def __repr__(self):
        return "\n".join(repr(activity) for activity in self.activities)

    def total_duration(self):
        return sum([activity.duration() for activity in self.activities], datetime.timedelta())

    def __str__(self):
        return '\n'.join(str(activity) for activity in self.activities)

    def return_gdf(self):
        gdf_data = []
        activity_sequence = self
        for activity in activity_sequence.activities:
            if activity.purpose == "Transit":
                next_activity = activity_sequence.activities[activity_sequence.activities.index(activity) + 1]
                # The below check seems deprecated
                # if isinstance(activity.destination, Point):
                if activity.destination != "Travel":
                    gdf_dict = {
                        'geometry': activity.route,
                        'mode': activity.mode,
                        'origin': activity.origin_coordinates,
                        'destination': activity.destination_coordinates,
                        'sampled_duration': activity.duration_minutes,
                        'calculated_duration': activity.calculated_duration,
                        'start_time': activity.start_time,
                        'purpose': next_activity.purpose
                    }
                    # Append dictionary to list
                    gdf_data.append(gdf_dict)
                elif activity.destination == "Travel":
                    # If the destination says "Travel", then it's a travel activity and we don't have a destination
                    # So we just plot a point at the origin of the travel activity
                    gdf_dict = {
                        'geometry': activity.origin_coordinates,
                        'mode': activity.mode,
                        'origin': activity.origin_coordinates,
                        'destination': activity.destination_coordinates,
                        'sampled_duration': activity.duration_minutes,
                        'calculated_duration': activity.calculated_duration,
                        'start_time': activity.start_time,
                        'purpose': activity.purpose
                    }
                    # Append dictionary to list
                    gdf_data.append(gdf_dict)
                else:
                    logger.error("Destination is not a Point or 'Travel'")
        # Create GeoDataFrame from collected data
        gdf = gpd.GeoDataFrame(gdf_data)
        return gdf

    def plot(self, plot_type='2d'):
        gdf = self.return_gdf()
        home = gdf.iloc[0]['origin']

        # Identify unique modes of transport
        unique_modes = set(row['mode'] for index, row in gdf.iterrows())

        # Create a color map for the modes
        colors = plt.cm.get_cmap('viridis', len(unique_modes))
        mode_colors = {mode: colors(i) for i, mode in enumerate(unique_modes)}

        # Check if gdf is not empty before proceeding
        if not gdf.empty:
            # Set geometry column explicitly in case it's not automatically recognized
            gdf = gdf.set_geometry('geometry')
            
            # Set CRS to EPSG:3006
            gdf.crs = "epsg:3006"
            
            if plot_type == '2d':
                # 2D Plot
                ax = gdf.plot(column='mode', legend=True, figsize=(10, 10))
                ax.set_title("Activity Routes by Mode")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                # Add home location
                ax.scatter(home.x, home.y, color='red', marker='*', s=100, label='Home')
            

            elif plot_type == 'spacetimecube_static':
                # Static Space-Time Cube
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                # Convert start_time to total seconds from the start of the day and then to hours
                gdf['time_hours'] = gdf['start_time'].apply(lambda x: pd.Timedelta(str(x)).total_seconds() / 3600)

                for index, row in gdf.iterrows():
                    x, y = zip(*[(point[0], point[1]) for point in list(row['geometry'].coords)])
                    z = [row['time_hours']] * len(x)
                    mode_color = mode_colors[row['mode']]
                    ax.plot(x, y, z, linestyle='-', marker='', color=mode_color)  # Use color mapped to mode

                    # Place a text label near the start of the activity
                    label_pos = 0
                    ax.text(x[label_pos], y[label_pos], z[label_pos], f"{row['purpose']}", color='black', fontsize=8, ha='right')

                    if index < len(gdf) - 1:
                        next_row = gdf.iloc[index + 1]
                        ax.plot([x[-1], x[-1]], [y[-1], y[-1]], [row['time_hours'], next_row['time_hours']], 'k:', linewidth=1)


                # After iterating through all rows, get the last activity's coordinates and time
                last_activity = gdf.iloc[-1]
                last_coords = list(last_activity['geometry'].coords)  # Convert coordinates to a list
                last_x, last_y = last_coords[-1]  # Take the last point of the last activity
                last_z = last_activity['time_hours']

                # Connect the last activity to home with a vertical dotted line back to the start of the day
                ax.plot([last_x, home.x], [last_y, home.y], [last_z, 0], 'k:', linewidth=1, )

                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('Time (Hours from start of day)')
                ax.set_zticks(range(0, 24, 3))  # Set z-axis to display each hour
                ax.set_zticklabels([f"{i}h" for i in range(0, 24, 3)])  # Label each tick with the hour


                # Get the current xlim and ylim
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()

                # Calculate the range of the x and y axis
                xrange = xmax - xmin
                yrange = ymax - ymin

                # Set the x and y ticks to display relative distances
                # Define the number of ticks you want
                num_ticks = 5  # Example: 5 ticks

                # Set x and y ticks
                ax.set_xticks([xmin + i * (xrange / (num_ticks - 1)) for i in range(num_ticks)])
                ax.set_yticks([ymin + i * (yrange / (num_ticks - 1)) for i in range(num_ticks)])

                # Set x and y tick labels
                ax.set_xticklabels([f"{int(i * (xrange / (num_ticks - 1)))}m" for i in range(num_ticks)])
                ax.set_yticklabels([f"{int(i * (yrange / (num_ticks - 1)))}m" for i in range(num_ticks)])



                # Reduce the grid lineweight
                ax.xaxis._axinfo['grid'].update(linewidth=0.1)
                ax.yaxis._axinfo['grid'].update(linewidth=0.1)
                ax.zaxis._axinfo['grid'].update(linewidth=0.1)

                # Add home location at the start of the day with a green marker
                ax.scatter(home.x, home.y, 0, color='green', marker="v", s=100, label='Start Home')

                # Add home location at the end of the day with a red marker
                ax.scatter(home.x, home.y, last_z, color='red', marker="v", s=100, label='End Home')

                # Reduce the thickness of the text
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.zaxis.label.set_size(8)

                # Ticks
                ax.xaxis.set_tick_params(labelsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.zaxis.set_tick_params(labelsize=8)


                # Prepare legend entries for modes of transport
                legend_entries = [Line2D([0], [0], color=mode_colors[mode], lw=4, label=mode) for mode in unique_modes]

                # Add home markers to the legend
                legend_entries.append(Line2D([0], [0], marker='v', color='green', label='Start Home', markersize=10, linestyle='None'))
                legend_entries.append(Line2D([0], [0], marker='v', color='red', label='End Home', markersize=10, linestyle='None'))

                # Set the legend with the mode entries
                ax.legend(handles=legend_entries)

                plt.show()


            elif plot_type == 'spacetimecube_interactive':
                # Interactive Space-Time Cube
                traces = []
                legend_added = {}  # To track which modes have been added to the legend

                # Color map for modes of transport
                unique_modes = list(set(row['mode'] for index, row in gdf.iterrows()))
                colors = plt.cm.get_cmap('viridis', len(unique_modes))
                mode_colors = {mode: f'rgb{colors(i)[:3]}' for i, mode in enumerate(unique_modes)}

                # Determine the range for x and y axis based on the geometry
                all_coords = [coord for row in gdf['geometry'] for coord in list(row.coords)]
                all_x, all_y = zip(*all_coords)
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)

                # Adjust the range if you want to start from 0
                xrange = xmax - xmin
                yrange = ymax - ymin

                # Plot each activity segment with transitions
                for index, row in gdf.iterrows():
                    x, y = zip(*[(coord[0] - xmin, coord[1] - ymin) for coord in list(row['geometry'].coords)])  # Adjusted coordinates
                    z = [pd.Timedelta(str(row['start_time'])).total_seconds() / 3600] * len(x)  # Time in hours
                    mode_color = mode_colors[row['mode']]
                    
                    # Check if the mode is already added to legend
                    show_legend = row['mode'] not in legend_added
                    legend_added[row['mode']] = True
                    
                    # Activity trace
                    trace = go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        name=row['mode'],
                        line=dict(color=mode_color, width=4),
                        hoverinfo='text',
                        text=f"Mode: {row['mode']}, Duration: {row['sampled_duration']} mins, Purpose: {row['purpose']}",
                        showlegend=show_legend
                    )
                    traces.append(trace)
                    
                    # Transition to next activity or home with a dotted line
                    if index < len(gdf) - 1:
                        next_row = gdf.iloc[index + 1]
                        next_x, next_y = next_row['origin'].x - xmin, next_row['origin'].y - ymin  # Adjusted coordinates
                        next_z = pd.Timedelta(str(next_row['start_time'])).total_seconds() / 3600
                        traces.append(go.Scatter3d(
                            x=[x[-1], next_x], y=[y[-1], next_y], z=[z[-1], next_z],
                            mode='lines',
                            line=dict(color='black', width=2, dash='dot'),
                            showlegend=False  # Hide these transitions from legend
                        ))

                # Calculate home coordinates adjusted for the plot
                home_x, home_y = home.x - xmin, home.y - ymin  # Adjusted coordinates
                home_start_z = 0  # Start of the day in hours
                home_end_z = pd.Timedelta(str(gdf.iloc[-1]['start_time'])).total_seconds() / 3600  # End of the last activity in hours

                # Home location at the start of the day (green marker)
                traces.append(go.Scatter3d(
                    x=[home_x], y=[home_y], z=[home_start_z],
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='Start Home',
                    hoverinfo='text',
                    text='Start Home'
                ))
                # Transition from home to the first activity (dotted line)
                if len(gdf) > 0:
                    first_activity = gdf.iloc[0]
                    first_coords = list(first_activity['geometry'].coords)
                    first_x, first_y = first_coords[0][0] - xmin, first_coords[0][1] - ymin  # Adjusted coordinates of the first point of the first activity
                    first_z = pd.Timedelta(str(first_activity['start_time'])).total_seconds() / 3600  # Start time of the first activity in hours

                    # Transition from home to the first activity (dotted line)
                    traces.append(go.Scatter3d(
                        x=[home_x, first_x], y=[home_y, first_y], z=[home_start_z, first_z],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dot'),
                        showlegend=False  # This transition should not appear in the legend
                    ))
                # Transition from the last activity back to home (dotted line)
                if len(gdf) > 0:
                    last_activity = gdf.iloc[-1]
                    last_coords = list(last_activity['geometry'].coords)
                    last_x, last_y = last_coords[-1][0] - xmin, last_coords[-1][1] - ymin  # Adjusted coordinates
                    traces.append(go.Scatter3d(
                        x=[last_x, home_x], y=[last_y, home_y], z=[home_end_z, home_end_z],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dot'),
                        showlegend=False
                    ))

                # Home location at the end of the day (red marker)
                traces.append(go.Scatter3d(
                    x=[home_x], y=[home_y], z=[home_end_z],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='End Home',
                    hoverinfo='text',
                    text='End Home'
                ))



                # Layout configuration, including x and y axis ticks and labels
                layout = go.Layout(
                    title="Space-Time Cube Visualization",
                    scene=dict(
                        # Axis configurations (as in the previous code)
                    ),
                    margin=dict(r=0, l=0, b=0, t=50)
                )

                # Create figure with traces and layout
                fig = go.Figure(data=traces, layout=layout)

                # Display the interactive plot
                fig.show()
        else:
            print("No routes available for plotting.")

        return gdf

    def from_nhts(self, df):
        """
        Create an ActivitySequence object from a sampled and matched activity sequence from the National Household Travel Survey (NHTS).
        
        Attributes:
            df (pandas.DataFrame): A DataFrame containing the sampled and matched activity sequence from the NHTS.

        Returns:
            ActivitySequence: An ActivitySequence object representing the activity sequence.
        """
        current_date = datetime.now().date()
        start_of_day = datetime.combine(current_date, time(3, 0))
        end_of_day = start_of_day + timedelta(days=1)

        # Some df checks to make sure that df makes sense
        # If activity duration is 0 days 00:00:00 remove the row
        #df = df[df['activity_duration_minutes'] != pd.Timedelta(seconds=0)]
        # Reset index, drop means that the old index is not added as a column

        # If the last trip of the day is not Home, then get the duration of the last activity based on sampled activity duration
        # And add a travel activity to get home that starts after the sampled duration of the last activity
        last_row = df.iloc[-1]
        if last_row['purpose'] != 'Home':
            df = add_travel_home_activity(df)

        #df = df.reset_index(drop=False)
        #logger.info('Number of activities in df: {}'.format(len(df)))
        #logger.info(" Starting day from Home...")
        # Add a column called is_worker to the dataframe
        #df['is_worker'] = False
        #.loc[row_indexer,col_indexer] = value instead
        df.loc[df.index, 'is_worker'] = False
        is_worker = False
        for index, row in df.iterrows():
            #logger.info(" Traveling to activity {}".format(row['purpose']))
            # Travel to activity
            travel_duration = calculate_duration(row['start_time'] - row['travel_duration_minutes'], row['start_time'])

            t_time = row['start_time']
            t_duration = travel_duration
            # What if we call this "Transit" instead of "Travel"?
            t_purpose = 'Transit'
            t_mode = row['mode']
            transit_activity = Activity(t_time, t_duration, t_purpose, t_mode)
            transit_activity.destination = row['purpose']
            self.activities.append(transit_activity)

            # Activity itself
            #logger.info(" Current activity {}".format(row['purpose']))
            #logger.info(" Duration of activity {}: {}".format(row['purpose'], row['activity_duration_minutes']))
            if pd.isna(row['activity_duration_minutes']):
                # If there is no next activity, then the activity duration is the time until the end of the day
                a_duration = 0
            else:
                a_duration = calculate_duration(row['end_time'] - row['activity_duration_minutes'], row['end_time'])
            
            # Only add activity if it has a duration greater than 0 or if it is the last activity of the day
            if a_duration > 0 or index < len(df) - 1 or row["purpose"] == "Pickup/Dropoff child":
            
                a_time = row['end_time']
                a_purpose = row['purpose']
                if a_purpose == 'Work':
                    is_worker = True

                if a_purpose != 'Travel': # Sometimes the activity itself is a travel activity
                    a_mode = None
                else:
                    a_mode = row['mode']
                self.activities.append(Activity(a_time, a_duration, a_purpose, a_mode))

        # Handle initial home activity
        # If the first activity of the day is not at 3:00, then there must be an initial home activity
        if df.iloc[0]['start_time'] > start_of_day:
            
            # Calculate duration of initial home activity - from 3:00 to start of first activity
            h_time = start_of_day
            h_duration = calculate_duration(start_of_day, df.iloc[0]['start_time'])
            h_purpose = 'Home'
            h_mode = None
            self.activities.insert(0, Activity(h_time, h_duration, h_purpose, h_mode))


        # Handle final home activity
        # If the last activity of the day is not at 3:00, then there must be a final home activity

        last_activity = self.activities[-1]
        last_activity_end_time = datetime.combine(current_date, last_activity.start_time)
        
        # Extract minutes from the timedelta object
        last_activity_duration_minutes = last_activity.duration().total_seconds() / 60
        last_activity_end_time += timedelta(minutes=last_activity_duration_minutes)

        if last_activity_end_time < end_of_day:
            fh_time = last_activity_end_time
            fh_duration = calculate_duration(last_activity_end_time, end_of_day)
            fh_purpose = 'Home'
            fh_mode = None
            self.activities.append(Activity(fh_time, fh_duration, fh_purpose, fh_mode))
            #logger.info(" Heading home...")

        # Update the is_worker column
        #df['is_worker'] = is_worker
        # .loc[row_indexer,col_indexer] = value instead
        df.loc[df.index, 'is_worker'] = is_worker
        # Extract person attributes
        self.sampled_person = df.iloc[0][['id', 'sex', 'age_group', 'house_type', 'child_count', 'adult_count', 'household_type', 'car_count','is_worker']].to_dict()

        # Validate ActivitySequence object and add it to the list of instances if valid
        if not self.is_valid():
            #logger.error("ActivitySequence is not valid")
            return False

        # Add ActivitySequence object to the list of instances
        self.samples.append(self)
        
        return self
        


    def is_valid(self):
        """
        Validates the ActivitySequence object.
        Specifically, it checks the following:
            - Start time is before end time for each activity
            - Duration of each activity is positive
            - Activities are in increasing order of start time
            - Activities do not overlap
            - Sum of activity durations is 24 hours
            - There is a "Transit" activity 
            - Mode and purpose are provided for each activity
        Returns:
            bool: True if the ActivitySequence is valid, False otherwise.
        """
        # Check if there are any activities to validate
        if not self.activities:
            return False

        # Check if start time of an activity is before its end time
        for activity in self.activities:
            if activity.start_time >= activity.end_time:
                # Check for the HOME exception
                if activity.purpose == 'Home' and activity.end_time <= time(12, 0):  # Assuming HOME can only go till noon of the next day
                    continue
                elif activity.purpose == "Pickup/Dropoff child":
                    continue
                else:
                    #logger.error(f"Start time is not before end time for activity: {activity}")
                    return False
        
        # Check if any activity has a negative duration
        for activity in self.activities:
            if activity.duration_minutes <= 0 and activity.purpose != "Pickup/Dropoff child":
                #logger.error(f"Negative or zero duration for activity: {activity}")
                return False

        # Check if activities are in increasing order of start time
        for i in range(len(self.activities) - 1):
            if self.activities[i].start_time >= self.activities[i+1].start_time and self.activities[i].purpose != "Pickup/Dropoff child":
                #logger.error(f"Activities are not in increasing order of start time: {self.activities[i]} and {self.activities[i+1]}")
                return False

        # Ensure activities don't overlap
        for i in range(len(self.activities) - 1):
            if self.activities[i].end_time > self.activities[i+1].start_time:
                #logger.error(f"Activities overlap: {self.activities[i]} and {self.activities[i+1]}")
                return False

        # Ensure activities don't exceed 24 hours, except for HOME
        if self.activities[-1].purpose != 'Home' and self.activities[-1].end_time > time(23, 59):
            #logger.error(f"Activity sequence exceeds 24 hours: {self.activities[-1]}")
            return False

        # Check the mode and purpose for each activity
        for activity in self.activities:
            if activity.purpose in ["Travel"]:
                if not activity.mode or not activity.purpose:
                    #logger.error(f"Missing mode or purpose for activity: {activity}")
                    return False
        
        # Check if sum of activity durations is 24 hours
        if sum([activity.duration_minutes for activity in self.activities]) != 1440:
            #logger.error("Sum of activity durations is not 24 hours")
            return False

        # Check if there is a "Transit" activity before every non-"Transit" activity (except for the first activity)
        for i in range(1, len(self.activities)):
            if self.activities[i].purpose != 'Transit' and self.activities[i-1].purpose != 'Transit':
                #logger.error(f"No transit activity before non-transit activity: {self.activities[i]}")
                return False
            
        # Make sure that there are no two consecutive "Transit" activities
        for i in range(1, len(self.activities)):
            if self.activities[i].purpose == 'Transit' and self.activities[i-1].purpose == 'Transit':
                #logger.error(f"Two consecutive transit activities: {self.activities[i]}")
                return False


        return True


class Location:
    """
    Represents a location in the simulation, such as a home, work, school, or other destination.

    Attributes:
        location_type (str): The type of location (e.g., home, work, school).
        location_name (str): The name of the location.
        location_coordinates (Point): The coordinates of the location.
        location_amenity (str, optional): The amenity of the location (e.g., hospital, park).
        route_car (Route, optional): The route to the location by car.
        route_walk (Route, optional): The route to the location by walking.
        route_bike (Route, optional): The route to the location by biking.
    """



    def __init__(self, location_type: str, location_name: str, location_coordinates: Point, location_amenity: str = None):
        self.location_type = location_type
        self.location_name = location_name
        self.location_coordinates = location_coordinates
        self.location_amenity = location_amenity
        # Routes are stored on OD Matrix, therefore the following are not used
        self.route_car : Route = None
        self.route_walk : Route = None
        self.route_bike : Route = None

    def __repr__(self):
        return f"{self.location_type} ({self.location_amenity}) - {self.location_name} @ {self.location_coordinates}"

class Route:
    """
    Represents a route between two locations, such as a home-to-work commute or a trip to a grocery store.

    Attributes:
        route_type (str): The type of route (e.g., car, walk, bike).
        route_path (MultiLineString): The path of the route as a MultiLineString.
        route_speed_kph (int): The speed of travel along the route in kilometers per hour.
        route_distance (float): The distance of the route in kilometers.
        route_travel_time_minutes (int): The travel time along the route in minutes.
    """
    def __init__(self, route_type: str, route_path: MultiLineString, route_speed_kph: int):
        self.route_type = route_type
        self.route_path = route_path
        self.route_speed_kph = route_speed_kph
        self.route_distance = route_path.length
        self.route_travel_time_minutes = int(self.route_distance / self.route_speed_kph * 60)
    
    def __repr__(self):
        return f"{self.route_type} - {self.route_distance} km @ {self.route_speed_kph} kph"

    def plot(self):
        logger.info(f"Plotting not defined")

def calculate_duration(start_time, end_time):
    """A helper function that calculates duration in minutes between two datetime objects.    
    """
    duration = end_time - start_time
    return duration.total_seconds() / 60

def add_travel_home_activity(df):
    """
    Add a travel activity to get home after the last activity of the day.
    """
    # Create a sampler object
    s = sampler.DurationSampler(DURATION_DIST)
    last_row = df.iloc[-1]
    last_purpose = last_row['purpose']
    last_end_time = last_row['end_time']
    last_mode = last_row['mode']

    # Get the longest duration from all activities
    average_duration = df['travel_duration_minutes'].mean()

    #logger.info("Last activity is not Home, adding a travel activity to get home...")
    
    # Try to sample the duration of the last activity from the distribution
    # If the sampled duration is None then try again till a valid duration is sampled
    if last_purpose == "Pickup/Dropoff child":
        duration_minutes_float = 0
    else:
        duration_minutes_float = None
        while duration_minutes_float is None:
            duration_minutes_float = s.sample_duration(last_purpose)
            if duration_minutes_float is None:
                logger.error("Sampled duration is None, trying again...")
    # Convert duration to a timedelta object
    duration_minutes = pd.Timedelta(minutes=duration_minutes_float)
    
    # Calculate the start time of the return to home trip
    return_start_time = last_end_time + duration_minutes
    # Update activity_duration_minutes for the last row
    #df.loc[df.index[-1], 'activity_duration_minutes'] = duration_minutes
    df.at[df.index[-1], 'activity_duration_minutes'] = duration_minutes

    
    new_row = {
                'start_time': return_start_time,
                'purpose': 'Home',
                'mode': last_mode,
                'end_time': return_start_time + average_duration,
                'distance_km': None,
                'activity_sequence': len(df) + 1,
                'travel_duration_minutes': average_duration,
                'activity_duration_minutes': None,  # Since it's the last activity
                'next_travel_start_time': None
            }
    df = df.append(new_row, ignore_index=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)
    return df

# Plotting functions
def _initialize_activity_counts(bins):
    """Initialize activity counts for all activity types.
    The counts are stored in a dictionary with activity types as keys and lists of counts for each bin as values.
    The following activity types are considered: Transit, Travel, Grocery, Shopping, Leisure, Home, Work, Education, Healthcare, Pickup/Dropoff child, and Other.
        
    """
    return {
        "Transit": [0]*bins,
        "Travel": [0]*bins,
        "Grocery": [0]*bins,
        "Shopping": [0]*bins,
        "Leisure": [0]*bins,
        "Home": [0]*bins,
        "Work": [0]*bins,
        "Education": [0]*bins,
        "Healthcare": [0]*bins,
        "Pickup/Dropoff child": [0]*bins,
        "Other": [0]*bins,
    }

def get_bin_index(time, bins):
    """Get the bin index for a given time."""
    return (time.hour * bins) // 24

def _count_activities(activity_sequences, count_transit_destinations=False, bins=24):
    bin_duration = timedelta(hours=24/bins)
    activity_counts = _initialize_activity_counts(bins)

    for sequence in activity_sequences:
        for activity in sequence.activities:
            activity_start = timedelta(hours=activity.start_time.hour, minutes=activity.start_time.minute)
            activity_end = activity_start + activity.duration_timedelta

            current_bin_start = timedelta(hours=int(activity_start.total_seconds() // bin_duration.total_seconds()) * bin_duration.total_seconds() / 3600)
            current_bin_end = current_bin_start + bin_duration

            while current_bin_start < activity_end:
                bin_index = int(current_bin_start.total_seconds() // bin_duration.total_seconds())
                overlap_duration = min(current_bin_end, activity_end) - max(current_bin_start, activity_start)
                overlap_hours = overlap_duration.total_seconds() / 3600

                if count_transit_destinations and activity.purpose == "Transit":
                    destination_activity = activity.destination  # Assuming destination is a string like "Work", "Home", etc.
                    if destination_activity not in activity_counts:
                        activity_counts[destination_activity] = [0] * bins
                    activity_counts[destination_activity][bin_index % bins] += overlap_hours
                elif not count_transit_destinations:
                    activity_type = activity.purpose if activity.purpose in activity_counts else "Missing data"
                    activity_counts[activity_type][bin_index % bins] += overlap_hours

                current_bin_start += bin_duration
                current_bin_end = current_bin_start + bin_duration

    return pd.DataFrame(activity_counts)

def _get_activity_colors():
    return {
    "Transit": "#9e0142",
    "Travel": "#d53e4f",
    'Grocery': "#f46d43",
    "Shopping": "#fdae61",
    "Leisure": "#fee08b",
    "Home" : "#ffffbf",
    "Work": "#e6f598",
    "Education": "#abdda4",
    'Healthcare': "#66c2a5",
    'Pickup/Dropoff child': "#3288bd",
    "Other": "#5e4fa2",
    # Add other activity types here
    }


def _plot_data(area,analysis_type,df, title, color_palette=None, bins=24, save=True):
    """Plot the stacked area chart for a given number of bins."""
    if color_palette is None:
        color_palette = _get_activity_colors()  # Define or fetch your color palette

    # Calculate the time labels based on the number of bins
    time_labels = [f"{(hour * 24 // bins + 3) % 24}:00" for hour in range(bins)]

    # Create a stacked area plot using Matplotlib with the specified color palette
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(np.arange(bins), df.values.T, labels=df.columns, colors=[color_palette[col] for col in df.columns], alpha=0.7)

    # Set the x-ticks and labels
    ax.set_xticks(np.arange(bins))
    ax.set_xticklabels(time_labels, rotation=45)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Aggregated trip count")

    # Customize Y-Axis Ticks (Example: Using thousands separators)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    # Title and Axis Labels
    ax.set_title(title, fontsize=12)

    # Adjust legend placement and size
    # Legend order is reversed to match the stacked order
    #ax.legend(title="Activity Type", loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')
    # Define custom legend handles to match the stacking order
    # Manually create legend handles based on the custom color palette
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_palette[col]) for col in df.columns]


    # Reverse the order of legend_handles and df.columns
    legend_handles = list(reversed(legend_handles))
    legend_labels = list(reversed(df.columns))

    # Add the legend with custom handles and labels
    ax.legend(legend_handles, legend_labels, title="Activity Type", loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')


    # Remove the frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_alpha(0.2)

    # Adjust grid line transparency
    ax.grid(axis="x", alpha=0.1)

    # Make all text size (legend, title, ticks, axis...) smaller
    plt.rcParams.update({'font.size': 8})

    # Title font size to 10
    ax.title.set_size(10)

    # Adjust the subplot to fit the legend
    plt.tight_layout()

    # Save the plot as a PNG file
    if save:
        plt.savefig(f"static/{area}_{analysis_type}.png", bbox_inches="tight")

    # Show the plot
    plt.show()

def create_color_palette(cmap, activity_labels):
    """
    Generate a color palette dictionary based on a colormap and activity labels.

    Args:
        cmap (str or matplotlib.colors.Colormap): The colormap to use.
        activity_labels (list): List of activity labels.

    Returns:
        dict: Color palette dictionary mapping activity labels to colors.
    """
    # Create a colormap object if cmap is a string
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Generate equidistant color values from the colormap
    num_colors = len(activity_labels)
    color_values = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    # Convert color values to hexadecimal strings
    color_palette = {label: rgb_to_hex(color) for label, color in zip(activity_labels, color_values)}

    return color_palette

def rgb_to_hex(rgb_color):
    """
    Convert an RGB color tuple to a hexadecimal color string.

    Args:
        rgb_color (tuple): RGB color tuple (e.g., (0.1, 0.2, 0.3)).

    Returns:
        str: Hexadecimal color string (e.g., '#1a3456').
    """
    r, g, b, _ = rgb_color
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return hex_color
#activity_labels = ["Transit", "Travel", "Grocery", "Shopping", "Leisure", "Home", "Work", "Education", "Healthcare", "Pickup/Dropoff child", "Other"]
#cmap = 'RdBu'  # Replace with your desired colormap
#divergent_colormaps = plt.colormaps()
#color_palette = create_color_palette(cmap, activity_labels)


def plot_amenity_demand(area,activity_sequences, color_palette=None, bins=24):
    df = _count_activities(activity_sequences, count_transit_destinations=True, bins=bins)
    analysis_type = "amenity_demand"
    _plot_data(area,analysis_type,df, "Activity Demand Profile (Starting at 3:00 AM)", color_palette= color_palette, bins=bins)
    return df

def plot_activity_engagement(area,activity_sequences, color_palette=None, bins=24):
    df = _count_activities(activity_sequences, bins = bins)
    analysis_type = "activity_engagement"
    _plot_data(area,analysis_type,df, "Activity Engagement Profile (Starting at 3:00 AM)", color_palette = color_palette,bins=bins)
    return df