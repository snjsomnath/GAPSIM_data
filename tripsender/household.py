# -----------------------------------------------------
# household.py
# -----------------------------------------------------
# Description: Contains the Household class used to represent a household in the simulation.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.household"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import uuid
import networkx as nx
import matplotlib.pyplot as plt
import logging
import pandas as pd
import matplotlib.patches as patches
from datetime import time
from typing import TYPE_CHECKING, List, Optional
from tripsender.person import Person
from tripsender.house import House
from sklearn.preprocessing import OneHotEncoder
# Plotting the counts
import matplotlib.pyplot as plt
from tripsender import fetcher
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

# Constants
MALE = 'Män'
FEMALE = 'Kvinnor'
COUPLE = 'Couple'

class Household:
    """
    Represents a household with members, cars, and other properties.

    A household represents a collection of persons that live together. It comprises:
        - Household category
        - House type
        - List of household members
        - Unique identifier
        - Reference to a house object
        - Preferred destinations for different activities related to the household
        - Number of cars in the household
        - Number of children in the household

    Attributes:
        uuid (UUID): Unique identifier for the household.
        category (str): Category of the household.
        children (int): Number of children in the household.
        has_children (bool): Whether the household has children.
        adults (int): Number of adults in the household.
        members (List[Person]): List of members in the household.
        house_type (Optional[str]): Type of house the household lives in.
        house (Optional[House]): Reference to the house object associated with the household.
        cars (int): Number of cars owned by the household.
        head_of_household (Optional[Person]): The head of the household.
    """
    instances: List['Household'] = []
    
    def __init__(self, category):
        """Initialize the Household with given attributes."""
        self.uuid: uuid.UUID = uuid.uuid4()
        self.category: str = category
        self.children : int = 0 
        self.has_children: bool = False
        self.adults: int = 0
        self.members : List[Person] = []
        self.instances.append(self)
        self.house_type: Optional[str] = None
        self.house: Optional[House] = None  # assuming there's a House class
        self.cars: int = 0
        self.head_of_household: Person = None

    def __repr__(self):
        """Representation of the Household instance."""
        return f"A {self.category} household with {len(self.members)} members. {self.children} children and {self.adults} adults."

    def add_member(self, person):
        """Adds a person to the household."""
        self.members.append(person)
        person.household = self
        # If person is not child, then increase adult count
        if not person.is_child:
            self.adults += 1
        # Set the household UUID to person.household_uuid
        person.household_uuid = self.uuid
        # If there are no persons in the household, the the first person to be added is the head of the household
        # As people are added to the household, check the age of the person and if the person is older than the current head of the household, then replace the head of the household with the new person
        if len(self.members) == 1:
            self.head_of_household = person
            person.is_head = True
        elif person.age > self.head_of_household.age:
            self.head_of_household.is_head = False
            self.head_of_household = person
            person.is_head = True
    
    def add_child(self, person):
        """Adds a child to the household."""
        # Confirm that the person is a child
        person.is_child = True
        self.members.append(person)
        person.household = self
        self.children += 1
        self.has_children = True
        # Set the household UUID to person.household_uuid
        person.household_uuid = self.uuid
        # Set parent uuid
        person.parent_uuid = self.head_of_household.uuid
        
    
    @classmethod
    def sync_children_in_households(cls):
        """Syncs the number of children in households."""
        for household in cls.instances:
            household.children = sum(1 for member in household.members if member.is_child)
            household.has_children = household.children > 0

    @classmethod
    def return_dataframe(cls):
        """Returns a dataframe with information about the households."""
        data = []
        for household in cls.instances:
            household_dict = {
                "uuid_household": household.uuid,
                "name_category": household.category,
                "count_children": household.children,
                "bool_children": household.has_children,
                "count_adults": household.adults,
                "count_members": len(household.members),
                "uuid_members": [member.uuid for member in household.members],
                "type_house": household.house_type,
                "uuid_house": household.house.uuid if household.house else None,
                "count_cars": household.cars,
                "head_of_household": household.head_of_household.uuid if household.head_of_household else None
            }
            data.append(household_dict)
        return pd.DataFrame(data)


    def info(self):
        """Returns a dictionary with information about the household."""
        info_dict = {
            "UUID": self.uuid,
            "Category": self.category,
            "Number of Members": len(self.members),
            "Members": [member.info() for member in self.members],
            "Children": self.children,
            "House Type": self.house_type,
            "Cars": self.cars
        }
        
        return info_dict

    @classmethod
    def clear_instances(cls):
        cls.instances = []

    @classmethod
    def class_info(cls):
        """Print information about the household categories and their compositions."""

        # Get categories
        categories = [household.category for household in cls.instances]
        unique_categories = set(categories)

        # Count males and females
        males = sum(1 for person in Person.instances if person.sex == MALE)
        females = len(Person.instances) - males

        # Count same-sex couples and households with children
        same_sex_couples = sum(1 for household in cls.instances if household.category == COUPLE and household.members[0].sex == household.members[1].sex)
        households_with_children = sum(1 for household in cls.instances if household.children)

        # Count total children
        total_children = sum(1 for household in cls.instances for person in household.members if person.is_child)

        # Count total cars
        total_cars = sum(household.cars for household in cls.instances)

        # Print the gathered information
        logger.info(f'There are {len(unique_categories)} categories in this dataset')
        logger.info(f'There are {len(cls.instances)} households')
        logger.info(f'There are {len(Person.instances)} persons')
        logger.info(f'The categories are {unique_categories}')
        
        for category in unique_categories:
            logger.info(f'     There are {categories.count(category)} {category} households')

        logger.info(f'There are {males} males and {females} females in the population')
        logger.info(f'There are {same_sex_couples} same sex couples out of {categories.count(COUPLE)} couples')
        logger.info(f'There are {households_with_children} households with children out of {len(cls.instances)} households')
        logger.info(f'There are {total_children} children out of {len(Person.instances)} persons')
        logger.info(f'There are {total_cars} cars in all households')
    
    def plot_member_graph(self):
        """ Plots a relational graph of all members in the household"""
        members = self.members
        adults = [member for member in members if member.is_child == False]
        children = [member for member in members if member.is_child == True]
        G = nx.Graph()

        # Adding household node with size = number of members and colour = red
        G.add_node(self.uuid, type="household", size=len(self.members)*10, color="#FF6961")

        # Adding adult nodes with size = age and colour = blue
        for adult in adults:
            G.add_node(adult.uuid, type="adult", age=adult.age, size=adult.age, color="#AEC6CF",gender=adult.sex)
        
        # Adding child nodes with size = age and colour = green
        for child in children:
            G.add_node(child.uuid, type="child", age=child.age, size=child.age, color="#77DD77",gender=child.sex)

        # Adding edges between household and adults
        for adult in adults:
            G.add_edge(self.uuid, adult.uuid)

        # Adding edges between adults and children with length = 1
        for adult in adults:
            for child in children:
                G.add_edge(adult.uuid, child.uuid, length=1)

        # Plotting the graph with node size and colour
        plt.figure(figsize=(10, 10))

        # Draw a hierarchical tree graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_size=[v['size']*50 for v in G.nodes.values()],
                node_color=[v['color'] for v in G.nodes.values()])

        # Add node labels with age and gender attributes
        node_labels = {}
        for node in G.nodes:
            if G.nodes[node]['type'] == 'household':
                continue
            age = G.nodes[node]['age']
            gender = G.nodes[node]['gender']
            node_labels[node] = f"Age: {age}\nGender: {gender}"

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="black", font_size=8)

        # Add the title and subtitle
        plt.title("Household Graph")
        plt.suptitle(f"Household Category: {self.category} - Number of Members: {len(self.members)}", fontsize=12)

        plt.show()

    def plot_activities(self,color_palette=None):
        if not any(member.activity_sequence.activities for member in self.members):
            print("No activities to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        if color_palette:
            activity_colors = color_palette
        else:
            activity_colors = self._get_activity_colors()

        y_ticks, y_labels = self._get_member_info()

        for y, member in zip(y_ticks, self.members):
            if member.activity_sequence is  None:
                self._add_activity_rectangles(ax, 0, 24, y, "#d4d4d4","Children activities are not included")
            elif member.activity_sequence.activities:
                for activity in member.activity_sequence.activities:
                    start_time, end_time = self._get_activity_times(activity)

                    self._add_activity_rectangles(ax, start_time, end_time, y, activity_colors[activity.purpose])

                    if activity.mode:
                        self._add_activity_mode_label(ax, start_time, end_time, y, activity.mode)

        self._set_plot_settings(ax, y_ticks, y_labels, activity_colors)

        plt.tight_layout()
        plt.show()

    def _get_activity_colors(self):
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

    def _get_member_info(self):
        y_ticks = range(1, len(self.members) + 1)
        y_labels = [member.__str__() for member in self.members]
        return y_ticks, y_labels

    def _get_activity_times(self, activity):
        if activity.start_time == time(3, 0) and activity.end_time == time(3, 0):
            return 0, 24
        return activity.start_time.hour + activity.start_time.minute / 60, activity.end_time.hour + activity.end_time.minute / 60
    #self._add_activity_rectangles(ax, 0, 24, y, "#d4d4d4","Children activities are not included")
    def _add_activity_rectangles(self, ax, start_time, end_time, y, color,text=None):
        if end_time < start_time:
            rects = [patches.Rectangle((start_time, y - 0.3), 24 - start_time, 0.6, facecolor=color, edgecolor=color),
                     patches.Rectangle((0, y - 0.3), end_time, 0.6, facecolor=color, edgecolor=color)]
        else:
            rects = [patches.Rectangle((start_time, y - 0.3), end_time - start_time, 0.6, facecolor=color, edgecolor=color)]
        
        for rect in rects:
            ax.add_patch(rect)
        
        if text:
            ax.text(12, y, text, ha='center', va='center', fontsize=8, color='black', rotation=0)

    def _add_activity_mode_label(self, ax, start_time, end_time, y, mode):
        ax.text((start_time + end_time) / 2, y, mode, ha='center', va='center', fontsize=8, color='white', rotation=90)

    def _set_plot_settings(self, ax, y_ticks, y_labels, activity_colors):
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_ylim(0.5, len(y_labels) + 0.5)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 1))
        ax.set_xlabel("Time of Day", fontsize=8)
        ax.set_ylabel("Household Members", fontsize=8)
        ax.set_title("Household Activities Timeline", fontsize=10)
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, color='#d4d4d4')
        handles = [patches.Patch(color=color, label=purpose) for purpose, color in activity_colors.items()]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=8)

        for spine in ax.spines.values():
            spine.set_visible(False)

    @classmethod
    def plot_cars(cls):
        # Plot distribution of cars in the neighborhood
        cars = {}
        for household in cls.instances:
            if household.cars in cars:
                cars[household.cars] += 1
            else:
                cars[household.cars] = 1

        # Display the bar chart
        plt.bar(cars.keys(), cars.values())
        plt.xticks(rotation=90)
        plt.title('Number of cars per household (After scaling)')
        plt.xlabel('Number of cars')
        plt.ylabel('Count')
        plt.show()
        for k, v in cars.items():
            print(f"Households with {k} cars: {v}")
    
    @classmethod
    def return_nhts(cls, onehotencode=False, drop=['primary_status']):
        """
        Returns a dataframe with information about the households and persons in a common NHTS format.
        """
        
        # Helper function to determine age group
        def _determine_age_group(age):
            if age < 25:
                return '16-24'
            elif age < 35:
                return '25-34'
            elif age < 45:
                return '35-44'
            elif age < 55:
                return '45-54'
            elif age < 65:
                return '55-64'
            elif age < 75:
                return '65-74'
            else:
                return '75+'
        
        list_of_dicts = []

        adults = []

        for household in cls.instances:
            for person in household.members:
                if person.age > 18:
                    adults.append(person)
                    age_group = _determine_age_group(person.age)

                    person_dict = {
                        'sex': person.sex,
                        'age_group': age_group,
                        'house_type': person.household.house_type,
                        'child_count': person.household.children,
                        'adult_count': len(person.household.members) - person.household.children,
                        'household_type': person.household.category,
                        'car_count': person.household.cars,
                        'primary_status': person.primary_status,
                    }

                    list_of_dicts.append(person_dict)

        df = pd.DataFrame(list_of_dicts)

        # If drop, then drop the columns in the list
        if drop:
            df = df.drop(drop, axis=1)
            
        if onehotencode:
            ohe = OneHotEncoder(sparse=False)  # Initialize the encoder
            categorical_variables = ['sex', 'age_group', 'house_type', 'household_type']
            df_ohe = pd.DataFrame(ohe.fit_transform(df[categorical_variables]))
            df_ohe.columns = ohe.get_feature_names()

            df = df.drop(categorical_variables, axis=1).reset_index(drop=True)
            df = pd.concat([df, df_ohe], axis=1)

        return df,adults

    @classmethod
    def plot_children_in_households(cls,year,area):

        # Creating a dictionary for each category
        household_categories = {
            'Single_no_children': [],
            'Single_children_0_24': [],
            'Single_children_25': [],
            'Couple_no_children': [],
            'Couple_children_0_24': [],
            'Couple_children_25': [],
            'Other_no_children': [],
            'Other_children_0_24': [],
            'Other_children_25': []
        }

        household_instances = cls.instances

        # A household can be in only one category from the list above
        # Looping through all households
        for household in household_instances:
            if household.children == 0:
                category = f'{household.category}_no_children'
            # If the household has children, then check the age of the children, if any of the children are below 25,
            # then the household is categorised as children_0_24, otherwise it is categorised as children_25
            elif any(person.age < 25 for person in household.members if person.is_child):
                category = f'{household.category}_children_0_24'
            else:
                category = f'{household.category}_children_25'
            
            household_categories[category].append(household)



        # Creating a list of counts and labels
        counts = []
        labels = []
        for key, value in household_categories.items():
            labels.append(key.replace("_", " "))
            counts.append(len(value))

        print('Number of households: ', sum(counts))

        # show data value on the plot
        for i in range(len(counts)):
            plt.text(x=i, y=counts[i], s=counts[i], ha='center', va='bottom')

        # Plotting the counts
        plt.bar(labels, counts)
        plt.xticks(rotation=90)
        plt.title('Household categories')
        plt.xlabel('Household category')
        plt.ylabel('Count')

        d = fetcher.fetch_older_children_data(year, area)
        scb_data = [int(value['values'][0]) for value in d['data']]
        plt.plot(labels, scb_data, color='red', marker='o')
        plt.legend(['SCB data', 'Our data'])

        # Calculate the error
        error = [counts[i] - scb_data[i] for i in range(len(counts))]

        # Plot the error
        plt.figure()
        plt.bar(labels, error)
        plt.xticks(rotation=90)
        plt.title('Incorrectly categorised persons')
        plt.xlabel('Household category')
        plt.ylabel('Error')
        plt.show()

    def update_has_child(self):
        """ Set the has_child attribute for adults in the household based on the number of children in the household"""
        if self.children > 0:
            self.has_child = True
        else:
            self.has_child = False
        for member in self.members:
            if member.is_child == False:
                member.has_child = self.has_child
