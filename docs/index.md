# Introduction to GAPSIM
GAPSIM (Gothenburg Activity-based Population Simulator)

Welcome to **GAPSIM**, a library designed for modelling synthetic populations for neighbourhoods in Gothenburg, Sweden. This documentation provides an overview of GAPSIM's core functionalities.

## Key Features

### 1. Population Synthesis
This module generates a synthetic population reflecting real-world demographics. It includes:
- **Person Generation**: Creates individuals based on age, household type, and sex distribution.
- **Household Formation**: Groups individuals into households considering various compositions.
- **Attribute Assignment**: Assigns socioeconomic attributes to individuals and households to match observed distributions.

### 2. Active Mobility Route Choice
This module models route choices for individuals engaged in active mobility, such as walking or cycling. It factors in preferences for route characteristics, ensuring realistic route choice predictions.

## Application Context

### Article Information
- **Title**: An activity-based synthetic population of Gothenburg, Sweden: Dataset of residents in neighbourhoods
- **Authors**: Sanjay Somanath*, Liane Thuvander, Alexander Hollberg
- **Affiliations**: Chalmers University of Technology, Department of Architecture and Civil Engineering, Sweden
- **Keywords**: Mobility, activity, energy, neighbourhood-planning, demand-modelling, accessibility, equity

### Abstract
The paper presents an end-to-end model for generating a synthetic population of Gothenburg residents, complete with activity schedules and mobility patterns. This model supports neighbourhood planning and integrates primary datasets to create synthetic individuals, households, and activity chains. The model is designed to capture the nuances of a neighbourhood’s built environment and demographic composition.

### Specifications
- **Data Type**: SQL database with multiple linked tables for each neighbourhood
- **Data Sources**: Statistics Sweden (SCB), National Household Travel Survey (NHTS), Swedish cadastral agency (Lantmäteriet), OpenStreetMap, and City of Gothenburg data
- **Accessibility**: Available on Zenodo ([10.5281/zenodo.10801935](https://www.doi.org/10.5281/zenodo.10801935))

### Value of the Data
- **Agent-Based Models**: Provides a high-quality synthetic population for agent-based models.
- **Urban and Neighbourhood Planning**: Assists in assessing the impacts of planning strategies.
- **Energy Demand Modelling**: Helps in evaluating the effects of different energy policies.
- **Behavioural Studies**: Facilitates neighbourhood-level behavioural studies.
- **Digital Twinning and Urban Simulation**: Supports the creation of digital twins and comprehensive simulations.

## Getting Started
To begin using TripSender, install the library and explore its core modules. The documentation offers detailed instructions, examples, and best practices to help you utilize the library's full potential. 

Note: This package is still a Work In Progress. Please contact the authors for access to the raw data required for population synthesis or on suggestions to setup the data pipelines yourself.

We hope TripSender enhances your neighbourhood modeling projects and contributes to efficient and sustainable neighbourhood planning. For detailed guidance on each module, please refer to the respective sections of the documentation. Happy modeling!

## API Documentation

- [Activity](activity.md)
- [Building](building.md)
- [House](house.md)
- [Household](household.md)
- [IO](io.md)
- [Location Assignment](location_assignment.md)
- [NHTS](nhts.md)
- [OD](od.md)
- [Person](person.md)
- [Population](population.md)
- [Routing](routing.md)
- [Sampler](sampler.md)
- [Synthpop](synthpop.md)
- [Utils](utils.md)
