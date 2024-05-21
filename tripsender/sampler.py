
# -----------------------------------------------------
# sampler.py
# -----------------------------------------------------
# Description: Contains the helper functions Sampler class used to sample from distributions.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.sampler"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, invgauss, genextreme, weibull_max,lognorm
from sklearn.mixture import GaussianMixture
import logging
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

class Sampler:
    def __init__(self, json_data):
        self.json_data = json_data
        self.gmm_cache = {}  # Cache for GMM objects

    def _get_gmm(self, num_components, parameters):
        # Check cache for an existing GMM with the desired number of components
        cache_key = f"gmm_{num_components}"
        if cache_key not in self.gmm_cache:
            # If not in cache, create a new GMM
            gmm = GaussianMixture(n_components=num_components, covariance_type='full')
            
            # Set means
            gmm.means_ = np.array(parameters[1]).reshape(num_components, -1)
            
            # Set covariances and ensure they are correctly shaped for 'full' covariance type
            covariances = np.array(parameters[2])
            if len(covariances.shape) == 2:
                covariances = covariances.reshape(num_components, covariances.shape[2], covariances.shape[2])
            gmm.covariances_ = covariances
            
            # Set weights and normalize them
            weights = np.array(parameters[0]).flatten()
            normalized_weights = weights / weights.sum()
            gmm.weights_ = normalized_weights
            
            # Store the newly created GMM in the cache
            self.gmm_cache[cache_key] = gmm
        
        # Return the GMM (either from cache or the newly created one)
        return self.gmm_cache[cache_key]

    def sample_from_distribution(self, distribution_type, parameters):
        if distribution_type == 'gamma':
            return gamma.rvs(*parameters)
        elif distribution_type == 'invgauss':
            return invgauss.rvs(*parameters)
        elif distribution_type == 'lognorm':
            return lognorm.rvs(*parameters)
        elif distribution_type == 'genextreme':
            return genextreme.rvs(*parameters)
        elif distribution_type == 'weibull_max':
            return weibull_max.rvs(*parameters)
        elif distribution_type in ['bimodal', 'trimodal']:
            num_components = 2 if distribution_type == 'bimodal' else 3
            gmm = self._get_gmm(num_components, parameters)
            return float(gmm.sample()[0][0][0])
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")


class DurationSampler(Sampler):
    def sample_duration(self, purpose, min_duration=None, max_duration=None):
        # Make purpose case insensitive
        purpose = purpose.lower()
        # Make labels in json_data case insensitive
        self.json_data = {k.lower(): v for k, v in self.json_data.items()}
        
        if purpose not in self.json_data:
            print(f"No data found for purpose: {purpose}")
            return None
        data = self.json_data[purpose]
        sample = self.sample_from_distribution(data['distribution'], data['parameters'])
        sample_minutes = sample * 60
        
        # Ensure the sample is within the desired range
        while (min_duration is not None and sample_minutes < min_duration) or (max_duration is not None and sample_minutes > max_duration):
            sample = self.sample_from_distribution(data['distribution'], data['parameters'])
            sample_minutes = sample * 60
            # Round to nearest minute
            sample_minutes = round(sample_minutes)

        return sample_minutes


class StartTimeSampler(Sampler):
    def sample_start_time(self, purpose, min_time=None, max_time=None):
        if purpose not in self.json_data:
            print(f"No data found for purpose: {purpose}")
            return None
        data = self.json_data[purpose]
        sample = self.sample_from_distribution(data['distribution'], data['parameters'])
        sample_time = abs(sample) % 24  # Wrap around to fit within 24 hours and ensure non-negative
        
        # Ensure the sample is within the desired range
        while (min_time is not None and sample_time < min_time) or (max_time is not None and sample_time > max_time):
            sample = self.sample_from_distribution(data['distribution'], data['parameters'])
            sample_time = abs(sample) % 24

        # Return in 'HHMM' format
        return f"{int(sample_time):02d}{int((sample_time * 60) % 60):02d}"
