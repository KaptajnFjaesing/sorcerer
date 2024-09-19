"""
Created on Tue Sep 10 19:28:04 2024

@author: Jonas Petersen
"""

import json
from typing import Dict

def get_default_model_config() -> Dict:
    """
    Returns the default model configuration as a dictionary.
    
    The configuration includes parameters for priors, the number of components, 
    forecast horizon, and other hyperparameters used in the model.
    
    Returns:
    - A dictionary containing the default model configuration.
    """
    model_config: Dict = {
        "number_of_individual_trend_changepoints": 3,
        "delta_mu_prior": 0,
        "delta_b_prior": 0.2,
        "m_sigma_prior": 0.1,
        "k_sigma_prior": 0.1,
        "fourier_mu_prior": 0,
        "fourier_sigma_prior" : 1,
        "precision_target_distribution_prior_alpha": 2,
        "precision_target_distribution_prior_beta": 1,
        "prior_probability_shared_seasonality_alpha": 1,
        "prior_probability_shared_seasonality_beta": 1,
        "individual_fourier_terms": [],
        "shared_fourier_terms": [] 
    }
    

    
    return model_config


def get_default_sampler_config() -> Dict:
    """
    Returns the default sampler configuration as a dictionary.
    
    The configuration includes parameters for MCMC sampling such as the number of draws, 
    tuning steps, chains, cores, and target acceptance rate.
    
    Returns:
    - A dictionary containing the default sampler configuration.
    """
    sampler_config: Dict = {
        "draws": 1000,
        "tune": 200,
        "chains": 1,
        "cores": 1,
        "sampler": "NUTS",
        "nuts_sampler": "pymc"
    }
    return sampler_config


def serialize_model_config(config: Dict) -> str:
    """
    Serializes the model configuration dictionary to a JSON string.
    
    Parameters:
    - config: A dictionary containing the model configuration.
    
    Returns:
    - A JSON string representation of the model configuration.
    """
    return json.dumps(config)


def deserialize_model_config(config_str: str) -> Dict:
    """
    Deserializes a JSON string to a model configuration dictionary.
    
    Parameters:
    - config_str: A JSON string representation of the model configuration.
    
    Returns:
    - A dictionary containing the model configuration.
    """
    return json.loads(config_str)
