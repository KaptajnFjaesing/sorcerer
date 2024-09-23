"""
Created on Tue Sep 10 19:28:04 2024

@author: Jonas Petersen
"""

import json
from typing import Dict

def get_default_model_config() -> Dict:
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
    sampler_config: Dict = {
        "draws": 1000,
        "tune": 200,
        "chains": 1,
        "cores": 1,
        "sampler": "NUTS",
        "nuts_sampler": "pymc",
        "verbose": True
    }
    return sampler_config

def serialize_model_config(config: Dict) -> str:
    return json.dumps(config)

def deserialize_model_config(config_str: str) -> Dict:
    return json.loads(config_str)
