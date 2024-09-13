"""
Created on Tue Sep 10 19:24:46 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
import hashlib
import json
from typing import (
    Dict, 
    Union
    )

def normalize_training_data(training_data: pd.DataFrame) -> tuple:
    time_series_columns = [x for x in training_data.columns if 'date' not in x]
    x_training_min = (training_data['date'].astype('int64')//10**9).min()
    x_training_max = (training_data['date'].astype('int64')//10**9).max()
    y_training_min = training_data[time_series_columns].min()
    y_training_max = training_data[time_series_columns].max()
    x_train = (training_data['date'].astype('int64')//10**9 - x_training_min)/(x_training_max - x_training_min)
    y_train = (training_data[time_series_columns]-y_training_min)/(y_training_max-y_training_min)
    return x_train, x_training_min, x_training_max, y_train, y_training_min, y_training_max

def create_fourier_features(
    x: np.array,
    number_of_fourier_components: int,
    seasonality_period: float
) -> np.array:
    """
    Create Fourier features for modeling seasonality in time series data.

    Parameters:
    - x: numpy array of the input time points.
    - number_of_fourier_components: integer, number of Fourier components to use.
    - seasonality_period: float, the base period for seasonality.

    Returns:
    - Fourier features as a numpy array.
    """
    # Frequency components
    frequency_component = pt.tensor.as_tensor_variable(2 * np.pi * (np.arange(number_of_fourier_components) + 1) * x[:, None])
    t = frequency_component[:, :, None] / seasonality_period  # Normalize by the period

    # Concatenate sine and cosine features
    return pm.math.concatenate((pt.tensor.cos(t), pt.tensor.sin(t)), axis=1)

def generate_hash_id(
    model_config: Dict[str, Union[int, float, Dict]],
    version: str,
    model_type: str
) -> str:
    """
    Generates a unique hash ID for a model based on its configuration, version, and type.

    Parameters
    ----------
    model_config : dict
        The configuration dictionary of the model.
    version : str
        The version of the model.
    model_type : str
        The type of the model.

    Returns
    -------
    str
        A unique hash string representing the model.
    """
    # Serialize the model configuration to a JSON string
    config_str = json.dumps(model_config, sort_keys=True)
    # Create a string to hash, combining the model type, version, and configuration
    hash_input = f"{model_type}-{version}-{config_str}"
    # Generate a unique hash ID
    hash_id = hashlib.md5(hash_input.encode()).hexdigest()
    return hash_id
