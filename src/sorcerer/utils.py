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

def compute_residuals(model_forecasts, test_data, min_forecast_horizon):
    """
    Compute the forecast error metrics for multiple models.

    Parameters:
    - model_forecasts: List of forecasted values from different models. Each element is a list of forecasts for each column.
    - test_data: DataFrame containing the actual values for the forecast period.
    - min_forecast_horizon: The minimum forecast horizon to consider for each model.

    Returns:
    - List of DataFrames, each containing the forecast errors for a model.
    """
    metrics_list = []

    # Iterate over each model's forecasts
    for forecasts in model_forecasts:
        # Initialize an empty DataFrame to store forecast errors for the current model
        errors_df = pd.DataFrame(columns=test_data.columns)
        
        # Calculate forecast errors for each column
        for i, column in enumerate(test_data.columns):
            # Get the forecast horizon for the current column
            forecast_horizon = len(forecasts[i].values)
            
            # Calculate forecast errors: actual values minus forecasted values
            errors_df[column] = test_data[column].values[-forecast_horizon:] - forecasts[i].values
        
        # Keep only the first `min_forecast_horizon` rows and reset index
        truncated_errors_df = errors_df.iloc[:min_forecast_horizon].reset_index(drop=True)
        
        # Append the DataFrame of errors for the current model to the list
        metrics_list.append(truncated_errors_df)
    stacked = pd.concat(metrics_list, axis=0)
    
    return stacked
