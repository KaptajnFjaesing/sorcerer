"""
Created on Tue Sep 10 19:24:46 2024

@author: Jonas Petersen
"""

import numpy as np
import pymc as pm
import pytensor as pt
import hashlib
import json
from typing import (
    Dict, 
    Union,
    Tuple
    )

def determine_significant_periods(
    series: np.array,
    x_train: np.array,
    threshold: float
) -> Tuple[float, np.array]:
    """
    Determine the significant periods in a time series using FFT.

    Parameters:
    - series: numpy array of the time series data.
    - x_train: numpy array of the training data (time points).
    - threshold: float threshold to determine significance.

    Returns:
    - dominant_period: float of the most dominant period.
    - significant_periods: numpy array of significant periods above the threshold.
    """
    # FFT of the time series
    fft_result = np.fft.fft(series)
    fft_magnitude = np.abs(fft_result)
    fft_freqs = np.fft.fftfreq(len(series), x_train[1] - x_train[0])

    # Get positive frequencies
    positive_freqs = fft_freqs[fft_freqs > 0]
    positive_magnitudes = fft_magnitude[fft_freqs > 0]

    # Find the dominant component
    max_magnitude = np.max(positive_magnitudes)
    max_index = np.argmax(positive_magnitudes)
    dominant_frequency = positive_freqs[max_index]
    dominant_period = 1 / dominant_frequency

    # Find components that are more than the threshold fraction of the maximum
    significant_indices = np.where(positive_magnitudes >= threshold * max_magnitude)[0]
    significant_frequencies = positive_freqs[significant_indices]
    significant_periods = 1 / significant_frequencies

    return dominant_period, significant_periods

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
