"""
Created on Tue Sep 10 19:26:17 2024

@author: Jonas Petersen
"""

import pandas as pd
import pymc as pm
import pytensor as pt
from sorcerer.utils import create_fourier_features

def trend_and_fourier_terms(
        X: pd.Series,
        baseline_slope: pd.Series,
        baseline_bias: pd.Series,
        model_config: dict,
        model: pm.Model
        ):
    
    dX = X.iloc[1]-X.iloc[0]
    number_of_observations = X.shape[0]
    number_of_time_series = baseline_slope.shape[0]
    
    with model:
        model.add_coords({"number_of_time_series": range(number_of_time_series)})
        model.add_coords({"number_of_observations": range(number_of_observations)})
        x = pm.Data('input_training', X)
                
        s = pt.tensor.linspace(0, 1, model_config["number_of_individual_trend_changepoints"] + 2)[1:-1] # max(x) for input is by definition 1
        A = (x[:, None] > s) * 1.
        k = pm.Normal(
            name = 'trend_k',
            mu = baseline_slope,
            sigma = model_config["k_sigma_prior"],
            shape = number_of_time_series
            )
        delta = pm.Laplace(
            name = 'trend_delta',
            mu = model_config["delta_mu_prior"],
            b = model_config["delta_b_prior"],
            shape = (
                number_of_time_series,
                model_config["number_of_individual_trend_changepoints"]
                )
            )
        m = pm.Normal(
            name = 'trend_m',
            mu = baseline_bias,
            sigma = model_config["m_sigma_prior"],
            shape = number_of_time_series
            )
        trend_term = (k + pm.math.dot(A, delta.T)) * x[:, None] + m + pm.math.dot(A, (-s * delta).T)

        seasonality_individual = pm.math.zeros((number_of_observations, number_of_time_series))
        for term in model_config["individual_fourier_terms"]:
            number_of_fourier_components = term['number_of_fourier_components']
            seasonality_period_baseline = term['seasonality_period_baseline'] * dX
            
            # Define distinct variable names for coefficients
            fourier_coefficients = pm.Normal(
                name = f'fourier_coefficients_{round(seasonality_period_baseline, 2)}_{number_of_fourier_components}',
                mu = model_config["fourier_mu_prior"],
                sigma = model_config["fourier_sigma_prior"],
                shape = (2 * number_of_fourier_components, number_of_time_series)
            )
            
            # Create Fourier features
            fourier_features = create_fourier_features(
                x = x,
                number_of_fourier_components = number_of_fourier_components,
                seasonality_period = seasonality_period_baseline
            )
            seasonality_individual += pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1)
        
        single_scale = pm.Normal(
            name = 'single_scale',
            mu = model_config["single_scale_mu_prior"],
            sigma = model_config["single_scale_sigma_prior"],
            shape = number_of_time_series
            )
        seasonality_individual = seasonality_individual*single_scale
        
        if len(model_config["shared_fourier_terms"]) > 0:
            shared_seasonalities = []
            for term in model_config["shared_fourier_terms"]:
                number_of_fourier_components = term['number_of_fourier_components']
                seasonality_period_baseline = term['seasonality_period_baseline'] * dX
                
                # Define distinct variable names for coefficients
                fourier_coefficients = pm.Normal(
                    name = f'fourier_coefficients_shared_{round(seasonality_period_baseline, 2)}_{number_of_fourier_components}',
                    mu = model_config["fourier_mu_prior"],
                    sigma = model_config["fourier_sigma_prior"],
                    shape = (2 * number_of_fourier_components, 1)
                )
                
                # Create Fourier features
                fourier_features = create_fourier_features(
                    x = x,
                    number_of_fourier_components = number_of_fourier_components,
                    seasonality_period = seasonality_period_baseline
                )
                
                # Calculate the seasonal term and store it
                shared_seasonality_term = pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1)
                shared_seasonalities.append(shared_seasonality_term)
        
            # Combine all shared seasonal terms into one array
            shared_seasonalities = pm.math.concatenate(shared_seasonalities, axis=1)
            
            shared_scale = pm.Normal(
                name = 'shared_scale',
                mu = model_config["shared_scale_mu_prior"],
                sigma = model_config["shared_scale_sigma_prior"],
                shape = (
                    len(model_config["shared_fourier_terms"]),
                    number_of_time_series
                    )
                )
            
            shared_seasonality = pm.math.dot(shared_seasonalities, shared_scale)
        else:
            shared_seasonality = 0

        return trend_term + seasonality_individual + shared_seasonality