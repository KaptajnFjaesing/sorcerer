"""
Created on Tue Sep 10 19:26:17 2024

@author: Jonas Petersen
"""

import numpy as np
import pymc as pm
import pytensor as pt
from sorcerer.utils import create_fourier_features


def add_linear_term(
    model: pm.Model,
    x: np.array,
    baseline_slope: float,
    baseline_bias: float,
    trend_name: str,
    number_of_time_series: int,
    number_of_trend_changepoints: int,
    delta_mu_prior: float,
    delta_b_prior: float,
    m_sigma_prior: float,
    k_sigma_prior: float
):
    """
    Adds a linear term to the model with change points for the trend.

    Parameters:
    - model: The PyMC model context.
    - x: numpy array of the input time points.
    - baseline_slope: float, initial slope of the trend.
    - baseline_bias: float, initial intercept of the trend.
    - trend_name: str, the name of the trend component.
    - number_of_time_series: int, number of time series being modeled.
    - number_of_trend_changepoints: int, number of changepoints for the trend.
    - maximum_x_value: float, maximum value of x for normalization.
    - delta_mu_prior: float, prior mean for change in slope.
    - delta_b_prior: float, prior scale (b) for change in intercept.
    - m_sigma_prior: float, prior standard deviation for intercept.
    - k_sigma_prior: float, prior standard deviation for slope.

    Returns:
    - A PyMC Deterministic variable representing the linear term with changepoints.
    """
    with model:
        s = pt.tensor.linspace(0, 1, number_of_trend_changepoints + 2)[1:-1] # max(x) for input is by definition 1
        A = (x[:, None] > s) * 1.
        k = pm.Normal(f'{trend_name}_k', mu=baseline_slope, sigma=k_sigma_prior, shape=number_of_time_series)
        delta = pm.Laplace(f'{trend_name}_delta', mu=delta_mu_prior, b=delta_b_prior, shape=(number_of_time_series, number_of_trend_changepoints))
        m = pm.Normal(f'{trend_name}_m', mu=baseline_bias, sigma=m_sigma_prior, shape=number_of_time_series)
    return (k + pm.math.dot(A, delta.T)) * x[:, None] + m + pm.math.dot(A, (-s * delta).T)

def add_fourier_term(
    model: pm.Model,
    x: np.array,
    number_of_fourier_components: int,
    name: str,
    dimension: int,
    seasonality_period_baseline: float,
    fourier_sigma_prior: float,
    fourier_mu_prior: float
):
    """
    In order to not depend on perfect specification of the periodicity, I have
    tried with the priors below, however, this introduces an increased rate of
    converges issues and divergences.
    
    seasonality_period = pm.Gamma(
        f'season_parameter_{name}',
        alpha=relative_uncertainty_factor_prior * seasonality_period_baseline,
        beta=relative_uncertainty_factor_prior
        )
    or
    
    seasonality_period = pm.Lognormal(
        f'seasonality_period_{name}',
        mu=np.log(seasonality_period_baseline),
        sigma=relative_uncertainty_factor_prior
        )
    """
    with model:
        fourier_coefficients = pm.Normal(f'fourier_coefficients_{name}', mu=fourier_mu_prior, sigma = fourier_sigma_prior, shape=(2 * number_of_fourier_components, dimension))
        fourier_features = create_fourier_features(x=x, number_of_fourier_components=number_of_fourier_components, seasonality_period=seasonality_period_baseline)
    return pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1)
