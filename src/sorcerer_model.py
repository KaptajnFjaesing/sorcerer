# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:49:06 2024

@author: Jonas Petersen
"""

import hashlib
import numpy as np
import pymc as pm
import pytensor as pt
import pandas as pd
from typing import Dict, Tuple, Union
from typing import Any
import arviz as az
import json


class SorcererModel:
    # Give the model a name
    _model_type = "SorcererModel"

    # And a version
    version = "0.1"

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
  
        sampler_config = (
            self.get_default_sampler_config() if sampler_config is None else sampler_config
        )
        self.sampler_config = sampler_config
        model_config = self.get_default_model_config() if model_config is None else model_config

        self.model_config = model_config  # parameters for priors etc.
        self.model = None  # Set by build_model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.posterior_predictive: az.InferenceData
        self.is_fitted_ = False

    def build_model(self, X, y, **kwargs):

        number_of_time_series = y.shape[1]
        baseline_slope = (y.values[-1] - y.values[0]) / (X.values[-1] - X.values[0])
        baseline_bias = y.values[0]
        
        # prior parameters
        number_of_individual_trend_changepoints = self.model_config["number_of_individual_trend_changepoints"]
        forecast_horizon = self.model_config["forecast_horizon"]
        target_standard_deviation = self.model_config["target_standard_deviation"]
        number_of_individual_fourier_components = self.model_config["number_of_individual_fourier_components"]
        period_threshold = self.model_config["period_threshold"]
        number_of_shared_seasonality_groups = self.model_config["number_of_shared_seasonality_groups"]
        delta_mu_prior = self.model_config["delta_mu_prior"]
        delta_b_prior = self.model_config["delta_b_prior"]
        k_sigma_prior = self.model_config["k_sigma_prior"]
        m_sigma_prior = self.model_config["m_sigma_prior"]
        relative_uncertainty_factor_prior = self.model_config["relative_uncertainty_factor_prior"]
         
        (dominant_period, significant_periods) = self.determine_significant_periods(
                series = y.values[:,0],
                x_train = X.values,
                threshold = period_threshold
                )
        seasonality_period_baseline = min(significant_periods)
        with pm.Model() as self.model:
            x = pm.Data('input', X, dims='number_of_input_observations')
            y = pm.Data('target', y, dims=['number_of_target_observations', 'number_of_time_series'])


            linear_term1 = self.add_linear_term(
                x=x,
                baseline_slope = baseline_slope,
                baseline_bias = baseline_bias,
                trend_name='linear1',
                number_of_time_series=number_of_time_series,
                number_of_trend_changepoints=number_of_individual_trend_changepoints,
                maximum_x_value = (1+forecast_horizon)*max(X),
                delta_mu_prior = delta_mu_prior,
                delta_b_prior = delta_b_prior,
                m_sigma_prior = m_sigma_prior,
                k_sigma_prior = k_sigma_prior
            )

            seasonality_individual1 = self.add_fourier_term(
                x=x,
                number_of_fourier_components=number_of_individual_fourier_components,
                name='seasonality_individual1',
                dimension=number_of_time_series,
                seasonality_period_baseline=seasonality_period_baseline,
                relative_uncertainty_factor_prior = relative_uncertainty_factor_prior
            )

            seasonality_shared = self.add_fourier_term(
                x=x,
                number_of_fourier_components=number_of_individual_fourier_components,
                name='seasonality_shared',
                dimension=number_of_shared_seasonality_groups - 1,
                seasonality_period_baseline=seasonality_period_baseline,
                relative_uncertainty_factor_prior = relative_uncertainty_factor_prior
            )

            all_models = pm.math.concatenate([x[:, None] * 0, seasonality_shared], axis=1)
            model_probs = pm.Dirichlet('model_probs', a=np.ones(number_of_shared_seasonality_groups), shape=(number_of_time_series, number_of_shared_seasonality_groups))
            chosen_model_index = pm.Categorical('chosen_model_index', p=model_probs, shape=number_of_time_series)
            shared_seasonality_models = all_models[:, chosen_model_index]

            prediction = (
                    linear_term1 +
                    seasonality_individual1 +
                    shared_seasonality_models
            )

            pm.Normal('target_distribution', mu=prediction, sigma=target_standard_deviation, observed=y, dims=['number_of_input_observations', 'number_of_time_series'])


    def add_linear_term(
            self,
            x: np.array,
            baseline_slope: float,
            baseline_bias:float,
            trend_name: str,
            number_of_time_series: int,
            number_of_trend_changepoints: int,
            maximum_x_value: float,
            delta_mu_prior: float,
            delta_b_prior: float,
            m_sigma_prior: float,
            k_sigma_prior: float
    ) -> pm.Deterministic:
        with self.model:
            s = pt.tensor.linspace(0, maximum_x_value, number_of_trend_changepoints + 2)[1:-1]
            A = (x[:, None] > s) * 1.
            k = pm.Normal(f'{trend_name}_k', mu=baseline_slope, sigma=k_sigma_prior, shape=number_of_time_series)
            delta = pm.Laplace(f'{trend_name}_delta', mu=delta_mu_prior, b=delta_b_prior, shape=(number_of_time_series, number_of_trend_changepoints))
            m = pm.Normal(f'{trend_name}_m', mu=baseline_bias, sigma=m_sigma_prior, shape=number_of_time_series)
        return pm.Deterministic(f'{trend_name}_trend', (k + pm.math.dot(A, delta.T)) * x[:, None] + m + pm.math.dot(A, (-s * delta).T))


    def add_fourier_term(
            self,
            x: np.array,
            number_of_fourier_components: int,
            name: str,
            dimension: int,
            seasonality_period_baseline: float,
            relative_uncertainty_factor_prior: float
    ) -> pm.Deterministic:
        with self.model:
            fourier_coefficients = pm.Normal(f'fourier_coefficients_{name}', mu=0, sigma=1,
                                             shape=(2 * number_of_fourier_components, dimension))
            seasonality_period = pm.Gamma(f'season_parameter_{name}',
                                          alpha=relative_uncertainty_factor_prior * seasonality_period_baseline,
                                          beta=relative_uncertainty_factor_prior)
            fourier_features = self.create_fourier_features(x=x, number_of_fourier_components=number_of_fourier_components,
                                                       seasonality_period=seasonality_period)
        return pm.Deterministic(f'{name}_fourier', pm.math.sum(fourier_features * fourier_coefficients[None, :, :], axis=1))


    def create_fourier_features(
            self,
            x: np.array,
            number_of_fourier_components: int,
            seasonality_period: float
    ) -> np.array:        
        frequency_component = pt.tensor.as_tensor_variable(2 * np.pi * (np.arange(number_of_fourier_components)+1) * x[:, None]) # (n_obs, number_of_fourier_components)
        t = frequency_component[:, :, None] / seasonality_period # (n_obs, n_fourier_components, n_time_series)
        return pm.math.concatenate((pt.tensor.cos(t), pt.tensor.sin(t)), axis=1)  # (n_obs, 2*number_of_fourier_components, n_time_series)
    

    def determine_significant_periods(
        self,
        series: np.array,
        x_train: np.array,
        threshold: float
    ) -> Tuple[float, np.array]:
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

        # Find components that are more than K fraction of the maximum
        significant_indices = np.where(positive_magnitudes >= threshold * max_magnitude)[0]
        significant_frequencies = positive_freqs[significant_indices]
        significant_periods = 1 / significant_frequencies
        return dominant_period, significant_periods


    @staticmethod
    def get_default_model_config() -> Dict:

        model_config: Dict = {
            "forecast_horizon": 0.5,
            "target_standard_deviation": 0.01,
            "number_of_individual_trend_changepoints": 20,
            "number_of_individual_fourier_components": 10,
            "period_threshold": 0.5,
            "number_of_shared_seasonality_groups": 3,
            "delta_mu_prior": 0,
            "delta_b_prior": 0.4,
            "m_sigma_prior": 1,
            "k_sigma_prior": 1,
            "relative_uncertainty_factor_prior": 1000

        }
        return model_config


    @staticmethod
    def get_default_sampler_config() -> Dict:

        sampler_config: Dict = {
            "draws": 1000,
            "tune": 200,
            "chains": 1,
            "cores": 1,
            "target_accept": 0.95,
        }
        return sampler_config


    @property
    def output_var(self):
        return "target_distribution"


    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        return self.model_config
    

    def _save_input_params(self, trace) -> None:

        pass

        pass


    def _generate_and_preprocess_model_data(
        self, 
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        self.model_coords = None  # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y


    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        progressbar: bool = True,
        random_seed: pm.util.RandomState = None,
        step: str = "NUTS",
        **kwargs: Any,
    ) -> az.InferenceData:
        self.build_model(X = X, y = y)
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        with self.model:
            if step == "NUTS":
                step=pm.NUTS()
            if step == "HMC":
                step=pm.HamiltonianMC()
            if step == "metropolis":
                step=pm.Metropolis()
            
            sampler_args = {**sampler_config, **kwargs}
            idata_temp = pm.sample(**sampler_args)
        
        self.idata = self.set_idata_attrs(idata_temp)


    def set_idata_attrs(self, idata=None):

        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        idata.attrs["model_config"] = json.dumps(self._serializable_model_config)
        # Only classes with non-dataset parameters will implement save_input_params
        if hasattr(self, "_save_input_params"):
            self._save_input_params(idata)
        return idata
    

    def sample_posterior_predictive(
            self,
            X_pred,
            **kwargs
    ) -> np.ndarray:
        with self.model:  # sample with new input data
            pm.set_data({'input':X_pred})
            self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, **kwargs)
        preds_out_of_sample = self.posterior_predictive.predictions_constant_data.sortby('input')['input']
        model_preds = self.posterior_predictive.predictions.sortby(preds_out_of_sample)

        return preds_out_of_sample, model_preds
    

    def get_posterior_predictive(self) -> az.InferenceData:
        return self.posterior_predictive

    @property
    def id(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(self.model_config.values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]