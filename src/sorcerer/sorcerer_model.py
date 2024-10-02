"""
Created on Mon Sep  9 19:49:06 2024

@author: Jonas Petersen
"""

import numpy as np
from pathlib import Path
import json
import pymc as pm
import pandas as pd
import arviz as az
from typing import (
    Dict,
    Tuple,
    Union
    )
import logging

from sorcerer.config import (
    get_default_model_config,
    get_default_sampler_config,
    serialize_model_config,
)

from sorcerer.model_components import trend_and_fourier_terms

from sorcerer.utils import (
    generate_hash_id,
    normalize_training_data
    )

class SorcererModel:

    def __init__(
        self,
        model_config: dict | None = None,
        model_name: str = "SorcererModel",
        model_version: str = None
    ):
        self.sampler_config = None
        self.model_config = (get_default_model_config() if model_config is None else model_config)
        self.training_model = None
        self.testing_model = None
        self.idata: az.InferenceData | None = None
        self.posterior_predictive: az.InferenceData
        self.model_name = model_name
        self.model_version = model_version
        self.map_estimate = None
        self.x_training_min = None
        self.x_training_max = None
        self.y_training_min = None
        self.y_training_max = None
        self.baseline_slope = None
        self.baseline_bias = None
        self.X = None
        self.Y = None
        self.logger = logging.getLogger("pymc")
        
    def build_model(self, X, Y):        
        with pm.Model() as self.training_model:
            target_mean = trend_and_fourier_terms(
                    X = X,
                    baseline_slope = self.baseline_slope,
                    baseline_bias = self.baseline_bias,
                    model_config = self.model_config,
                    model = self.training_model
                    )
            precision_target = pm.Gamma(
                name = 'precision_target_distribution',
                alpha = self.model_config["precision_target_distribution_prior_alpha"],
                beta = self.model_config["precision_target_distribution_prior_beta"],
                dims = 'number_of_time_series'
            )
            pm.Normal(
                name = 'target_distribution',
                mu = target_mean,
                sigma = 1/precision_target,
                observed = Y,
                dims = ['number_of_observations', 'number_of_time_series']
                )

    def fit(
        self,
        training_data: pd.DataFrame,
        sampler_config: dict | None = None
    ) -> az.InferenceData:
        
        (self.X,
         self.x_training_min,
         self.x_training_max,
         self.Y,
         self.y_training_min,
         self.y_training_max,
         self.baseline_slope,
         self.baseline_bias
         )  = normalize_training_data(training_data = training_data)
        self.build_model(X = self.X, Y = self.Y)
        self.sampler_config = (get_default_sampler_config() if sampler_config is None else sampler_config)
        if not self.sampler_config['verbose']:
            self.logger.setLevel(logging.CRITICAL)
            self.sampler_config['progressbar'] = False
        else:
            self.logger.setLevel(logging.INFO)
            self.sampler_config['progressbar'] = True
        
        with self.training_model:
            if self.sampler_config['sampler'] == "MAP":
                self.map_estimate = [pm.find_MAP(progressbar = self.sampler_config['progressbar'])]
            else:
                if self.sampler_config['sampler'] == "NUTS":
                    sampler=pm.NUTS()
                if self.sampler_config['sampler'] == "HMC":
                    sampler=pm.HamiltonianMC()
                if self.sampler_config['sampler'] == "metropolis":
                    sampler=pm.Metropolis()
                idata_temp = pm.sample(step = sampler, **{k: v for k, v in self.sampler_config.items() if (k != 'sampler' and k != 'verbose')})
                self.idata = self.set_idata_attrs(idata_temp)

    def set_idata_attrs(self, idata = None):
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_name"] = self.model_name
        idata.attrs["model_version"] = self.model_version
        idata.attrs["sampler_config"] = serialize_model_config(self.sampler_config)
        idata.attrs["model_config"] = serialize_model_config(self._serializable_model_config)
        return idata

    def sample_posterior_predictive(
        self,
        test_data,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_test = (test_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
        with pm.Model() as self.testing_model:
            target_mean = trend_and_fourier_terms(
                    X = X_test,
                    baseline_slope = self.baseline_slope,
                    baseline_bias = self.baseline_bias,
                    model_config = self.model_config,
                    model = self.testing_model
                    )
            precision_target = pm.Gamma(
                name = 'precision_target_distribution',
                alpha = self.model_config["precision_target_distribution_prior_alpha"],
                beta = self.model_config["precision_target_distribution_prior_beta"],
                dims = 'number_of_time_series'
            )
            pm.Normal(
                name = 'predictions',
                mu = target_mean,
                sigma = 1/precision_target,
                dims = ['number_of_observations', 'number_of_time_series']
                )
            if self.sampler_config['sampler'] == "MAP":
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        trace = self.map_estimate,
                        predictions = True,
                        progressbar = False,
                        var_names = ['predictions'],
                        **kwargs
                        )
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        trace = self.map_estimate,
                        predictions = True,
                        progressbar = True,
                        var_names = ['predictions'],
                        **kwargs
                        )
            else:
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        trace = self.idata,
                        predictions = True,
                        progressbar = False,
                        var_names = ['predictions'],
                        **kwargs
                        )
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        trace = self.idata,
                        predictions = True,
                        progressbar = True,
                        var_names = ['predictions'],
                        **kwargs
                        )

        return self.posterior_predictive.predictions
    
    def normalize_data(self,
                       training_data,
                       test_data
                       )-> tuple:
        if self.x_training_min is not None:
            time_series_columns = [x for x in training_data.columns if 'date' not in x]
            X_train = (training_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            y_train = (training_data[time_series_columns]-self.y_training_min)/(self.y_training_max-self.y_training_min)
            X_test = (test_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            y_test = (test_data[time_series_columns]-self.y_training_min)/(self.y_training_max-self.y_training_min)
            return (
                X_train,
                y_train,
                X_test,
                y_test
                )
        else:
            raise RuntimeError("Data can only be normalized after .fit() has been called.")
            return None
    
    def get_posterior_predictive(self) -> az.InferenceData:
        return self.posterior_predictive

    @property
    def id(self) -> str:
        return generate_hash_id(self.model_config, self.model_version, self.model_name)

    @property
    def output_var(self):
        return "target_distribution"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        return self.model_config
    
    def get_model(self):
        return self.training_model
    
    def get_idata(self):
        return self.idata
    
    def save(self, fname: str) -> None:
        if self.idata is not None and "posterior" in self.idata:
            if self.sampler_config['sampler'] == "MAP":
                raise RuntimeError("The MAP method cannot be saved.")
            file = Path(str(fname))
            self.idata.to_netcdf(str(file))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
                
    def load(self, fname: str):
        filepath = Path(str(fname))
        self.idata = az.from_netcdf(filepath)
        self.model_config = json.loads(self.idata.attrs["model_config"])
        self.sampler_config = json.loads(self.idata.attrs["sampler_config"])
        self.build_model(
            X = self.idata.constant_data.input,
            y = self.idata.constant_data.target
            )
    
    def get_mean_model_components(self):
        s = np.linspace(0, 1, self.model_config["number_of_individual_trend_changepoints"] + 2)[1:-1] # max(x) for input is by definition 1
        A = (self.X.values[:, np.newaxis] > s) * 1.
        mean_k = self.idata.posterior['trend_k'].mean(('chain','draw')).values
        mean_delta = self.idata.posterior['trend_delta'].mean(('chain','draw')).values
        mean_m = self.idata.posterior['trend_m'].mean(('chain','draw')).values
        
        trend = (mean_k+ np.dot(A, mean_delta.T)) * self.X.values[:, np.newaxis] + mean_m + np.dot(A, (-s * mean_delta).T)
        
        seasonality_individual = np.zeros((self.X.shape[0], self.Y.shape[1]))
        for term in self.model_config["individual_fourier_terms"]:
            dX = self.X.iloc[1]-self.X.iloc[0]
            frequency_component = 2 * np.pi * (np.arange(term['number_of_fourier_components']) + 1) * self.X.values[:, np.newaxis]
            t = frequency_component[:, :, None] / (term['seasonality_period_baseline'] * dX)
            fourier_features = np.concatenate((np.cos(t), np.sin(t)), axis=1)
            fourier_coefficients = self.idata.posterior[f"fourier_coefficients_{round(term['seasonality_period_baseline'] * dX, 2)}_{term['number_of_fourier_components']}"].mean(('chain','draw')).values
            seasonality_individual += (fourier_features * fourier_coefficients[None, :, :]).sum(axis = 1)
        
        seasonality_individual = seasonality_individual*self.idata.posterior["single_scale"].mean(('chain','draw')).values
        
        shared_seasonalities = []
        for term in self.model_config["shared_fourier_terms"]:
            dX = self.X.iloc[1]-self.X.iloc[0]
            frequency_component = 2 * np.pi * (np.arange(term['number_of_fourier_components']) + 1) * self.X.values[:, np.newaxis]
            t = frequency_component[:, :, None] / (term['seasonality_period_baseline'] * dX)
            fourier_features = np.concatenate((np.cos(t), np.sin(t)), axis=1)
            fourier_coefficients = self.idata.posterior[f"fourier_coefficients_shared_{round(term['seasonality_period_baseline'] * dX, 2)}_{term['number_of_fourier_components']}"].mean(('chain','draw')).values
            shared_seasonalities.append( (fourier_features * fourier_coefficients[None, :, :]).sum(axis = 1))
        shared_seasonalities = np.concatenate(shared_seasonalities, axis=1)
        shared_seasonality = np.dot(shared_seasonalities, self.idata.posterior["shared_scale"].mean(('chain','draw')).values)
        
        return self.X, trend, seasonality_individual, shared_seasonality