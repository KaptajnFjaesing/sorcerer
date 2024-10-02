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
        self.logger = logging.getLogger("pymc")
        
    def build_model(self, X, Y):        
        with pm.Model() as self.training_model:
            trend_and_fourier_parts = trend_and_fourier_terms(
                    X = X,
                    baseline_slope = self.baseline_slope,
                    baseline_bias = self.baseline_bias,
                    model_config = self.model_config,
                    model = self.training_model
                    )
            if self.model_config['autoregressive_order'] > 0:
                number_of_time_series = self.baseline_slope.shape[0]
                rho = pm.Normal(
                    name = 'rho',
                    mu = self.model_config['rho_mu_prior'],
                    sigma = self.model_config['rho_sigma_prior'],
                    shape = (number_of_time_series, self.model_config['autoregressive_order'])
                    )
                ar_precision_prior = pm.Gamma(
                    name = 'ar_precision_prior',
                    alpha = self.model_config['ar_precision_alpha_prior'],
                    beta = self.model_config['ar_precision_beta_prior']
                    )  
                initial_distribution = pm.Normal.dist(
                    mu = self.model_config['init_mu_prior'],
                    sigma = self.model_config['init_sigma_prior'],
                    shape = (number_of_time_series, self.model_config['autoregressive_order'])
                    )
                autoregressive_part = pm.AR(
                    name = 'autoregressive_part',
                    rho = rho,
                    sigma = 1/ar_precision_prior,
                    init_dist = initial_distribution,
                    constant = False,
                    dims = ['number_of_time_series', 'number_of_observations']
                ).T
            else:
                autoregressive_part = 0
   
            precision_target = pm.Gamma(
                name = 'precision_target',
                alpha = self.model_config["precision_target_distribution_prior_alpha"],
                beta = self.model_config["precision_target_distribution_prior_beta"],
                shape = number_of_time_series
            )
            pm.Normal(
                name = 'target_distribution',
                mu = trend_and_fourier_parts + autoregressive_part,
                sigma = 1/precision_target,
                observed = Y,
                dims=['number_of_observations', 'number_of_time_series']
                )

    def fit(
        self,
        training_data: pd.DataFrame,
        sampler_config: dict | None = None
    ) -> az.InferenceData:
        
        (X,
         self.x_training_min,
         self.x_training_max,
         Y,
         self.y_training_min,
         self.y_training_max,
         self.baseline_slope,
         self.baseline_bias
         )  = normalize_training_data(training_data = training_data)
        self.build_model(X = X, Y = Y)
        self.sampler_config = (get_default_sampler_config() if sampler_config is None else sampler_config)
        
        if not self.sampler_config.get("return_inferencedata", False):
            self.sampler_config['return_inferencedata'] = True
            self.logger.warning("InferenceWarning: return_inferencedata forced to be true")

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
                idata_temp = pm.sample(
                    step = sampler,
                    **{k: v for k, v in self.sampler_config.items() if (k != 'sampler' and k != 'verbose')})
                self.idata = self.set_idata_attrs(idata_temp)

    def set_idata_attrs(self, idata=None):
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
            
            trend_and_fourier_parts = trend_and_fourier_terms(
                    X = X_test,
                    baseline_slope = self.baseline_slope,
                    baseline_bias = self.baseline_bias,
                    model_config = self.model_config,
                    model = self.testing_model
                    )
            if self.model_config['autoregressive_order'] > 0:
                number_of_time_series = self.baseline_slope.shape[0]
                self.testing_model.add_coords({"number_of_observations_minus_one": range(- 1, X_test.shape[0], 1)})
                rho = pm.Normal(
                    name = 'rho',
                    mu = self.model_config['rho_mu_prior'],
                    sigma = self.model_config['rho_sigma_prior'],
                    shape = (number_of_time_series, self.model_config['autoregressive_order'])
                    )
                ar_precision_prior = pm.Gamma(
                    name = 'ar_precision_prior',
                    alpha = self.model_config['ar_precision_alpha_prior'],
                    beta = self.model_config['ar_precision_beta_prior']
                    )  
                initial_distribution = pm.DiracDelta.dist(
                    c = self.idata.posterior['autoregressive_part'].mean(('chain','draw')).values[:,-self.model_config['autoregressive_order']:],
                    shape = (number_of_time_series, self.model_config['autoregressive_order'])
                    )
                autoregressive_part = pm.AR(
                    name = 'autoregressive_part',
                    rho = rho,
                    sigma = 1/ar_precision_prior,
                    init_dist = initial_distribution,
                    constant = False,
                    dims = ['number_of_time_series', 'number_of_observations_minus_one']
                ).T
            else:
                autoregressive_part = 0

            precision_target = pm.Gamma(
                name = 'precision_target',
                alpha = self.model_config["precision_target_distribution_prior_alpha"],
                beta = self.model_config["precision_target_distribution_prior_beta"],
                shape = number_of_time_series
            )
            pm.Normal(
                name = 'predictions',
                mu = trend_and_fourier_parts+autoregressive_part[1:,:],
                sigma = 1/precision_target,
                dims = ['number_of_observations', 'number_of_time_series']
                )
            if self.sampler_config['sampler'] == "MAP":
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        self.map_estimate,
                        predictions=True,
                        progressbar = False,
                        var_names=['predictions'],
                        **kwargs
                        )
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        self.map_estimate,
                        predictions=True,
                        progressbar = True,
                        var_names=['predictions'],
                        **kwargs
                        )
            else:
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        self.idata,
                        predictions=True,
                        progressbar = False,
                        var_names=['predictions'],
                        **kwargs
                        )
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(
                        self.idata,
                        predictions=True,
                        progressbar = True,
                        var_names=['predictions'],
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
        return self.model
    
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