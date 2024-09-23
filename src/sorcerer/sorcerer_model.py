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
from sorcerer.model_components import (
    add_linear_term,
    add_fourier_term
)
from sorcerer.utils import (
    generate_hash_id,
    normalize_training_data
    )


class SorcererModel:

    def __init__(
        self,
        model_config: dict | None = None,
        model_name: str = "SorcererModel",
        version: str = None,
    ):
        self.sampler_config = None
        self.model_config = (get_default_model_config() if model_config is None else model_config)
        self.model = None
        self.idata: az.InferenceData | None = None
        self.posterior_predictive: az.InferenceData
        self.model_name = model_name
        self.version = version
        self.map_estimate = None
        self.x_training_min = None
        self.x_training_max = None
        self.y_training_min = None
        self.y_training_max = None
        self.logger = logging.getLogger("pymc")
        
    def build_model(self, X, y):
        number_of_time_series = y.shape[1]
        baseline_slope = (y.values[-1] - y.values[0]) / (X.values[-1] - X.values[0])
        baseline_bias = y.values[0]
        
        with pm.Model() as self.model:
            x = pm.Data('input', X, dims='number_of_input_observations')
            y = pm.Data('target', y, dims=['number_of_target_observations', 'number_of_time_series'])

            linear_term = add_linear_term(
                x=x,
                baseline_slope=baseline_slope,
                baseline_bias=baseline_bias,
                trend_name='linear',
                number_of_time_series=number_of_time_series,
                number_of_trend_changepoints=self.model_config["number_of_individual_trend_changepoints"],
                delta_mu_prior=self.model_config["delta_mu_prior"],
                delta_b_prior=self.model_config["delta_b_prior"],
                m_sigma_prior=self.model_config["m_sigma_prior"],
                k_sigma_prior=self.model_config["k_sigma_prior"],
                model=self.model
            )
            
            if len(self.model_config["individual_fourier_terms"]) > 0:
                seasonality_individual = pm.math.sum([
                    add_fourier_term(
                        x=x,
                        number_of_fourier_components= term['number_of_fourier_components'],
                        name=f"seasonality_individual_{round(term['seasonality_period_baseline'],2)}",
                        dimension=number_of_time_series,
                        seasonality_period_baseline=term['seasonality_period_baseline']*(X[1]-X[0]),
                        model=self.model,
                        fourier_sigma_prior = self.model_config["fourier_sigma_prior"],
                        fourier_mu_prior = self.model_config["fourier_mu_prior"]
                    ) for term in self.model_config["individual_fourier_terms"]
                    ], axis = 0)
            else:
                self.warning.info("No individual seasonalities included. If desired, specifications must be added to the model_config.")

            if len(self.model_config["shared_fourier_terms"]) > 0:
                shared_seasonalities = pm.math.concatenate([
                    add_fourier_term(
                        x=x,
                        number_of_fourier_components=term['number_of_fourier_components'],
                        name=f"seasonality_shared_{round(term['seasonality_period_baseline'], 2)}",
                        dimension=1,
                        seasonality_period_baseline=term['seasonality_period_baseline'] * (X[1] - X[0]),
                        model=self.model,
                        fourier_sigma_prior=self.model_config["fourier_sigma_prior"],
                        fourier_mu_prior=self.model_config["fourier_mu_prior"]
                    ) for term in self.model_config["shared_fourier_terms"]
                ], axis=1)
                
                prior_probability_shared_seasonality = pm.Beta(
                    'prior_probability_shared_seasonality',
                    alpha=self.model_config["prior_probability_shared_seasonality_alpha"],
                    beta=self.model_config["prior_probability_shared_seasonality_beta"],
                    shape = number_of_time_series
                    )
                include_seasonality = pm.Bernoulli(
                    'include_seasonality',
                    p=prior_probability_shared_seasonality,
                    shape=(len(self.model_config["shared_fourier_terms"]), number_of_time_series)
                )
                shared_seasonality = pm.math.dot(shared_seasonalities, include_seasonality)
            else:
                self.warning.info("No shared seasonalities included. If desired, specifications must be added to the model_config.")
                shared_seasonality = 0
            
            target_mean = (
                linear_term +
                seasonality_individual +
                shared_seasonality
            )
            
            precision_target = pm.Gamma(
            'precision_target_distribution',
            alpha = self.model_config["precision_target_distribution_prior_alpha"],
            beta = self.model_config["precision_target_distribution_prior_beta"],
            dims = 'number_of_time_series'
            )
            pm.Normal('target_distribution', mu=target_mean, sigma=1/precision_target, observed=y, dims=['number_of_input_observations', 'number_of_time_series'])

    def fit(
        self,
        training_data: pd.DataFrame,
        sampler_config: dict | None = None
    ) -> az.InferenceData:
        
        (
            X,
            self.x_training_min,
            self.x_training_max,
            y,
            self.y_training_min,
            self.y_training_max
            )  = normalize_training_data(training_data = training_data)
        self.build_model(X = X,y = y)
        self.sampler_config = (get_default_sampler_config() if self.sampler_config is None else self.sampler_config)
        if not self.sampler_config['verbose']:
            self.logger.setLevel(logging.CRITICAL)
            self.sampler_config['progressbar'] = False
        else:
            self.logger.setLevel(logging.INFO)
            self.sampler_config['progressbar'] = True
        
        with self.model:
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

    def set_idata_attrs(self, idata=None):
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_name"] = self.model_name
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = serialize_model_config(self.sampler_config)
        idata.attrs["model_config"] = serialize_model_config(self._serializable_model_config)
        return idata

    def sample_posterior_predictive(
        self,
        test_data,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self.model:
            x_test = (test_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            pm.set_data({'input': x_test})
            if self.sampler_config['sampler'] == "MAP":
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.map_estimate, predictions=True, progressbar = False, **kwargs)
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.map_estimate, predictions=True, progressbar = True, **kwargs)
            else:
                if not self.sampler_config['verbose']:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, progressbar = False, **kwargs)
                else:
                    self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, progressbar = True, **kwargs)
        
        preds_out_of_sample = self.posterior_predictive.predictions_constant_data.sortby('input')['input']
        model_preds = self.posterior_predictive.predictions.sortby(preds_out_of_sample)
        return preds_out_of_sample, model_preds
    
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
        return generate_hash_id(self.model_config, self.version, self.model_name)

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