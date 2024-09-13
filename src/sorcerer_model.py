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
    Union,
    Any
    )

# Import from other modules
from src.config import (
    get_default_model_config,
    get_default_sampler_config,
    serialize_model_config,
)
from src.model_components import (
    add_linear_term,
    add_fourier_term
)
from src.utils import (
    generate_hash_id,
    normalize_training_data
    )


class SorcererModel:

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        model_name: str = "SorcererModel",
        version: str = None
    ):
        # Initialize configurations
        sampler_config = (
            get_default_sampler_config() if sampler_config is None else sampler_config
        )
        self.sampler_config = sampler_config
        model_config = get_default_model_config() if model_config is None else model_config
        self.model_config = model_config  # parameters for priors, etc.
        self.model = None  # Set by build_model
        self.idata: az.InferenceData | None = None  # idata is generated during fitting
        self.posterior_predictive: az.InferenceData
        self.model_name = model_name
        self.version = version
        self.method = "NUTS"
        self.map_estimate = None
        self.x_training_min = None
        self.x_training_max = None
        self.y_training_min = None
        self.y_training_max = None
        
    def build_model(self, X, y, seasonality_periods, **kwargs):
        """
        Builds the PyMC model based on the input data and configuration.
        """
        number_of_time_series = y.shape[1]
        baseline_slope = (y.values[-1] - y.values[0]) / (X.values[-1] - X.values[0])
        baseline_bias = y.values[0]

        # Extract prior parameters from model configuration
        config = self.model_config
        
        # Define PyMC model
        with pm.Model() as self.model:
            x = pm.Data('input', X, dims='number_of_input_observations')
            y = pm.Data('target', y, dims=['number_of_target_observations', 'number_of_time_series'])

            # Add model components
            linear_term = add_linear_term(
                x=x,
                baseline_slope=baseline_slope,
                baseline_bias=baseline_bias,
                trend_name='linear',
                number_of_time_series=number_of_time_series,
                number_of_trend_changepoints=config["number_of_individual_trend_changepoints"],
                maximum_x_value= 1/config["test_train_split"],
                delta_mu_prior=config["delta_mu_prior"],
                delta_b_prior=config["delta_b_prior"],
                m_sigma_prior=config["m_sigma_prior"],
                k_sigma_prior=config["k_sigma_prior"],
                model=self.model
            )
            
            seasonality_individual = pm.math.sum([
                add_fourier_term(
                    x=x,
                    number_of_fourier_components=config["number_of_individual_fourier_components"],
                    name=f'seasonality_individual_{round(seasonality_period_baseline,2)}',
                    dimension=number_of_time_series,
                    seasonality_period_baseline=seasonality_period_baseline,
                    relative_uncertainty_factor_prior=config["relative_uncertainty_factor_prior"],
                    model=self.model
                ) for seasonality_period_baseline in seasonality_periods
                ], axis = 0)

            seasonality_shared = pm.math.sum([
                add_fourier_term(
                    x=x,
                    number_of_fourier_components=config["number_of_shared_fourier_components"],
                    name=f'seasonality_shared_{round(seasonality_period_baseline,2)}',
                    dimension=config["number_of_shared_seasonality_groups"],
                    seasonality_period_baseline=seasonality_period_baseline,
                    relative_uncertainty_factor_prior=config["relative_uncertainty_factor_prior"],
                    model=self.model
                ) for seasonality_period_baseline in seasonality_periods
                ], axis = 0)
                        
            all_models = pm.math.concatenate([x[:, None] * 0, seasonality_shared], axis=1)
            model_probs = pm.Dirichlet('model_probs', a=np.ones(config["number_of_shared_seasonality_groups"]+1), shape=(number_of_time_series, config["number_of_shared_seasonality_groups"]+1))
            chosen_model_index = pm.Categorical('chosen_model_index', p=model_probs, shape=number_of_time_series)
            shared_seasonality_models = all_models[:, chosen_model_index]
            
            prediction = (
                linear_term +
                seasonality_individual +
                shared_seasonality_models
            )
            
            precision_target = pm.Gamma(
            'precision_target_distribution',
            alpha = config["precision_target_distribution_prior_alpha"],
            beta = config["precision_target_distribution_prior_beta"],
            dims = 'number_of_time_series'
            )

            pm.Normal('target_distribution', mu=prediction, sigma=1/pm.math.sqrt(precision_target), observed=y, dims=['number_of_input_observations', 'number_of_time_series'])

    def fit(
        self,
        training_data: pd.DataFrame,
        seasonality_periods: np.array,
        progressbar: bool = True,
        random_seed: pm.util.RandomState = None,
        method: str = "NUTS",
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fits the model to the data using the specified sampler configuration.
        """
        self.method = method
        (
            X,
            self.x_training_min,
            self.x_training_max,
            y,
            self.y_training_min,
            self.y_training_max
            )  = normalize_training_data(training_data = training_data)
        print("Normalized periods:", seasonality_periods*(X[1]-X[0]))
        self.build_model(
            X = X,
            y = y,
            seasonality_periods = seasonality_periods*(X[1]-X[0])
            )
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        with self.model:
            if self.method == "MAP":
                self.map_estimate = [pm.find_MAP()]
            else:
                if self.method == "NUTS":
                    step=pm.NUTS()
                if self.method == "HMC":
                    step=pm.HamiltonianMC()
                if self.method == "metropolis":
                    step=pm.Metropolis()
                idata_temp = pm.sample(step = step, **sampler_config)
                self.idata = self.set_idata_attrs(idata_temp)

    def set_idata_attrs(self, idata=None):
        """
        Sets attributes to the inference data object for identification and metadata.
        """
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
        """
        Samples from the posterior predictive distribution using the fitted model.
        """
        with self.model:
            x_test = (test_data['date'].astype('int64')//10**9 - self.x_training_min)/(self.x_training_max - self.x_training_min)
            pm.set_data({'input': x_test})
            if self.method == "MAP":
                self.posterior_predictive = pm.sample_posterior_predictive(self.map_estimate, predictions=True, **kwargs)
            else:
                self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, **kwargs)
        
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
        """
        Returns the posterior predictive distribution.
        """
        return self.posterior_predictive

    @property
    def id(self) -> str:
        """
        Returns a unique ID for the model instance based on its configuration.
        """
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
        """
        Save the model's inference data to a file.

        Parameters
        ----------
        fname : str
            The name and path of the file to save the inference data with model parameters.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model hasn't been fit yet (no inference data available).

        Examples
        --------
        This method is meant to be overridden and implemented by subclasses.
        It should not be called directly on the base abstract class or its instances.

        >>> class MyModel(ModelBuilder):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>> model = MyModel()
        >>> model.fit(data)
        >>> model.save('model_results.nc')  # This will call the overridden method in MyModel
        """
        
     
        if self.idata is not None and "posterior" in self.idata:
            if self.method == "MAP":
                raise RuntimeError("The MAP method cannot be saved.")
            file = Path(str(fname))
            self.idata.to_netcdf(str(file))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
                

    def load(self, fname: str):
        filepath = Path(str(fname))
        self.idata = az.from_netcdf(filepath)
        # needs to be converted, because json.loads was changing tuple to list
        self.model_config = json.loads(self.idata.attrs["model_config"])
        self.sampler_config = json.loads(self.idata.attrs["sampler_config"])
        
        self.build_model(
            X = self.idata.constant_data.input,
            y = self.idata.constant_data.target
            )
        
