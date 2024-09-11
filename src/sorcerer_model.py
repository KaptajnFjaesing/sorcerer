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
    determine_significant_periods
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
        
    def build_model(self, X, y, **kwargs):
        """
        Builds the PyMC model based on the input data and configuration.
        """
        number_of_time_series = y.shape[1]
        baseline_slope = (y.values[-1] - y.values[0]) / (X.values[-1] - X.values[0])
        baseline_bias = y.values[0]

        # Extract prior parameters from model configuration
        config = self.model_config
        (
            dominant_period,
            significant_periods,
        ) = determine_significant_periods(
            series=y.values[:, 0],
            x_train=X.values,
            threshold=config["period_threshold"],
        )
        seasonality_period_baseline = min(significant_periods)

        # Define PyMC model
        with pm.Model() as self.model:
            x = pm.Data('input', X, dims='number_of_input_observations')
            y = pm.Data('target', y, dims=['number_of_target_observations', 'number_of_time_series'])

            # Add model components
            linear_term1 = add_linear_term(
                x=x,
                baseline_slope=baseline_slope,
                baseline_bias=baseline_bias,
                trend_name='linear1',
                number_of_time_series=number_of_time_series,
                number_of_trend_changepoints=config["number_of_individual_trend_changepoints"],
                maximum_x_value= 1/config["test_train_split"],
                delta_mu_prior=config["delta_mu_prior"],
                delta_b_prior=config["delta_b_prior"],
                m_sigma_prior=config["m_sigma_prior"],
                k_sigma_prior=config["k_sigma_prior"],
                model=self.model
            )

            seasonality_individual1 = add_fourier_term(
                x=x,
                number_of_fourier_components=config["number_of_individual_fourier_components"],
                name='seasonality_individual1',
                dimension=number_of_time_series,
                seasonality_period_baseline=seasonality_period_baseline,
                relative_uncertainty_factor_prior=config["relative_uncertainty_factor_prior"],
                model=self.model
            )

            seasonality_shared = add_fourier_term(
                x=x,
                number_of_fourier_components=config["number_of_shared_fourier_components"],
                name='seasonality_shared',
                dimension=config["number_of_shared_seasonality_groups"],
                seasonality_period_baseline=seasonality_period_baseline,
                relative_uncertainty_factor_prior=config["relative_uncertainty_factor_prior"],
                model=self.model
            )

            all_models = pm.math.concatenate([x[:, None] * 0, seasonality_shared], axis=1)
            model_probs = pm.Dirichlet('model_probs', a=np.ones(config["number_of_shared_seasonality_groups"]+1), shape=(number_of_time_series, config["number_of_shared_seasonality_groups"]+1))
            chosen_model_index = pm.Categorical('chosen_model_index', p=model_probs, shape=number_of_time_series)
            shared_seasonality_models = all_models[:, chosen_model_index]
            
            prediction = (
                linear_term1 +
                seasonality_individual1 +
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
        X: pd.DataFrame,
        y: pd.DataFrame,
        progressbar: bool = True,
        random_seed: pm.util.RandomState = None,
        step: str = "NUTS",
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fits the model to the data using the specified sampler configuration.
        """
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
        X_pred,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples from the posterior predictive distribution using the fitted model.
        """
        with self.model:
            pm.set_data({'input': X_pred})
            self.posterior_predictive = pm.sample_posterior_predictive(self.idata, predictions=True, **kwargs)
        
        preds_out_of_sample = self.posterior_predictive.predictions_constant_data.sortby('input')['input']
        model_preds = self.posterior_predictive.predictions.sortby(preds_out_of_sample)

        return preds_out_of_sample, model_preds

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
        
