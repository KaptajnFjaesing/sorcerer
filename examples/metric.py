#%%
"""
Created on Mon Sep  9 20:22:19 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
from examples.load_data import normalized_weekly_store_category_household_sales
from src.sorcerer_model import SorcererModel

df = normalized_weekly_store_category_household_sales()

# %% Define model


time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

model_name = "SorcererModel"
version = "v0.1"
method = "MAP"
seasonality_periods = np.array([52])


#%%

min_forecast_horizon = 26
max_forecast_horizon = 52
model_forecasts = []
for forecast_horizon in range(min_forecast_horizon,52+1):
    training_data = df_time_series.iloc[:-forecast_horizon]
    test_data = df_time_series.iloc[-forecast_horizon:]
    
    y_train_min = training_data[unnormalized_column_group].min()
    y_train_max = training_data[unnormalized_column_group].max()
    
    model_config = {
        "test_train_split": len(training_data)/len(df_time_series),
        "number_of_individual_trend_changepoints": 20,
        "number_of_individual_fourier_components": 10,
        "number_of_shared_fourier_components": 5,
        "period_threshold": 0.5,
        "number_of_shared_seasonality_groups": 2,
        "delta_mu_prior": 0,
        "delta_b_prior": 0.3,
        "m_sigma_prior": 1,
        "k_sigma_prior": 1,
        "precision_target_distribution_prior_alpha": 100,
        "precision_target_distribution_prior_beta": 0.05,
        "relative_uncertainty_factor_prior": 1000
    }
    sorcerer = SorcererModel(
        model_config = model_config,
        model_name = model_name,
        version = version
        )
    sorcerer.fit(
        training_data = training_data,
        seasonality_periods = seasonality_periods,
        method = method
        )
    (preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(test_data = test_data)
    model_forecasts.append([pd.Series((model_preds["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]]) for i in range(len(time_series_columns)-1)])


# %%
from src.utils import compute_residuals

test_data = df.iloc[-max_forecast_horizon:].reset_index(drop = True)

stacked = compute_residuals(
         model_forecasts = model_forecasts,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )   

stacked.to_pickle(r'C:\Users\roman\Documents\git\TimeSeriesForecastingReview\data\results\stacked_forecasts_sorcerer.pkl')