#%%
"""
Created on Mon Sep  9 20:22:19 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from examples.load_data import normalized_weekly_store_category_household_sales
from src.sorcerer_model import SorcererModel

df = normalized_weekly_store_category_household_sales()

# %% Define model

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]

df_time_series = df[time_series_columns]

model_name = "SorcererModel"
version = "v0.1"
forecast_horizon = 52

training_data = df_time_series.iloc[:-forecast_horizon]
test_data = df_time_series.iloc[-forecast_horizon:]

# Sorcerer
sampler_config = {
    "draws": 500,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS"
}

model_config = {
    "number_of_individual_trend_changepoints": 4,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.2,
    "m_sigma_prior": 0.5,
    "k_sigma_prior": 0.5,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 5,
    "precision_target_distribution_prior_alpha": 2,
    "precision_target_distribution_prior_beta": 0.1,
    "relative_uncertainty_factor_prior": 1000,
    "probability_to_include_shared_seasonality_prior": 0.5,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': 52,'number_of_fourier_components': 4}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': 52,'number_of_fourier_components': 4},
        {'seasonality_period_baseline': 4,'number_of_fourier_components': 1}
    ]
}

if sampler_config['sampler'] == "MAP":
    model_config['precision_target_distribution_prior_alpha'] = 100
    model_config['precision_target_distribution_prior_beta'] = 0.1

sorcerer = SorcererModel(
    model_config = model_config,
    model_name = model_name,
    sampler_config = sampler_config,
    version = version
    )


# %% Fit model

sorcerer.fit(training_data = training_data)
"""
if sampler_config["sampler"] != "MAP":
    fname = "examples/models/sorcer_model_v02.nc"
    sorcerer.save(fname)
"""
#%% Produce forecast

"""
Load from stored model
fname = "examples/models/sorcer_model_v01.nc"
sorcerer.load(fname)
"""

(preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(test_data = df_time_series)

#%% Plot forecast along with test data
(X_train, y_train, X_test, y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )

hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(time_series_columns)-1):
    ax = axs[i]  # Get the correct subplot
    
    ax.plot(X_train, y_train[y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, y_test[y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(preds_out_of_sample, (model_preds["target_distribution"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        preds_out_of_sample.values,
        hdi_values[0].values[:,i],  # lower bound of the HDI
        hdi_values[1].values[:,i],  # upper bound of the HDI
        color= 'blue',   # color of the shaded region
        alpha=0.4,      # transparency level of the shaded region
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure

# %%
plt.savefig('./examples/figures/forecast.png')
