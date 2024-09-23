#%%
"""
Created on Mon Sep  9 20:22:19 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from src.sorcerer.sorcerer_model import SorcererModel
from examples.load_data import load_m5_weekly_store_category_sales_data

_,df,_ = load_m5_weekly_store_category_sales_data()


#%%
model_name = "SorcererModel"
version = "v0.3"
forecast_horizon = 30

training_data = df.iloc[:-forecast_horizon]
test_data = df.iloc[-forecast_horizon:]

# Sorcerer
sampler_config = {
    "draws": 500,
    "tune": 100,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True
}

number_of_weeks_in_a_year = 52.1429

model_config = {
    "number_of_individual_trend_changepoints": 20,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 2,
    "precision_target_distribution_prior_beta": 3,
    "prior_probability_shared_seasonality_alpha": 1,
    "prior_probability_shared_seasonality_beta": 1,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 20}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 10},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/4,'number_of_fourier_components': 1},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/12,'number_of_fourier_components': 1},
    ]
}

sorcerer = SorcererModel(
    model_config = model_config,
    model_name = model_name,
    sampler_config = sampler_config,
    version = version
    )


# %% Fit model

sorcerer.fit(training_data = training_data)

# %%
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

(preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(test_data = test_data)

#%% Plot forecast along with test data
(X_train, y_train, X_test, y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )
column_names = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x)]

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
    ax.set_title(column_names[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure

plt.savefig('./examples/figures/forecast.png')


# %%
idata = sorcerer.get_idata()
print(dir(idata.posterior))
print()
print(idata.posterior.dims)

# %% Test if the periods used are similar to the prescribed ones
import pymc as pm

pm.plot_trace(idata, var_names = [x for x in dir(idata.posterior) if 'fourier_coefficients_seasonality' in x and 'dim' not in x])
pm.plot_trace(idata, var_names = [x for x in dir(idata.posterior) if 'linear' in x and 'dim' not in x])
pm.plot_trace(idata, var_names = ['prior_probability_shared_seasonality','precision_target_distribution'])


# %% Plot trend, seasonality ect


import pandas as pd

x_total = pd.concat([X_train,X_test])


s = np.linspace(0, 1, model_config['number_of_individual_trend_changepoints'] + 2)[1:-1] # max(x) for input is by definition 1
A = (X_train.values[:, None] > s) * 1.

linear_k = idata.posterior['linear_k'].mean(('chain','draw')).values
linear_m = idata.posterior['linear_m'].mean(('chain','draw')).values
linear_delta = idata.posterior['linear_delta'].mean(('chain','draw')).values

trend = (linear_k + np.dot(A, linear_delta.T)) * X_train.values[:, None] + linear_m + np.dot(A, (-s * linear_delta).T)

A2 = (x_total.values[:, None] > s) * 1.
trend2 = (linear_k + np.dot(A2, linear_delta.T)) * x_total.values[:, None] + linear_m + np.dot(A2, (-s * linear_delta).T)

plt.figure()
plt.title("Trends")
for i in range(len(column_names)):
    plt.plot(trend[:,i], label = column_names[i])
plt.legend()


for y in :
    
    season_parameter_seasonality_individual_52 = idata.posterior[y].mean(('chain','draw')).values
    fourier_coefficients_seasonality_individual_52 = idata.posterior[''].mean(('chain','draw')).values
    frequency_component = 2 * np.pi * (np.arange(model_config['individual_fourier_terms'][0]['number_of_fourier_components']) + 1) * X_train.values[:, None]
    t = frequency_component[:, :, None] / season_parameter_seasonality_individual_52  # Normalize by the period
    fourier_features = np.concatenate((np.cos(t), np.sin(t)), axis=1)
    individual_seasonality_52 = np.sum(fourier_features * fourier_coefficients_seasonality_individual_52[None, :, :] , axis =1)

season_parameter_seasonality_shared_52 = idata.posterior['season_parameter_seasonality_shared_52'].mean(('chain','draw')).values
fourier_coefficients_seasonality_shared_52 = idata.posterior['fourier_coefficients_seasonality_shared_52'].mean(('chain','draw')).values
frequency_component = 2 * np.pi * (np.arange(model_config['shared_fourier_terms'][0]['number_of_fourier_components']) + 1) * X_train.values[:, None]
t = frequency_component[:, :, None] / season_parameter_seasonality_shared_52  # Normalize by the period
fourier_features = np.concatenate((np.cos(t), np.sin(t)), axis=1)
shared_seasonality_52 =np.sum(fourier_features * fourier_coefficients_seasonality_shared_52[None, :, :] , axis =1)

season_parameter_seasonality_shared_13 = idata.posterior['season_parameter_seasonality_shared_13'].mean(('chain','draw')).values
fourier_coefficients_seasonality_shared_13 = idata.posterior['fourier_coefficients_seasonality_shared_13'].mean(('chain','draw')).values
frequency_component = 2 * np.pi * (np.arange(model_config['shared_fourier_terms'][1]['number_of_fourier_components']) + 1) * X_train.values[:, None]
t = frequency_component[:, :, None] / season_parameter_seasonality_shared_13  # Normalize by the period
fourier_features = np.concatenate((np.cos(t), np.sin(t)), axis=1)
shared_seasonality_13 =np.sum(fourier_features * fourier_coefficients_seasonality_shared_13[None, :, :] , axis =1)


plt.figure()
plt.title("Shared seasonality period 52")
plt.plot(shared_seasonality_52[:,0][np.argmin(shared_seasonality_52[:,0]):np.argmin(shared_seasonality_52[:,0])+52])
plt.plot(shared_seasonality_13[:,0][np.argmin(shared_seasonality_13[:,0]):np.argmin(shared_seasonality_13[:,0])+13], 'o-')

plt.figure()
plt.title("Individual seasonalities")
for i in range(len(column_names)):
    plt.plot(individual_seasonality_52[:,i], label = column_names[i])
plt.legend()