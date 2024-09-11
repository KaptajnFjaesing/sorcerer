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

n_weeks = 52
normalized_column_group = [x for x in df.columns if '_normalized' in x ]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

training_data = df.iloc[:-n_weeks]
test_data = df.iloc[-n_weeks:]

# Feature engineering
x_train = (training_data['date'].astype('int64')//10**9 - (training_data['date'].astype('int64')//10**9).min())/((training_data['date'].astype('int64')//10**9).max() - (training_data['date'].astype('int64')//10**9).min())
y_train = (training_data[unnormalized_column_group]-training_data[unnormalized_column_group].min())/(training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min())

x_test = (test_data['date'].astype('int64')//10**9 - (training_data['date'].astype('int64')//10**9).min())/((training_data['date'].astype('int64')//10**9).max() - (training_data['date'].astype('int64')//10**9).min())
y_test = (test_data[unnormalized_column_group]-training_data[unnormalized_column_group].min())/(training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min())

x_total = (df['date'].astype('int64')//10**9 - (training_data['date'].astype('int64')//10**9).min())/((training_data['date'].astype('int64')//10**9).max() - (training_data['date'].astype('int64')//10**9).min())
y_total = (df[unnormalized_column_group]-training_data[unnormalized_column_group].min())/(training_data[unnormalized_column_group].max()-training_data[unnormalized_column_group].min())
# %% Define model

model_name = "SorcererModel"
version = "v0.1"

sampler_config = {
    "draws": 2000,
    "tune": 200,
    "chains": 4,
    "cores": 1
}

model_config = {
    "test_train_split": len(training_data)/len(df),
    "number_of_individual_trend_changepoints": 20,
    "number_of_individual_fourier_components": 10,
    "number_of_shared_fourier_components": 5,
    "period_threshold": 0.5,
    "number_of_shared_seasonality_groups": 1,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.2,
    "m_sigma_prior": 1,
    "k_sigma_prior": 1,
    "precision_target_distribution_prior_alpha": 2,
    "precision_target_distribution_prior_beta": 0.1,
    "relative_uncertainty_factor_prior": 1000
}

sorcerer = SorcererModel(
    sampler_config = sampler_config,
    model_config = model_config,
    model_name = model_name,
    version = version
    )


# %% Fit model
sorcerer.fit(
    X = x_train,
    y = y_train,
    step="NUTS"
    )

fname = "examples/models/sorcer_model_v01.nc"
sorcerer.save(fname)
#%% Produce forecast

"""
Load from stored model
fname = "examples/models/sorcer_model_v01.nc"
sorcerer.load(fname)
"""

(preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(X_pred = x_total)

#%% Plot forecast along with test data

hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil(y_test.shape[1] / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(y_test.shape[1]):
    ax = axs[i]  # Get the correct subplot
    ax.plot(x_total, y_total[y_total.columns[i]], color = 'tab:red',  label='Data')
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

plt.savefig('./examples/figures/forecast.pdf')
