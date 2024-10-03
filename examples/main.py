#%%
"""
Created on Mon Sep  9 20:22:19 2024

@author: Jonas Petersen
"""
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sorcerer.sorcerer_model import SorcererModel
from examples.load_data import load_m5_weekly_store_category_sales_data

_,df,_ = load_m5_weekly_store_category_sales_data()

nan_count = [0, 102, 73, 17, 180, 42, 9, 4, 0, 8]
time_series_column_group = [x for x in df.columns if 'HOUSEHOLD' in x]

# Assign NaNs to the start of each column dynamically
for col, n in zip(time_series_column_group, nan_count):
    df.loc[:n-1, col] = np.nan
#%%
model_name = "SorcererModel"
model_version = "v0.4.2"
forecast_horizon = 30

training_data = df.iloc[:-forecast_horizon]
test_data = df.iloc[-forecast_horizon:]

# Sorcerer
sampler_config = {
    "draws": 2000,
    "tune": 500,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True,
    "nuts_sampler": "numpyro"
}

number_of_weeks_in_a_year = 52.1429

model_config = {
    "number_of_individual_trend_changepoints": int(len(training_data)/number_of_weeks_in_a_year),
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 50,
    "precision_target_distribution_prior_beta": 0.1,
    "single_scale_mu_prior": 0,
    "single_scale_sigma_prior": 1,
    "shared_scale_mu_prior": 1,
    "shared_scale_sigma_prior": 1,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 20}
    ],
    "shared_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 10},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/4,'number_of_fourier_components': 5},
        {'seasonality_period_baseline': number_of_weeks_in_a_year/12,'number_of_fourier_components': 1},
    ]
}

sorcerer = SorcererModel(
    model_config = model_config,
    model_name = model_name,
    model_version = model_version
    )


# %% Fit model

sorcerer.fit(
    training_data = training_data,
    sampler_config = sampler_config
    )

#%% Model predictions

model_preds = sorcerer.sample_posterior_predictive(test_data = test_data)

#%% Plot forecast along with test data
(X_train, Y_train, X_test, Y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )

hdi_values = az.hdi(model_preds)["predictions"].transpose("hdi", ...)

n_cols = 2
n_rows = int(np.ceil(len(time_series_column_group) / n_cols))
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(time_series_column_group)):
    ax = axs[i]
    ax.plot(X_train, Y_train[Y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, Y_test[Y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(X_test, (model_preds["predictions"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        X_test,
        hdi_values[0].values[:,i],
        hdi_values[1].values[:,i],
        color= 'blue',
        alpha=0.4
    )
    ax.set_title(time_series_column_group[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend(loc = 'center left')
    ax.set_xlim([-0.05,max(X_test)+0.1])

plt.savefig('./examples/figures/forecast.png')

# %% Plot model components for training data

X_train, trend, seasonality_individual, shared_seasonality = sorcerer.get_mean_model_components()

n_cols = 2
n_rows = int(np.ceil(len(time_series_column_group) / n_cols))
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(27, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(time_series_column_group)):
    ax = axs[i]
    ax.plot(X_train, Y_train[Y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_train, trend[:,i], color = 'tab:blue',  label='Trend')
    ax.plot(X_train, seasonality_individual[:,i], color = 'tab:green',  label='Individual Seasonality')
    ax.plot(X_train, shared_seasonality[:,i], color = 'tab:orange',  label='Shared Seasonality')
    ax.plot(X_train, trend[:,i]+seasonality_individual[:,i]+shared_seasonality[:,i], color = 'black',  label='Complete Fit')
    
    ax.set_title(time_series_column_group[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend(loc = 'center right')
    ax.set_xlim([-0.05,max(X_test)+0.1])

plt.savefig('./examples/figures/model_components.png')

#%% Plot weight contributions

idata = sorcerer.get_idata()
individual_weights = idata.posterior["single_scale"].mean(('chain','draw')).values
shared_weights = idata.posterior["shared_scale"].mean(('chain','draw')).values

plt.figure(figsize = (20,10),constrained_layout=True)
plt.scatter(individual_weights, time_series_column_group, marker = 'o', label = "Individual seasonality weights (period mixture)")

for i, shared in enumerate(shared_weights):
    plt.scatter(shared, time_series_column_group, marker = 'o', label = f"Shared seasonality weights (period {round(model_config['shared_fourier_terms'][i]['seasonality_period_baseline'],2)})")

plt.xlabel('Scale')
plt.ylabel('Time series')
plt.grid(visible=True, which='both', linewidth=0.6, color='gray', alpha=0.7)
plt.legend()

plt.savefig('./examples/figures/weight_contributions.png')