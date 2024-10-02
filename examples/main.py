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
model_version = "v0.5.0"
forecast_horizon = 30

training_data = df.iloc[:-forecast_horizon]
test_data = df.iloc[-forecast_horizon:]

# Sorcerer
sampler_config = {
    "draws": 200,
    "tune": 50,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True,
    "return_inferencedata": True
}

number_of_weeks_in_a_year = 52.1429

model_config = {
    "number_of_individual_trend_changepoints": 40,
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 10,
    "precision_target_distribution_prior_beta": 0.1,
    "prior_probability_shared_seasonality_alpha": 1,
    "prior_probability_shared_seasonality_beta": 1,
    "autoregressive_order": 1,
    "rho_mu_prior": 1,
    "rho_sigma_prior": 1,
    "ar_precision_alpha_prior": 100,
    "ar_precision_beta_prior": 0.1,
    "init_mu_prior": 1,
    "init_sigma_prior": 1,
    
        
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
    model_version = model_version
    )


# %% Fit model

sorcerer.fit(
    training_data = training_data,
    sampler_config = sampler_config
    )

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



#%%

model_preds = sorcerer.sample_posterior_predictive(test_data = test_data)

#%% Plot forecast along with test data
(X_train, y_train, X_test, y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )
column_names = [x for x in df.columns if 'HOUSEHOLD' in x]

hdi_values = az.hdi(model_preds)["predictions"].transpose("hdi", ...)

n_cols = 2
n_rows = int(np.ceil(len(column_names) / n_cols))
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(column_names)):
    ax = axs[i]
    ax.plot(X_train, y_train[y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, y_test[y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(X_test, (model_preds["predictions"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        X_test,
        hdi_values[0].values[:,i],
        hdi_values[1].values[:,i],
        color= 'blue',
        alpha=0.4
    )
    ax.set_title(column_names[i])
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()
    ax.set_xlim([-0.05,max(X_test)+0.1])

plt.savefig('./examples/figures/forecast.png')