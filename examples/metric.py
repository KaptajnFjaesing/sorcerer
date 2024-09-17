#%%
"""
Created on Mon Sep  9 20:22:19 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
from examples.load_data import normalized_weekly_store_category_household_sales
from src.sorcerer_model import SorcererModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = normalized_weekly_store_category_household_sales()

# %% Define model

time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]

df_time_series = df[time_series_columns]

model_name = "SorcererModel"
version = "v0.1"
seasonality_periods = np.array([52])


def exponential_smoothing(
        df: pd.DataFrame,
        time_series_column: str,
        seasonality_period: int,
        forecast_periods: int
        ):
    
    df_temp = df.copy()
    df_temp.set_index('date', inplace=True)

    # Extract the time series data
    time_series = df_temp[time_series_column]
    
    # Fit Exponential Smoothing Model
    model = ExponentialSmoothing(
        time_series, 
        seasonal='add',  # Use 'add' for additive seasonality or 'mul' for multiplicative seasonality
        trend='add',      # Use 'add' for additive trend or 'mul' for multiplicative trend
        seasonal_periods=seasonality_period,  # Adjust for the frequency of your seasonal period (e.g., 12 for monthly data)
        freq=time_series.index.inferred_freq
    )
    fit_model = model.fit()
    return fit_model.forecast(steps=forecast_periods)


min_forecast_horizon = 26
max_forecast_horizon = 52
model_forecasts_sorcerer = []
model_forecasts_exponential = []
for forecast_horizon in range(min_forecast_horizon,max_forecast_horizon+1):
    training_data = df_time_series.iloc[:-forecast_horizon]
    test_data = df_time_series.iloc[-forecast_horizon:]
    
    y_train_min = training_data[unnormalized_column_group].min()
    y_train_max = training_data[unnormalized_column_group].max()
    
    # Sorcerer
    sampler_config = {
        "draws": 500,
        "tune": 100,
        "chains": 1,
        "cores": 1,
        "sampler": "NUTS"
    }

    model_config = {
        "test_train_split": len(training_data)/len(df_time_series),
        "number_of_individual_trend_changepoints": 10,
        "number_of_individual_fourier_components": 10,
        "number_of_shared_fourier_components": 10,
        "number_of_shared_seasonality_groups": 4,
        "delta_mu_prior": 0,
        "delta_b_prior": 0.2,
        "m_sigma_prior": 1,
        "k_sigma_prior": 1,
        "precision_target_distribution_prior_alpha": 2,
        "precision_target_distribution_prior_beta": 0.1,
        "relative_uncertainty_factor_prior": 1000
    }
    
    if sampler_config['sampler'] == "MAP":
        model_config['precision_target_distribution_prior_alpha'] = 100
        model_config['precision_target_distribution_prior_beta'] = 0.01
        
    sorcerer = SorcererModel(
        model_config = model_config,
        model_name = model_name,
        sampler_config = sampler_config,
        version = version
        )
    sorcerer.fit(
        training_data = training_data,
        seasonality_periods = seasonality_periods
        )
    (preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(
        test_data = test_data,
        progressbar = False)
    
    y_train = (training_data[unnormalized_column_group]-y_train_min)/(y_train_max-y_train_min)
    y_train['date'] = training_data['date']
    model_denormalized = []
    for i in range(len(unnormalized_column_group)):
        model_results = exponential_smoothing(
                df = y_train,
                time_series_column = unnormalized_column_group[i],
                seasonality_period = 52,
                forecast_periods = forecast_horizon
                )
        model_denormalized.append(model_results*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]])
    model_forecasts_exponential.append(model_denormalized)
    model_forecasts_sorcerer.append([pd.Series((model_preds["target_distribution"].mean(("chain", "draw")).T)[i].values*(y_train_max[y_train_max.index[i]]-y_train_min[y_train_min.index[i]])+y_train_min[y_train_min.index[i]]) for i in range(len(unnormalized_column_group))])



#%% Compute MASEs
import matplotlib.pyplot as plt
from src.utils import compute_residuals

abs_mean_gradient_training_data = pd.read_pickle('./data/abs_mean_gradient_training_data.pkl')

stacked_sorcerer = compute_residuals(
         model_forecasts = model_forecasts_sorcerer,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )

average_abs_residual_sorcerer = stacked_sorcerer.abs().groupby(stacked_sorcerer.index).mean() # averaged over rolls
average_abs_residual_sorcerer.columns = unnormalized_column_group
MASE_sorcerer = average_abs_residual_sorcerer/abs_mean_gradient_training_data

print("avg MASE_sorcerer: ", MASE_sorcerer.mean(axis = 1).mean())

stacked_exponential_smoothing = compute_residuals(
         model_forecasts = model_forecasts_exponential,
         test_data = test_data[unnormalized_column_group],
         min_forecast_horizon = min_forecast_horizon
         )

average_abs_residual_exponential_smoothing = stacked_exponential_smoothing.abs().groupby(stacked_exponential_smoothing.index).mean() # averaged over rolls
average_abs_residual_exponential_smoothing.columns = unnormalized_column_group
MASE_exponential_smoothing = average_abs_residual_exponential_smoothing/abs_mean_gradient_training_data

print("avg MASE_exponential_smoothing: ", MASE_exponential_smoothing.mean(axis = 1).mean())

# Calculate the number of rows needed for 2 columns
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed

# Create subplots with 2 columns and computed rows   
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten the axs array to iterate over it easily
axs = axs.flatten()

# Loop through each column to plot
for i in range(len(unnormalized_column_group)):
    ax = axs[i]  # Get the correct subplot
    ax.plot(MASE_sorcerer[unnormalized_column_group[i]], color = 'tab:blue',  label='MASE Sorcerer')
    ax.plot(MASE_exponential_smoothing[unnormalized_column_group[i]], color = 'tab:red',  label='MASE Exp Smoothing')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.set_title(unnormalized_column_group[i])
    ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])  # Remove unused axes to clean up the figure


#%%
stacked_sorcerer.to_pickle(r'C:\Users\roman\Documents\git\TimeSeriesForecastingReview\data\results\stacked_forecasts_sorcerer.pkl')


#%% Compare exponential smoothing to sorcerer
import matplotlib.pyplot as plt

iteration = -25
horizon = forecast_horizon+iteration+1

exp_forecast = model_forecasts_exponential[iteration]
sorcerer_forecast = model_forecasts_sorcerer[iteration]
n_cols = 2  # We want 2 columns
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))  # Number of rows needed

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(unnormalized_column_group)):
    ax = axs[i]  # Get the correct subplot
    ax.plot(df_time_series['date'].iloc[:-horizon], df_time_series[unnormalized_column_group][unnormalized_column_group[i]].iloc[:-horizon], color = 'black',  label='Training Data')
    ax.plot(df_time_series['date'].iloc[-horizon:], df_time_series[unnormalized_column_group][unnormalized_column_group[i]].iloc[-horizon:], color = 'gray',  label='Test Data')
    ax.plot(df_time_series.iloc[-forecast_horizon:]['date'].iloc[-horizon:-horizon+min_forecast_horizon],exp_forecast[i][:min_forecast_horizon], color = 'tab:red', label='Exponential Smoothing Model')
    ax.plot(df_time_series.iloc[-forecast_horizon:]['date'].iloc[-horizon:-horizon+min_forecast_horizon],sorcerer_forecast[i][:min_forecast_horizon], color = 'tab:blue', label='Sorcerer Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.set_title(time_series_columns[i])
    ax.legend()