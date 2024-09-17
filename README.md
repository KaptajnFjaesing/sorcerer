# Sorcerer

**Sorcerer** is a hierarchical Bayesian Generalized Additive Model (GAM) for time series forecasting, inspired by [timeseers](https://github.com/MBrouns/timeseers) and the [PyMC model builder class](https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html). Like **timeseers**, **Sorcerer** builds upon the ideas from Facebook's [Prophet](https://facebook.github.io/prophet/), aiming to provide a Bayesian framework for forecasting multiple time series with shared parameters simultaneously.

However, **Sorcerer** extends beyond what existing tools offer by focusing on two key enhancements:

1. **Version Control and Compatibility for Multivariate Models**: Sorcerer introduces a model version control approach inspired by PyMC's model builder, but adapted to work with the multivariate time series cases that PyMC's original class does not handle.
   
2. **Novel Approach to Shared Seasonalities**: Sorcerer takes a novel approach to shared seasonalities, in which there is a user-specified set of shared seasonalities (no seasonality is one element of that set) that each time series can opt into. The shared seasonalities are learned from data, with only the number set by the user.  

By combining these innovations, **Sorcerer** aims to offer a more comprehensive and scalable solution for time series forecasting, while retaining the interpretability and flexibility of a Bayesian framework.


# Usage

## Load data
```python
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from src.load_data import normalized_weekly_store_category_household_sales
from src.sorcerer_model2 import SorcererModel

df = normalized_weekly_store_category_household_sales()

```

## Define model
```python
time_series_columns = [x for x in df.columns if ('HOUSEHOLD' in x and 'normalized' not in x) or ('date' in x)]
df_time_series = df[time_series_columns]

model_name = "SorcererModel"
version = "v0.1"
forecast_horizon = 45

training_data = df_time_series.iloc[:-forecast_horizon]
test_data = df_time_series.iloc[-forecast_horizon:]

# Sorcerer
sampler_config = {
    "draws": 2000,
    "tune": 500,
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
```

## Fit model
```python
seasonality_periods = np.array([52])

sorcerer.fit(
    training_data = training_data,
    seasonality_periods = seasonality_periods
    )

```

```python
Sequential sampling (1 chains in 1 job)
CompoundStep
>NUTS: [linear_k, linear_delta, linear_m, fourier_coefficients_seasonality_individual_0.32, season_parameter_seasonality_individual_0.32, fourier_coefficients_seasonality_shared_0.32, season_parameter_seasonality_shared_0.32, model_probs, precision_target_distribution]
>CategoricalGibbsMetropolis: [chosen_model_index]
Sampling chain 0, 0 divergences ------------------------ 100% 0:00:00 / 0:23:22
[?25hSampling 1 chain for 500 tune and 2_000 draw iterations (500 + 2_000 draws total) took 1402 seconds.
Chain 0 reached the maximum tree depth. Increase `max_treedepth`, increase `target_accept` or reparameterize.
Only one chain was sampled, this makes it impossible to run some convergence checks
```

## Produce forecasts
```python
(preds_out_of_sample, model_preds) = sorcerer.sample_posterior_predictive(test_data = df_time_series)
```

```python
Sampling: [target_distribution]
Sampling ... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 / 0:00:02
```

## Plot forecasts along with test and training data
```python
(X_train, y_train, X_test, y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )

hdi_values = az.hdi(model_preds)["target_distribution"].transpose("hdi", ...)
n_cols = 2
n_rows = int(np.ceil((len(time_series_columns)-1) / n_cols))
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)
axs = axs.flatten()
for i in range(len(time_series_columns)-1):
    ax = axs[i]  
    ax.plot(X_train, y_train[y_train.columns[i]], color = 'tab:red',  label='Training Data')
    ax.plot(X_test, y_test[y_test.columns[i]], color = 'black',  label='Test Data')
    ax.plot(preds_out_of_sample, (model_preds["target_distribution"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
    ax.fill_between(
        preds_out_of_sample.values,
        hdi_values[0].values[:,i],
        hdi_values[1].values[:,i], 
        color= 'blue', 
        alpha=0.4,    
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.grid(True)
    ax.legend()

for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j]) 
```

![Forecasts](examples/figures/forecast.png)