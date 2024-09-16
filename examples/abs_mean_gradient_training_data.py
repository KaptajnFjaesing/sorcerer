# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:27:40 2024

@author: petersen.jonas
"""

from src.utils import normalized_weekly_store_category_household_sales

df = normalized_weekly_store_category_household_sales()

max_forecast_horizon = 52
unnormalized_column_group = [x for x in df.columns if 'HOUSEHOLD' in x and 'normalized' not in x]
abs_mean_gradient_training_data = df.iloc[:-max_forecast_horizon].reset_index(drop = True)[unnormalized_column_group].diff().dropna().abs().mean(axis = 0)

abs_mean_gradient_training_data.to_pickle('./data/abs_mean_gradient_training_data.pkl')


