
# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

#%% 

# inputs
os.chdir(r"G:\My Drive\7_load\3_errors")
gru_05 = pd.read_csv("1_input\gru_custom_loss_0.5_penalty_results.csv").set_index('datetime')
gru_08 = pd.read_csv("1_input\gru_custom_loss_0.8_penalty_results.csv").set_index('datetime')
gru_10 = pd.read_csv("1_input\gru_custom_loss_1.0_penalty_results.csv").set_index('datetime')
gru_12 = pd.read_csv("1_input\gru_custom_loss_1.2_penalty_results.csv").set_index('datetime')
gru_15 = pd.read_csv("1_input\gru_custom_loss_1.5_penalty_results.csv").set_index('datetime')

#%% calculate the mapes

# 
test = gru_10.filter(like='actual')

# fitted values
gru_05 = gru_05.filter(like='fit')
gru_08 = gru_08.filter(like='fit')
gru_10 = gru_10.filter(like='fit')
gru_12 = gru_12.filter(like='fit')
gru_15 = gru_15.filter(like='fit')

# ensemble
mix = (gru_05 + gru_08 + gru_10 + gru_12 + gru_15) / 5

# naive
naive = test.shift(24)

#%% mapes

# mape values
mape_gru_05 = -1 * ((test - gru_05.values) / test.values) * 100
mape_gru_08 = -1 * ((test - gru_08.values) / test.values) * 100
mape_gru_10 = -1 * ((test - gru_10.values) / test.values) * 100
mape_gru_12 = -1 * ((test - gru_12.values) / test.values) * 100
mape_gru_15 = -1 * ((test - gru_15.values) / test.values) * 100
mape_mix     = -1 * ((test - mix.values)     / test.values) * 100
mape_naive   = -1 * ((test - naive.values)   / test.values) * 100

#%%
# mape gru
mape_gru_05 = mape_gru_05.melt(
    ignore_index=False, 
    value_vars = mape_gru_05.columns,
    var_name = 'hours_ahead', 
    value_name= '0.5')

# mape gru
mape_gru_08 = mape_gru_08.melt(
    ignore_index=False, 
    value_vars = mape_gru_08.columns,
    var_name = 'hours_ahead', 
    value_name= '0.8')

# mape gru
mape_gru_10 = mape_gru_10.melt(
    ignore_index=False, 
    value_vars = mape_gru_10.columns,
    var_name = 'hours_ahead', 
    value_name= '1.0')

# mape gru
mape_gru_12 = mape_gru_12.melt(
    ignore_index=False, 
    value_vars = mape_gru_12.columns,
    var_name = 'hours_ahead', 
    value_name= '1.2')

# mape gru
mape_gru_15 = mape_gru_15.melt(
    ignore_index=False, 
    value_vars = mape_gru_15.columns,
    var_name = 'hours_ahead', 
    value_name= '1.5')

# mape gru
mape_mix = mape_mix.melt(
    ignore_index=False, 
    value_vars = mape_mix.columns,
    var_name = 'hours_ahead', 
    value_name= 'avg')

# actual values
test = test.melt(
    ignore_index=False, 
    value_vars = test.columns, 
    var_name = 'hours_ahead', 
    value_name= 'actual load'
    )

# mape dnn
naive = mape_naive.melt(
    ignore_index=False, 
    value_vars = mape_naive.columns,
    var_name = 'hours_ahead', 
    value_name= 'naive')

# merge
result = mape_gru_05
result = result.merge(mape_gru_08, on=['datetime','hours_ahead'])
result = result.merge(mape_gru_10, on=['datetime','hours_ahead'])
result = result.merge(mape_gru_12, on=['datetime','hours_ahead'])
result = result.merge(mape_gru_15, on=['datetime','hours_ahead'])
result = result.merge(naive, on=['datetime','hours_ahead'])
result = result.merge(mape_mix, on=['datetime','hours_ahead'])
result = result.merge(test, on=['datetime','hours_ahead'])

# fix hours ahead
result['hours_ahead'] = result['hours_ahead'].str.strip('actual_').astype(int)

#%% add time dummies

# function to return a season
def get_season(month):
    season = np.zeros_like(month, dtype='U6')  # create an empty array of strings of the same shape as the input
    season[np.isin(month, [12, 1, 2])] = 'winter'
    season[np.isin(month, [3, 4, 5])] = 'spring'
    season[np.isin(month, [6, 7, 8])] = 'summer'
    season[np.isin(month, [9, 10, 11])] = 'fall'
    season[~np.isin(month, np.arange(1, 13))] = 'invalid month'
    return season

# add dummies for weekdays, weekends, month, season
result.index = pd.to_datetime(result.index)
result['hour']     = result.index.hour
result['day']     = result.index.day
result['month']    = result.index.month
result['year']     = result.index.year
result['day_name'] = result.index.day_name()
result['weekend']  = np.isin(result['day_name'], ['Saturday','Sunday'])
result['season']   = get_season(result.index.month)

#%%

result.sort_values(by=['year','month','day','hour'], inplace=True)

#%%

# output:
result.to_csv('2_output/mapes_gru.csv')

#%% eof
