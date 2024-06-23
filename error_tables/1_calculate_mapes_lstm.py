
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
lstm_05 = pd.read_csv("1_input\lstm_custom_loss_0.5_penalty_results.csv").set_index('datetime')
lstm_08 = pd.read_csv("1_input\lstm_custom_loss_0.8_penalty_results.csv").set_index('datetime')
lstm_10 = pd.read_csv("1_input\lstm_results.csv").set_index('datetime')
lstm_12 = pd.read_csv("1_input\lstm_custom_loss_1.2_penalty_results.csv").set_index('datetime')
lstm_15 = pd.read_csv("1_input\lstm_custom_loss_1.5_penalty_results.csv").set_index('datetime')

#%% calculate the mapes

# 
test = lstm_10.filter(like='actual')

# fitted values
lstm_05 = lstm_05.filter(like='fit')
lstm_08 = lstm_08.filter(like='fit')
lstm_10 = lstm_10.filter(like='fit')
lstm_12 = lstm_12.filter(like='fit')
lstm_15 = lstm_15.filter(like='fit')

# ensemble
mix = (lstm_05 + lstm_08 + lstm_10 + lstm_12 + lstm_15) / 5

# naive
naive = test.shift(24)

#%% mapes

# mape values
mape_lstm_05 = -1 * ((test - lstm_05.values) / test.values) * 100
mape_lstm_08 = -1 * ((test - lstm_08.values) / test.values) * 100
mape_lstm_10 = -1 * ((test - lstm_10.values) / test.values) * 100
mape_lstm_12 = -1 * ((test - lstm_12.values) / test.values) * 100
mape_lstm_15 = -1 * ((test - lstm_15.values) / test.values) * 100
mape_mix     = -1 * ((test - mix.values)     / test.values) * 100
mape_naive   = -1 * ((test - naive.values)   / test.values) * 100

#%%
# mape lstm
mape_lstm_05 = mape_lstm_05.melt(
    ignore_index=False, 
    value_vars = mape_lstm_05.columns,
    var_name = 'hours_ahead', 
    value_name= '0.5')

# mape lstm
mape_lstm_08 = mape_lstm_08.melt(
    ignore_index=False, 
    value_vars = mape_lstm_08.columns,
    var_name = 'hours_ahead', 
    value_name= '0.8')

# mape lstm
mape_lstm_10 = mape_lstm_10.melt(
    ignore_index=False, 
    value_vars = mape_lstm_10.columns,
    var_name = 'hours_ahead', 
    value_name= '1.0')

# mape lstm
mape_lstm_12 = mape_lstm_12.melt(
    ignore_index=False, 
    value_vars = mape_lstm_12.columns,
    var_name = 'hours_ahead', 
    value_name= '1.2')

# mape lstm
mape_lstm_15 = mape_lstm_15.melt(
    ignore_index=False, 
    value_vars = mape_lstm_15.columns,
    var_name = 'hours_ahead', 
    value_name= '1.5')

# mape lstm
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
result = mape_lstm_05
result = result.merge(mape_lstm_08, on=['datetime','hours_ahead'])
result = result.merge(mape_lstm_10, on=['datetime','hours_ahead'])
result = result.merge(mape_lstm_12, on=['datetime','hours_ahead'])
result = result.merge(mape_lstm_15, on=['datetime','hours_ahead'])
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
result.to_csv('2_output/mapes_lstm.csv')

#%% eof
