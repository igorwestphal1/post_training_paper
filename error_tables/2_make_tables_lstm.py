
# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

#%% 

# inputs
os.chdir(r"G:\My Drive\7_load_forecasting\3_errors")
mapes = pd.read_csv("2_output\mapes_lstm.csv")
mapes['datetime'] = pd.to_datetime(mapes['datetime'])
mapes = mapes.set_index('datetime')

#%%

mapes.dropna(axis=0, inplace=True)

#%%

stats = (mapes[['0.5','0.8','1.0','1.2', '1.5', 'naive', 'avg']].abs().describe())
stats.to_csv('2_output/error_stats_lstm.csv')

#%%

mapes['hour2'] = (mapes.index + pd.to_timedelta(mapes['hours_ahead'].astype(int), unit='H'))
mapes['hour2'] = mapes['hour2'].dt.hour

#%% mapes

# get stats by hour
stats_by_hour = (mapes[['0.5','0.8','1.0','1.2', '1.5','naive','avg']].abs().groupby(mapes['hour2']).describe())
sort_cols = stats_by_hour.sort_index(axis=1,level=[1,0]).columns
stats_by_hour = stats_by_hour.loc[:,sort_cols]

# keep = (stats_by_hour.columns.to_flat_index().str.contains('min', regex=False)  |
#         stats_by_hour.columns.to_flat_index().str.contains('25%', regex=False)  | 
#         stats_by_hour.columns.to_flat_index().str.contains('50%', regex=False)  | 
#         stats_by_hour.columns.to_flat_index().str.contains('75%', regex=False)  | 
#         stats_by_hour.columns.to_flat_index().str.contains('max', regex=False)
#         )
# stats_by_hour = stats_by_hour.loc[:,keep]

# output
stats_by_hour.to_csv('2_output/stats_by_hour_lstm.csv')

#%% plot stats by hour

a = stats_by_hour.filter(like='mean')
a.columns = a.columns.get_level_values(0)
plt.figure(figsize=(4,4))
plt.plot(a['naive'], label='Naive', linestyle='solid', linewidth = 1, color = 'black')
# plt.plot(a['0.5'], alpha = 0.5)
# plt.plot(a['0.8'], alpha = 0.5)
plt.plot(a['1.0'], label='LSTM', linestyle='solid', linewidth = 1, color = 'blue')
# plt.plot(a['1.2'], alpha = 0.5)
# plt.plot(a['1.5'], alpha = 0.5)
plt.plot(a['avg'], label='LSTM+PTS', linestyle='solid', linewidth = 1, color = 'red')
plt.ylim(3.5,8.5)
plt.xlabel('Hour of the Day')
plt.ylabel('MAPE (%)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
plt.grid(True)
plt.tight_layout()
plt.savefig('2_output/mape_per_hour_lstm.pdf')

#%% stats by load quantiles

# set load quantiles
mapes['decile'] = pd.qcut(mapes['actual load'], q = 10, labels = False) + 1

# get stats by load quantiles
stats_by_decile = (mapes[['0.5','0.8','1.0','1.2', '1.5', 'naive', 'avg']].abs().groupby(mapes['decile']).describe())

# group stats by statistic instead of the model
sort_cols = stats_by_decile.sort_index(axis=1,level=[1,0]).columns
stats_by_decile = stats_by_decile.loc[:,sort_cols]

# output
stats_by_decile.to_csv('2_output/stats_by_decile_lstm.csv')

#%% 
a = stats_by_decile.filter(like='mean')
a.columns = a.columns.get_level_values(0)
plt.figure(figsize=(4,4))
plt.plot(a['naive'], label='Naive', linestyle='solid', linewidth = 1, color = 'black')
# plt.plot(a['0.5'], label='0.5', alpha = 0.5)
# plt.plot(a['0.8'], label='0.8', alpha = 0.5)
plt.plot(a['1.0'], label='LSTM', linestyle='solid', linewidth = 1, color = 'blue')
# plt.plot(a['1.2'], label='1.2', alpha = 0.5)
# plt.plot(a['1.5'], label='1.5', alpha = 0.5)
plt.plot(a['avg'], label='LSTM+PTS', linestyle='solid', linewidth = 1, color = 'red')
plt.ylim(3,7)
plt.xlabel('Load Decile')
plt.ylabel('MAPE (%)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
plt.grid(True)
plt.tight_layout()
plt.savefig('2_output/mape_per_decile_lstm.pdf')

#%%
mapes['avg_forecast'] = (100 - mapes['avg']) * mapes['actual load'] / 100

#%%
fig1 = plt.figure(figsize=(6,4))
plt.hist2d(
    x=mapes['actual load'], 
    y=mapes['avg_forecast'], 
    cmap='Reds', 
    bins=(80,80),
    range=[[8000, 20000], [8000, 20000]])
plt.xlabel('Real Load (MW)')
plt.ylabel('Forecasted Load (MW)')
plt.colorbar()
plt.show()
fig1.savefig('2_output/comparison_lstm.pdf')

#%%