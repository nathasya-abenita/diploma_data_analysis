import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#%% Read data and initialize figure

# Read data
ds = xr.open_dataset('./data/enso34/enso34_index.nc')
ds_anom = xr.open_dataset('./data/enso34/era5_monthly_sst_enso34_anom.nc')
ds_smooth_anom = xr.open_dataset('./data/enso34/era5_monthly_sst_enso34_smooth_anom.nc')


# Read date bounds
time1, time2 = ds['valid_time'].data[0], ds['valid_time'].data[-1]

# Initialize axes
fig, axs = plt.subplots(2, 1, figsize=(6, 9))

#%% Plot anomalies

ds_anom['sst'].plot(ax=axs[0], label='anomalies')
ds_smooth_anom['sst'].plot(ax=axs[0], label='5-months running mean')

# Add baseline at 0 anomalies
axs[0].hlines(0, time1, time2, color='k')

# Decoration
axs[0].set_xlim(time1, time2)
axs[0].set_ylim(-3, 3.5)
axs[0].set_xlabel('')
axs[0].set_ylabel('deg')
leg = axs[0].legend(ncol=2)
leg.get_frame().set_alpha(0.1)
axs[0].set_title('Anomalies compared to baseline')


#%% Plot final index
ds['sst'].plot(ax=axs[1])

# Add baseline at 0 anomalies
axs[1].hlines(0, time1, time2, color='k')

# Decoration
axs[1].set_xlim(time1, time2)
axs[1].set_xlabel('Month')
axs[1].set_ylabel('Niño 3.4 Index')
axs[1].set_title('Smoothed anomalies divided by its baseline standard deviation')

#%% Superior title and save figure
plt.suptitle('Data: ERA5 monthly average, Baseline: 1950-1979\nMethodology following NCAR technical note')
plt.savefig('./output/enso34_index.png')