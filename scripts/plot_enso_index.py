import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Read data
ds = xr.open_dataset('./data/enso34/enso34_index.nc')

# Read date bounds
time1, time2 = ds['valid_time'].data[0], ds['valid_time'].data[-1]

# Plot index
fig, ax = plt.subplots()
ds['sst'].plot(ax=ax)

# Add baseline at 0 anomalies
plt.hlines(0, time1, time2, color='k')

# Decoration and save output
plt.xlim(time1, time2)
plt.xlabel('Month')
plt.ylabel('Niño 3.4 Index')
plt.suptitle('Data: ERA5 monthly average, Baseline: 1950-1979')
plt.title('Methodology following NCAR technical note')
plt.savefig('./output/enso34_index.png')