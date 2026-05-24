import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_best_track_data(nc_file):
    """Load best track data."""

    # Read file
    ds = xr.open_dataset(nc_file)

    # Select storm, saved as DataFrame
    storm_id = 295 # Storm Senyar ID
    ds_sel = ds.sel(storm=storm_id)
    df = ds_sel[['lat', 'lon', 'usa_wind', 'usa_pres']].to_dataframe().dropna()

    # Clean DataFrame
    df = df.reset_index(drop=True)
    df['time'] = df['time'].dt.round('s')
    df = df.rename(columns={'usa_wind': 'wind', 'usa_pres': 'min_pressure'})
    return df[::2]

# Create figure with 1 row and 3 columns
font_size = 12
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size
})

#%% USER INPUT

# Track file
trackfile = './data/IBTrACS.last3years.v04r01.nc'
df = load_best_track_data(trackfile)

# ERA5 file
ncfile = "./data/era5_daily_tp/tp_daily_Senyar.nc"

# Defined event box
event_lon_min, event_lon_max = 96, 104
event_lat_min, event_lat_max = 0, 5

# Plotting extent
plot_lon_min, plot_lon_max = 90, 110
plot_lat_min, plot_lat_max = -8, 8

# Time range
time_start = "2025-11-23"
time_end   = "2025-11-27"

#%% OPEN DATA

ds = xr.open_dataset(ncfile)

tp = ds["tp"]

# SELECT TIME ONLY

tp_sel = tp.sel(
    valid_time=slice(time_start, time_end)
)

# Accumulated precipitation (mm)
tp_acc = tp_sel.sum(dim="valid_time") * 1000

#%% PLOT

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot extent
ax.set_extent(
    [plot_lon_min, plot_lon_max,
     plot_lat_min, plot_lat_max],
    crs=ccrs.PlateCarree()
)

# PRECIPITATION

pcm = ax.pcolormesh(
    tp_acc.longitude,
    tp_acc.latitude,
    tp_acc,
    cmap="Blues",
    shading="auto",
    transform=ccrs.PlateCarree(),
    zorder=1,
    vmin=100,
    vmax=500
)


# MASK OCEAN WITH WHITE

ax.add_feature(
    cfeature.OCEAN,
    facecolor="white",
    zorder=2
)

# Land + coastlines
ax.add_feature(
    cfeature.LAND,
    facecolor="none",
    edgecolor="none",
    zorder=3
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=1,
    zorder=4
)

ax.add_feature(
    cfeature.BORDERS,
    linestyle=":",
    zorder=4
)

# CYCLONE TRACK
ax.plot(df['lon'], df['lat'], '--', color='k', label='Cyclone Track')


# EVENT BOX

ax.plot(
    [
        event_lon_min,
        event_lon_max,
        event_lon_max,
        event_lon_min,
        event_lon_min
    ],
    [
        event_lat_min,
        event_lat_min,
        event_lat_max,
        event_lat_max,
        event_lat_min
    ],
    color="red",
    linewidth=2.5,
    label="Defined event",
    transform=ccrs.PlateCarree(),
    zorder=5
)

# Legend
ax.legend(loc="upper right")

# Gridlines
gl = ax.gridlines(
    draw_labels=True,
    linestyle="--",
    alpha=0.5
)

gl.top_labels = False
gl.right_labels = False

# Colorbar
cbar = plt.colorbar(
    pcm,
    ax=ax,
    orientation="vertical",
    pad=0.02
)

cbar.set_label("Accumulated rainfall (mm)")

# Title
plt.title(
    "Rainfall during Cyclone Senyar (ERA5)\n"
    "23–27 November 2025"
)

plt.savefig('./output/project/domain.png')