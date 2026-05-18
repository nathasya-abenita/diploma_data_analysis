import cdsapi

# Define region of interest - Sumatra: [96°E–104°E, 0°N–5°N]
bbox = [5, 96, 0, 104] # Global extent: [90, -180, -90, 180] - [north, west, south, east]

# Dataset name
dataset = "derived-era5-single-levels-daily-statistics"

# Define access to Climate Data Store API
client = cdsapi.Client()

# Define years and loop over each year
year_list = range(2025, 2025+1)
for year in year_list:
    print(f'handling year of {year}...')
    # Define request form
    request = {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "daily_statistic": "daily_sum",
        "time_zone": "utc+07:00",
        "frequency": "1_hourly",
        "area": bbox
    }

    # Send request
    client.retrieve(dataset, request).download()
