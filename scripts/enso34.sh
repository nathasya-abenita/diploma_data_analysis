#!/bin/bash

# Methodology from NCAR
# Source: https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni

# Nino X Index computation: 
# (a) Compute area averaged total SST from Niño X region; 
# (b) Compute monthly climatology (e.g., 1950-1979) for area averaged total SST from Niño X region, 
# and subtract climatology from area averaged total SST time series to obtain anomalies; 
# (c) Smooth the anomalies with a 5-month running mean; 
# (d) Normalize the smoothed values by its standard deviation over the climatological period.

# Define paths and file name
datadir=data
outdir=data/enso34
fname=era5_monthly_sst

# Define ENSO3.5 region
region="-170,-120,-5,5" # lon1,lon2,lat1,lat2
region_tag="enso34"

# Define baseline year
year1=1950
year2=1979
yearend=2026

# Cut area to region
ifile=${datadir}/${fname}.nc
cutfile=${outdir}/${fname}_${region_tag}.nc
echo cdo sellonlatbox,${region} $ifile $cutfile

# Slice year
finfile=${outdir}/${fname}_${region_tag}_${year1}_${year2}.nc
cdo selyear,${year1}/${yearend} ${cutfile} ${finfile}

# Compute area average time series
areafile=${outdir}/${fname}_${region_tag}_fldmean.nc
cdo fldmean ${finfile} ${areafile}

# Compute baseline climatology
basefile=${outdir}/${fname}_${region_tag}_base.nc
cdo ymonmean -selyear,${year1},${year2} ${areafile} ${basefile}

# Compute anomalies
anomfile=${outdir}/${fname}_${region_tag}_anom.nc
cdo ymonsub ${areafile} ${basefile} ${anomfile}

# Smooth the anomalies with a 5-month running mean
smoothfile=${outdir}/${fname}_${region_tag}_smooth_anom.nc
cdo runmean,5 ${anomfile} ${smoothfile}

#  Normalize the smoothed values by its standard deviation over the climatological period
stdfile=${outdir}/${fname}_${region_tag}_std.nc
indexfile=${outdir}/${region_tag}_index.nc
cdo timstd -selyear,${year1}/${year2} ${smoothfile} ${stdfile}
cdo div ${smoothfile} ${stdfile} ${indexfile}