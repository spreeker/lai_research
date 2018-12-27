# LAI_thesis

Code for research on Leaf Area index.

The goal from this code base is to create global maps
of the most influential climate variables on vegetation.

Also combinations of climate variables are used.

We combine nasa data of LAI (MOD15A2) with nasa landuse data (MCD12Q1)
and global climate varables CRU (cru_ts3) form 2000 to 2010.

code to download and convert datasets should be all there.

Code is very much research and experimental code.

Software engineering. S.J Preeker.


Thesis
======

plot_basemap_climatic_variable_CRU.py
--------------------------

The code does two things. First, it plots time series of climatic variable for a specific location. Secondly the same climatic variable at the global map scale. It works with monthly time series.

The file has to have this structure: nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{nc_var}.dat.nc','r')

You have to fill in three variables:
1. startyear: start year of your data
2. endyear: end year of your data
3. nc_variable: shortcut for the climatic variable (pre/vap/tmp/pet)

-----------------------------

time_series.py
-----------------------------

The code makes a time series plot of the LAI from specific location. It is based on LAT and LON

You have to fill in three variables:
1. directory of the folder with LAI data (hdf4 files)
2. LAT and LON of the location
3. For hdf modis files to convert lat,lon to x,y in the 1200x1200 grid the origin coordinates of the measurement, we need to provide pixel size, correct projection. The pixelsize and projection is probably already correct but the origin needs to be specified for each file.

---------------------------

plot_hdf5_MODIS.py
---------------------------

The code makes a map of the LAI for the specific tile.
