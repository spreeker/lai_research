""""
The file to describe all the settings used in the files.
"""
import os

PROJECT_ROOT = os.path.dirname(__file__)

WORLDMAP = 'world.hdf5'

# This configuration object is dynamicaly updated when running
# worldmap.py.  'groupname' will be modis tile. h18v03 for example.
# hdf5 storage will contain all prepared data for modis tile
conf = {
    #  groupname of five locations
    # 'groupname': "mala",
    # 'groupname': "mexico",
    # 'groupname': "usa",
    'groupname': "amazon",
    # 'groupname': "german_forest",

    'hdf5storage': 'lai_cube6.hdf5',

    # time settings
    'startyear': 2001,
    'endyear': 2010,
    # 'ncvar': 'tmp',
    'time_idx': 42,
    # some random month counting from the start year (for the CRU map)
    'v006': False,
    'startmonth': 0,
    'endmonth': 120,
    # automatic runs save somewhere else
    'csv': 'result-run.csv',
    # 'greenrange': [0,  2]    # 1
    'greenrange': [0,  9],     # 2
    # 'greenrange': [2,  4]
    # 'greenrange': [3,  5]
    # 'greenrange': [4,  6]
    # 'world': 'world_run_ten.nc'
    'world': 'jolly_maps_17.nc'
}

# locations used for local analysis / plots / debug
locations = {
    # location: lat lon
    'mala': {'lat': 4.819, 'lon': 102.500},
    'h28v08': {'lat': 4.819, 'lon': 102.500},

    'german_forest': {'lat': 51.1156, 'lon': 7.5046},
    'h18v03': {'lat': 51.1156, 'lon': 7.5046},

    'amazon': {'lat': -5.709000, 'lon': -66.295},
    'h12v10': {'lat': -5.709000, 'lon': -66.295},

    'mexico': {'lat': 16.6515, 'lon': -94.6572},
    'h08v07': {'lat': 16.6515, 'lon': -94.6572},

    'usa': {'lat': 33.2985, 'lon': -92.4491},
    'h10v05': {'lat': 33.2985, 'lon': -92.4491},
}


# v005 lai locations. # now we use 'lai_v006_world'
lai_locations = {
    'mala': 'Mala_2001_2010',
    'german_forest': 'MODIS_NL_2001_2010',
    'amazon': 'Amazon_2001_2010',
    'usa': 'USA_2001_2010',
    'mexico': 'Mexico_2001_2010',
}


"""
1. Kierspe, Germany
MCD12Q1 CLASS/Band: 4 (Deciduous Broadleaf Trees)
Lat Lon (51.1156, 7.5046)
MODIS_NL_2001_2010
groupname german_forest

2. Tapaua, State of Amazonas, Brazil
MCD12Q1 CLASS/Band: 2 (Evergreen Broadleaf Trees)
Lat Lon (-5.709000, -66.295)
Amazon_2001_2010
groupname amazon

3. San Miguel Chimalapa, Oax., Mexico
MCD12Q1 CLASS/Band: 4
Lat Lon (16.6515, -94.6572)
Mexico_2001_2010
groupname mexico

4. Township, AR, USA
MCD12Q1 CLASS/Band: 4
Lat Lon (33.2985, -92.4491)
USA_2001_2010
groupname usa

5. Chiku, Kelantan, Malaysia
MCD12Q1 CLASS/Band: 2
Lat Lon (4.819, 102.500)
Mala_2001_2010
groupname mala

"""
