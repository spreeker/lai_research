# Plot of the CRU dataset as a function of time 2001-2010 and a global map of some day from that period.

import argparse
import datetime
import numpy
import read_modis
import logging
from netCDF4 import Dataset as netcdf

import h5py
import h5util
import numpy as np

from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
from settings import conf

import extract_green


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

startyear = conf['startyear']
endyear = conf['endyear']
# nc_var = conf['ncvar']

CACHE = {
    'tmp': {},
    'pre': {},
    'vap': {},
    'pet': {},
}

NC_VARS = ('tmp', 'vap', 'pet', 'pre')


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def fill_cache(ds_var):
    nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{ds_var}.dat.nc', 'r')
    nc_attrs, nc_dims, nc_vars = ncdump(nc, verb=False)
    # Extract data from NetCDF file
    lats = nc.variables['lat'][:]  # extract/copy the data
    lons = nc.variables['lon'][:]
    time = nc.variables['time'][:]
    nc_ds = nc.variables[ds_var][:]
    CACHE[ds_var]['lats'] = lats
    CACHE[ds_var]['lons'] = lons
    ts_dt = fix_time(time)
    CACHE[ds_var]['time'] = ts_dt
    CACHE[ds_var]['days'] = time
    CACHE[ds_var]['ds'] = nc_ds
    # draw_basemap(nc_ds, ts_dt, lons, lats, ds_var)


def store_tensor_from_cru(grid, grid_idx):
    """Given grid, store relevant CRU data as tensor in hdf5

    Extract for all cru variables all data at locations/grid
    and store it in hdf5

    saves it in hdf5 dataset in  'groupname/cru'

    It is stored as mulidimensional matrix
    (gridsize) * 120 * variables = 4.

    dimensions:
        - grid size
        - 120 months
        - 4 variables
    """
    global NC_VARS
    size = len(grid)
    # suqares, 120 months, 4  values
    tensor = np.zeros((size, 120, 4))
    # reset cache
    for v in CACHE.values():
        v.clear()

    for di, ds_var in enumerate(NC_VARS):
        if not CACHE.get(ds_var):
            fill_cache(ds_var)
            ds = CACHE[ds_var]['ds']
            for i, (lon, lat) in enumerate(grid):
                lon_idx, lat_idx = grid_idx[i]
                values_at_loc = ds[:, lon_idx, lat_idx]
                # save_location(lon, lat, ds_var, values_at_loc, hdf5)
                tensor[i, :, di] = values_at_loc
                # print(tensor[i, :, di])

    log.debug(tensor.shape)

    # open storage hdf5 file.
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, 'a') as hdf5:
        groupname = conf['groupname']
        cru_groupname = f"{groupname}/cru/"
        ds = h5util.set_dataset(hdf5, cru_groupname, tensor)
        # store meta data
        cru_groupname = f"{groupname}/grid"
        a = np.array(grid)
        h5util.set_dataset(hdf5, cru_groupname, a)
        # store times
        cru_groupname = f"{groupname}/months"
        a = [t.timestamp() for t in CACHE[ds_var]['time']]
        h5util.set_dataset(hdf5, cru_groupname, a)
        # store variable information
        ds.attrs['vars'] = " ".join(NC_VARS)
        log.debug(f'Saved CRU {cru_groupname}')
        d = ["grid", "months", "var"]
        ds.attrs['dimensions'] = " ".join(d)


def grid_for(bbox):
    """
    Extract a grid of locations in bbox

    :param hdf5: optional hdf5 file.
    :param save: save / update hdf5 data
    :return: None.
    """
    box_lats = [p[1] for p in bbox]
    box_lons = [p[0] for p in bbox]

    lat_min = min(box_lats)
    lon_min = min(box_lons)

    lat_max = max(box_lats)
    lon_max = max(box_lons)

    ds_var = 'tmp'
    nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{ds_var}.dat.nc', 'r')
    nc_attrs, nc_dims, nc_vars = ncdump(nc, verb=False)
    all_lats = nc.variables['lat'][:]   # extract/copy the data
    all_lons = nc.variables['lon'][:]

    lats = numpy.array(all_lats)
    lons = numpy.array(all_lons)

    # invalidate all locations outside our given bbox
    # with 1 extra box spacing
    padding_lat = 0.55
    padding_lon = 0.55
    # set every element outside range to zero
    lats[lats < lat_min - padding_lat] = 0
    lats[lats > lat_max + padding_lat] = 0
    lons[lons < lon_min - padding_lon] = 0
    lons[lons > lon_max + padding_lon] = 0

    grid = []
    grid_idx = []

    for lat_idx, lat in enumerate(lats):
        for lon_idx, lon in enumerate(lons):
            # if zero values are out of range.
            if lat == 0:
                continue
            if lon == 0:
                continue

            grid.append((lon, lat))
            grid_idx.append((lat_idx, lon_idx))

    return grid, grid_idx


def cru_filter(points, size_x, size_y):
    """Filter points outside grid
    """
    filtered_points = []

    for x, y in points:
        if x > size_x:
            continue
        if x < 0:
            continue
        if y > size_y:
            continue
        if y < 0:
            continue

        filtered_points.append((int(x), int(y)))

    return filtered_points


def find_xy_cru_grid(geotransform, projection, size_x, size_y, grid):

    points = []

    for lon, lat in grid:
        x, y = read_modis.determine_xy(geotransform, projection, lon, lat)
        points.append((x, y))
        # print(f'{x}, {y}, {lon}, {lat}')

    return cru_filter(points, size_x, size_y)


def collect_cru_in(bbox):
    """
    Given bbox. store lai information
    """
    lon_lat_grid, grid_idx = grid_for(bbox)
    store_tensor_from_cru(lon_lat_grid, grid_idx)


def fix_time(times):
    """
    # List of all times in the file as datetime objects
    """
    dt_time = []
    for t in times:
        start = datetime.date(1900, 1, 1)   # This is the "days since" part
        delta = datetime.timedelta(int(t))  # Create a time delta object from the number of days
        offset = start + delta   # Add the specified number of days to 1900
        dm = datetime.datetime
        d = dm.combine(offset, dm.min.time())
        dt_time.append(d)

    return dt_time


def draw_basemap(nc_ds, dt_time, lons, lats, nc_var):
    """Plot of global temperature on our random day"""
    #
    fig = pyplot.figure()

    # fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
    # for other projections.
    # m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=0)
    m = Basemap(projection='cyl', resolution='c', lon_0=0)
    m.drawcoastlines()
    m.drawmapboundary()
    time_idx = conf['time_idx']
    # Make the plot continuous
    ds_cyclic, lons_cyclic = addcyclic(nc_ds[time_idx, :, :], lons)
    # Shift the grid so lons go from -180 to 180 instead of 0 to 360.
    pre_cyclic, lons_cyclic = shiftgrid(180., ds_cyclic, lons_cyclic, start=False)
    # Create 2D lat/lon arrays for Basemap
    lon2d, lat2d = numpy.meshgrid(lons_cyclic, lats)
    # Transforms lat/lon into plotting coordinates for projection
    x, y = m(lon2d, lat2d)
    # Plot of pre with 11 contour intervals
    cs = m.contourf(x, y, ds_cyclic, 20, cmap=pyplot.cm.Spectral_r)
    cbar = pyplot.colorbar(cs, orientation='horizontal', shrink=0.9)
    dot_time = dt_time[time_idx]
    # cbar.set_label
    pyplot.title(f"Global {nc_var} for {dot_time.year}.{dot_time.month}")
    pyplot.show()


def draw_plot(dt_time, time_idx, lat_idx, lon_idx, nc_ds):
    """
    :param fig:
    :return:
    """

    # A plot
    # fig = pyplot.figure()
    pyplot.figure()
    dot_time = dt_time[time_idx]

    pyplot.plot(dt_time, nc_ds[:, lat_idx, lon_idx], c='r')
    pyplot.plot(dt_time[time_idx], nc_ds[time_idx, lat_idx, lon_idx], c='b', marker='o')
    pyplot.text(dt_time[time_idx], nc_ds[time_idx, lat_idx, lon_idx], dot_time, ha='right')

    # fig.autofmt_xdate()
    # pyplot.ylabel("%s (%s)" % (nc.variables['pre'].var_desc,\
    #                        nc.variables['pre'].units))
    pyplot.xlabel("Time")
    pyplot.title(f"Local {nc_var} from {startyear} to {endyear}")
    pyplot.show()


def collect_cru():
    """
    Collect all relavant CRU information for
    current Landuse map in settings.conf
    """
    # first extract green map.
    ds, values, green, xarr, yarr = extract_green.extract()
    # find out wich bouding (lai/landuse) bbox we are currently working on.
    bbox = read_modis.make_lonlat_bbox(ds, verb=False)
    # log.debug('BBOX: %s', bbox)
    # store CRU data in hdf5
    collect_cru_in(bbox)
    # convert lat,lon to x, y
    # grid = find_xy_cru_grid(ds, lon_lat_grid)


def collect_cru_v006():
    """
    Collect all relavant CRU information for
    current Landuse map in settings.conf
    """
    hdf_file = extract_green.modis_hv()
    if not hdf_file:
        return
    # first extract green map.
    try:
        ds, values, green, xarr, yarr = extract_green.extract_v006(hdf_file)
    except ValueError:
        log.error("no green")
        return
    # find out wich bouding (lai/landuse) bbox we are currently working on.
    bbox = read_modis.make_lonlat_bbox(ds, verb=False)
    # log.debug('BBOX: %s', bbox)
    # store CRU data in hdf5
    collect_cru_in(bbox)
    # convert lat,lon to x, y
    # grid = find_xy_cru_grid(ds, lon_lat_grid)




def main(args):
    if args.collect_cru:
        collect_cru()


if __name__ == '__main__':
    # print_hdf_info()
    desc = "Create CRU LAI prediction map"
    inputparser = argparse.ArgumentParser(desc)
    inputparser.add_argument(
        '--collect_cru',
        action='store_true',
        default=False,
        help="Extract CRU data for currenct configured map")

    args = inputparser.parse_args()
    main(args)
