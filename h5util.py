"""
Store/Access data in hdf5 file using some compression

Otherwise we have files which takes many gigabytes..

Saves hdf5 dataset in  'groupname/xxxx'
"""

import os
import numpy as np
import logging
import h5py
import datetime
from settings import conf
import netCDF4 as nc4

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def set_dataset(hdf, path, data):
    """Replace of set data in path of hdf file"""
    if path in hdf.keys():
        del hdf[path]
    dl = hdf.create_dataset(
        path, data=data, compression="gzip",
        compression_opts=1
    )
    return dl


def layer_exists(name):
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    _hdfpath = f'{groupname}/{name}'

    if not os.path.exists(storage_name):
        return False

    with h5py.File(storage_name, 'r') as hdf:
        if _hdfpath in hdf.keys():
            return True
    return False


def save(name, data, attrs={}):
    """Save data to hdf5. defaults to tile we are working on
    """
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    _hdfpath = f'{groupname}/{name}'

    write_mode = "a"      # Append
    if not os.path.exists(storage_name):
        write_mode = "w"  # Create new file
    with h5py.File(storage_name, write_mode) as data_file:
        ds = set_dataset(data_file, _hdfpath, data)
        for k, v in attrs.items():
            ds.attrs[k] = v


def load_dataset(ds_name, attrs=False):
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    _hdf_path = f'{groupname}/{ds_name}'

    with h5py.File(storage_name, "r") as data_file:
        if attrs:
            return(dict(list(data_file[_hdf_path].attrs.items())))
        return np.array(data_file[_hdf_path])


def load_path(path):
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:
        return np.array(data_file[f'{path}'])


def load_attrs(ds_name, keys):
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:
        ds = data_file[f'{groupname}/{ds_name}']
        attrs = {}
        for k in keys:
            attrs[k] = ds.attrs[k]
        return attrs


def load_timestamps(ds_name='timestamps'):
    timestamps = load_dataset(ds_name)

    time_x = []
    for t in timestamps:
        dt = datetime.date.fromtimestamp(t)
        time_x.append(dt)

    return time_x


def load_cru_grid():
    return load_dataset('grid')


def _printname(name):
    log.debug(name)


def print_paths():
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "a") as data_file:
        data_file.visit(_printname)


def world_data_save(name, data, attrs={}):
    """Save world data to hdf5. defaults to tile we are working on
    WIP
    """
    storage_name = conf['world']

    attrs['_FillValue'] = -1

    write_mode = "a"      # Append
    if not os.path.exists(storage_name):
        write_mode = "w"  # Create new file

    with h5py.File(storage_name, write_mode) as data_file:
        ds = set_dataset(data_file, name, data)
        for k, v in attrs.items():
            ds.attrs[k] = v


def set_world_meta(lats, lons):
    write_mode = "a"      # Append
    storage_name = conf['world']
    _, lons, lats, _ = world_grid()

    with h5py.File(storage_name, write_mode) as data_file:
        world_data_save(
            'Longitude', lons,
            attrs={
                'long_name': 'longitude',
                'units': 'degrees_east',
                '_CoordinateAxisType': 'Lon'
            }
        )
        world_data_save(
            'Latitude', lats,
            attrs={
                'long_name': 'latitude',
                'units': 'degrees_north',
                '_CoordinateAxisType': 'Lat'
            }
        )

        data_file.attrs['where_projdef'] = 'WGS84'
        data_file.attrs['where_LL_lon'] = -179.25
        data_file.attrs['where_LL_lat'] = -85.25
        data_file.attrs['where_UR_lon'] = 179.75
        data_file.attrs['where_UR_lat'] = 89.75
        data_file.attrs['no_data_value'] = -1

        # turns out that GDAL for unknown datasets
        # adds 180 degrees which are near the edges.
        # which puts our map in the wrong location
        # panolply does the right thing. only needs Latitude and Longitude
        # use netcdf..
        # information

        # const char *const pszProj4String = GetMetadataItem("where_projdef");
        # const char *const pszLL_lon = GetMetadataItem("where_LL_lon");
        # const char *const pszLL_lat = GetMetadataItem("where_LL_lat");
        # const char *const pszUR_lon = GetMetadataItem("where_UR_lon");
        # const char *const pszUR_lat = GetMetadataItem("where_UR_lat");
        # if( pszProj4String == nullptr ||
        #     p


def _range(start, step, length):
    all_options = [start + x*step for x in range(length)]
    return all_options


def world_grid():
    """
    Create a lon lat grid equal to cru
    """
    # grid matches cru
    lons = _range(-179.75, .5, 720)
    lats = _range(-89.75, .5, 360)
    empty_world = np.zeros((len(lats), len(lons)))
    empty_world.fill(-1)

    grid_idx_lookpup = {}

    for lat_idx, lat in enumerate(lats):
        for lon_idx, lon in enumerate(lons):
            grid_idx_lookpup[(lon, lat)] = (lon_idx, lat_idx)
    # return lons, lats, empty_world
    return grid_idx_lookpup, lons, lats, empty_world


def world_data_load(name):
    """Save world data from hdf5
    """
    storage_name = conf['world']
    write_mode = "r"  # read

    grid_idx, lons, lats, empty_world = world_grid()

    if not os.path.exists(storage_name):
        return

    with nc4.Dataset(storage_name, write_mode) as world:
        # load current global data if it exists.
        try:
            data = world[name][:]
            return np.array(data)
        except (IndexError, KeyError):
            return


def save_netcdf(name, data):
    """Update or save data layer
    """
    if not np.any(data > -1):
        log.info('no data to save..')
        return

    storage_name = conf['world']

    write_mode = "a"      # Append
    if not os.path.exists(storage_name):
        write_mode = "w"  # Create new file

    grid_idx, lons, lats, _empty = world_grid()

    # with nc4.Dataset(storage_name, write_mode, # format='NETCDF4'
    # QGIS / GDAL CAN NOT HANDLE NETCDF4/HDF5 PROPERLY.
    with nc4.Dataset(storage_name, write_mode, format="NETCDF3_CLASSIC") as root_grp:
        if write_mode == 'w':
            root_grp.Conventions = "CF-1.6"
            root_grp.title = "global LAI stats green vegetation types"

        world = root_grp

        if 'lon' not in root_grp.dimensions:
            root_grp.createDimension('lon', len(lons))
            root_grp.createDimension('lat', len(lats))

        if "lon" not in root_grp.variables:
            longitude = root_grp.createVariable(
                'lon', 'f4', 'lon', zlib=True)
            longitude[:] = lons
            latitude = root_grp.createVariable(
                'lat', 'f4', 'lat', zlib=True)
            latitude[:] = lats
            latitude.units = 'degrees_north'
            longitude.units = 'degrees_east'

        try:
            # update the data
            world[name][:] = data
        except (IndexError, KeyError):
            # set new data layer
            values = world.createVariable(
                name, 'f4', ('lat', 'lon'), fill_value=-1, zlib=True)
            values[:] = data
            values.valid_min = 0

        log.debug('Saved nc %s %d', name, np.sum(data > -1))
