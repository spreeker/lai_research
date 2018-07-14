"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
and create monthly time-series in a huge matrix

Recipy followed:

- Load raw LAI nasa data is loaded into a huge matrix.
We create a 8 day 1200*1200 matrix. So for 10 years this means
a matrix of 477*1200*1200

- Convert this 8 day matrix in a 120 month matrix.
so we end up with 120*1200*1200


"""
import os
import argparse
import glob
import dateparser
from matplotlib import pyplot
import logging
import read_modis
import h5py
import h5util
import numpy as np
import datetime
# import load_datasets
# import q

import settings
from settings import conf
from settings import lai_locations

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

LAI_LAYERS = []
LAI_CUBE = []
META = []


def extract_lai(dataset, _geotransform, _projection):
    """

    :param dataset: hdf source
    :param geotransform: hdf geotransform information
    :param projection: projection

    adds (datetime, lai_value) to global LAI_VALUES
    """

    band = dataset.GetRasterBand(1)

    metadata = dataset.GetMetadata_Dict()

    if band is None:
        log.error(dataset)
        raise ValueError('Could not read hdf file')

    # bandtype = gdal.GetDataTypeName(band.DataType)
    lai = band.ReadAsArray()
    # values are 10* real values.
    # value = lai[y][x] / 10
    measurement_time = dateparser.parse(metadata['RANGEBEGINNINGDATE'])

    LAI_LAYERS.append((measurement_time, lai))


def load_lai_from_hdf_nasa():
    """
    Given location and hdf direcotry in settings.conf load all LAI
    values for given location over time.
    Each hdf file is a 8 day average.
    :return:  global LAI_VALUES will be filled.
    """
    hdf_files = get_lai_gdal_modis_files()

    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    hdf_files.sort()
    log.debug(hdf_files)

    for hdf_name in hdf_files:
        log.debug('Loading %s', hdf_name)
        ds, geotransform, projection = read_modis.load_modis_data(
            hdf_name
            # 'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km'
        )
        extract_lai(ds, geotransform, projection)

    # sort values by date.
    LAI_LAYERS.sort()


def load_lai_from_hdf_nasa_v006():
    """
    Given location and hdf direcotry in settings.conf load all LAI
    values for given location over time.
    Each hdf file is a 8 day average.
    :return:  global LAI_VALUES will be filled.
    """
    hdf_files = get_lai_gdal_modis_files_0006()

    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    hdf_files.sort()
    log.debug(hdf_files)

    for hdf_name in hdf_files:
        log.debug('Loading %s', hdf_name)
        ds, geotransform, projection = read_modis.load_modis_data(
            hdf_name
            # 'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km'
        )
        extract_lai(ds, geotransform, projection)

    # sort values by date.
    LAI_LAYERS.sort()


def get_lai_gdal_modis_files():
    """Return LAI gdal location urls
    """

    # hdf LAI directory data
    groupname = conf['groupname']
    hdf_dir = lai_locations[groupname]
    hdf_dir = os.path.join(settings.PROJECT_ROOT, hdf_dir)
    search_path = '*.hdf'

    source_hdf_dir = os.path.join(
        settings.PROJECT_ROOT,
        settings.lai_locations[conf['groupname']],
    )

    hdf_files = glob.glob(os.path.join(source_hdf_dir, search_path))

    if not hdf_files:
        log.exception(source_hdf_dir)
        raise ValueError('Directory hdf4 LAI source wrong.')

    read_modis.load_modis_data(hdf_files[0])
    return

    hdf_files = [
        f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km'
        for hdf_name in hdf_files]

    return hdf_files


def get_lai_gdal_modis_files_0006():
    search_path = f"*.{conf['location']}.*.hdf"
    source_hdf_dir = os.path.join(
        settings.PROJECT_ROOT,
        'lai_v006_world')

    log.debug(source_hdf_dir)
    log.debug(search_path)

    hdf_files = glob.glob(os.path.join(source_hdf_dir, search_path))

    log.info('HDF LAI FILES %d', len(hdf_files))

    log.info(hdf_files)

    if not hdf_files:
        log.info("no source lai files found")
        return

    read_modis.load_modis_data(hdf_files[0])

    hdf_files = [
        f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m'
        for hdf_name in hdf_files]

    return hdf_files


def extract_lai_meta():
    """
    Extract from one modis file the META information
    """
    for hdf_gdal_name in get_lai_gdal_modis_files()[:1]:
        print(hdf_gdal_name)
        ds, geotransform, projection = read_modis.load_modis_data(
            hdf_gdal_name
            # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m',
            # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            # f'HDF4_EOS: EOS_GRID:"{hdf_name}": MOD12Q1:Land_Cover_Type_5',
            # f'HDF4_EOS:EOS_GRID:"ground_usage":MOD12Q1:Land_Cover_Type_5',
            )
        bbox = read_modis.make_lonlat_bbox(ds)

    return geotransform, projection, bbox


def extract_lai_meta_v006():
    """
    Extract from one modis file the META information
    """
    for hdf_gdal_name in get_lai_gdal_modis_files_0006()[:1]:
        print(hdf_gdal_name)
        ds, geotransform, projection = read_modis.load_modis_data(
            hdf_gdal_name
            # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m',
            # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            # f'HDF4_EOS: EOS_GRID:"{hdf_name}": MOD12Q1:Land_Cover_Type_5',
            # f'HDF4_EOS:EOS_GRID:"ground_usage":MOD12Q1:Land_Cover_Type_5',
            )
        bbox = read_modis.make_lonlat_bbox(ds)

    return geotransform, projection, bbox




def store_meta():
    """Store meta information about LAI

    TODO maybe CF complient so we can use panoply etc?
    """
    geotransform, projection, bbox = extract_lai_meta()
    h5util.save('lai/meta/', [bbox], attrs={
        'geotransform': geotransform,
        'projection': str(projection),
        'bbox': bbox,
    })


def make_cube():

    lai_cube = np.array(
        [matrix for time, matrix in LAI_LAYERS]
    )

    time_matrix = [time for time, cell in LAI_LAYERS]

    return time_matrix, lai_cube


def save_lai_8_day():
    """Store all hdf LAI matrixes in one file
    """

    time_matrix, lai_cube = make_cube()
    time_matrix = [time.timestamp() for time in time_matrix]
    log.debug('X- time count %d', len(time_matrix))

    groupname = conf['groupname'] + '/lai/nasa'
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "a") as data_file:
        h5util.set_dataset(data_file, groupname, 'timestamps', time_matrix)
        h5util.set_dataset(data_file, groupname, lai_cube)

    log.info('Stored all lai layers of %s in %s', groupname, storage_name)


def clean_nasa():
    """Delete raw source lai data to shrik file size
    """
    storage_name = conf['hdf5storage']
    groupname = conf['groupname'] + '/lai/nasa'
    with h5py.File(storage_name, "a") as data_file:
        del data_file[groupname]


def plot_matrix(ds):
    pyplot.imshow(ds, vmin=0, vmax=26)
    pyplot.show()


def plot_location(values):
    pyplot.plot(values)
    pyplot.show()


def _printname(name):
    log.debug(name)


def load_cube():
    """Load the stored LAI cube

    example usage:

    # access one layer
    plot_matrix(lai_cube[1][:][:])

    # access one location over 10 years
    values = lai_cube[:, 1000, 1000]
    """
    # lai_cube = h5util.load_datasets('lai/nasa')

    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "a") as data_file:
        data_file.visit(_printname)
        cube = f'{groupname}/lai/nasa'
        lai_cube = np.array(data_file[cube])
        timestamps = data_file['timestamps']
        dt = datetime.datetime
        timestamps = [dt.fromtimestamp(d) for d in timestamps]

    return timestamps, lai_cube


def create_lai_day_layer(timestamps):
    """Create day - layer index within cube"""
    lai_day_layer = []

    for layer_idx, d8t in enumerate(timestamps):
        for i in range(8):
            day1 = datetime.timedelta(days=1)
            oneday = d8t + i * day1
            lai_day_layer.append((oneday, layer_idx))

    return lai_day_layer


def avg_month(month_layers, cube):
    """Sum up all 8 day data into one month
    """

    if conf['v006']:
        m = np.zeros((2400, 2400))
    else:
        m = np.zeros((1200, 1200))

    for idx in month_layers:
        m = np.add(m, cube[idx, :, :])

    return np.divide(m, len(month_layers))


def gen_months(day_layers, cube):
    """
    Generate average months.
    """
    month_layers = []
    current_month = None
    current_year = None
    month = None

    # create month avg values of lai
    for day, layer in day_layers:
        month = day.month
        year = day.year

        if current_month is None:
            current_month = month
        if current_year is None:
            current_year = year

        if month is not current_month:
            log.debug('%d %d %d %d', year, month, current_year, current_month)
            yield current_year, current_month, avg_month(month_layers, cube)
            # reset month counting.
            current_month = month
            current_year = year
            month_layers = []

        # store layer of one day.
        month_layers.append(layer)

    if month_layers:
        # store last month
        yield current_year, current_month, avg_month(month_layers, cube)



def savitzky_golay(cube):
    from scipy import signal
    dvs = signal.savgol_filter(cube, 9, 3, axis=0, mode='mirror')
    pyplot.plot(dvs[:120, 1000, 1000])
    pyplot.plot(cube[:120, 1000, 1000])
    pyplot.show()
    return dvs


def make_month_cube(day_layers, source_cube):

    x, y = 1200, 1200
    if settings.conf['v006']:
        x, y = 2400, 2400

    month_cube = np.zeros((120, x, y), dtype=np.float)

    # pyplot.plot(timestamps, [1 for t in timestamps])
    for year, month, month_layer in gen_months(day_layers, source_cube):
        idx = (year - 2000) * 12 + (month - 1)
        # if idx >= 120:
        # break
        month_cube[idx] = month_layer
        log.debug('Month %s %s idx %d', year, month, idx)

    return month_cube


def make_smooth_cube(timestamps, cube, x=None, y=None):

    smooth_cube = savitzky_golay(cube)

    if x and y:
        xsteps = 75
        raw = cube[:xsteps, x, y] / 10
        smooth = smooth_cube[:xsteps, x, y] / 10
        plot_lai(timestamps[:xsteps], raw, smooth)

    return smooth_cube


def convert_to_120month_cube(timestamps, cube, name='month'):
    """"
    Convert 8 day period cube to smoothed month
    period cube.
    """
    day_layers = create_lai_day_layer(timestamps)
    mc = make_month_cube(day_layers, cube)
    # smc = make_month_cube(day_layers, smooth_cube)

    storage_name = conf['hdf5storage']

    with h5py.File(storage_name, "a") as data_file:
        data_file.visit(_printname)
        groupname = conf['groupname']
        target = f'{groupname}/lai/{name}'
        h5util.set_dataset(data_file, target, mc.astype(float))
        # target = f'{groupname}/lai/smooth_month'
        # h5util.set_dataset(data_file, target, smc.astype(float))


def plot_lai(timestamps, lai_values, smooth):
    """Plot to Validate if smoothing went ok.

    :param timestamps dates:
    :param lai_values:
    :param smooth:
    """

    assert len(smooth) == len(lai_values)

    pyplot.plot(timestamps, lai_values, label='Original LAI dataset' )

    pyplot.plot(timestamps, smooth, label='smooth')

    start = timestamps[0]
    end = timestamps[-1]
    start = f"{start.year}-{start.month}"
    end = f"{end.year}-{end.month}"

    pyplot.title(
        f"LAI for {start}-{end} Month")
    pyplot.legend()
    pyplot.show()


def plot_months(points):
    """Plot lai and smoothed lai for months

    :param lai_values_by_month:
    :param smooth: also plot smoothed version
    :return: PLOT of lai values.
    """
    storage_name = conf['hdf5storage']
    h5util.load_timestamps('months')
    with h5py.File(storage_name, "r") as data_file:
        data_file.visit(_printname)
        groupname = conf['groupname']
        smc = f'{groupname}/lai/smooth_month'
        mc = f'{groupname}/lai/month'
        m_cube = np.array(data_file[mc])
        sm_cube = np.array(data_file[smc])

    # 1000, 1000 is an example location in the middle of the dessert
    for n in points[::2]:
        pyplot.plot(
            m_cube[:50, n, n+1], label='Original LAI dataset')
        pyplot.plot(sm_cube[:50, n, n+1], label='smooth')
        pyplot.title(f"LAI for 2001-2010 Month")
        pyplot.legend()
        pyplot.show()


def plot_map(layer=10):
    cube = h5util.load_dataset('lai/smooth_month')
    log.debug(cube.shape)
    layer = cube[layer, :, :]
    log.debug(layer.shape)
    pyplot.imshow(layer[0, :, :])
    pyplot.show()


def make_hv_cube():

    LAI_LAYERS.sort()

    if LAI_LAYERS:
        timestamps, cube = make_cube()

        # timestamps = [time.timestamp() for time in timestamps]
        log.debug('X- time count %d', len(timestamps))

        pyplot.title('lai dates')
        pyplot.plot(timestamps, [1 for x in timestamps])
        pyplot.show()

        # convert_to_120month_cube(timestamps, cube, 'month')
        smooth = make_smooth_cube(timestamps, cube, 1000, 1000)
        convert_to_120month_cube(timestamps, smooth, 'smooth_month')
        # convert_to_120month_cube(timestamps, cube, 'smooth_month')
        LAI_LAYERS.clear()


def main(args):
    cube = None

    x, y = None, None

    if args.plot:
        x, y = args.plot
        x = int(x)
        y = int(y)

    if args.nasa:
        load_lai_from_hdf_nasa()
        # save results
        if not args.savemonthly:
            save_lai_8_day()
    if args.nasaclean:
        clean_nasa()

    if args.smooth:
        assert x
        assert y
        timestamps, cube = load_cube()
        make_smooth_cube(timestamps, cube, x, y)
        return

    if args.savemonthly:
        if LAI_LAYERS:
            timestamps, cube = make_cube()
        else:
            timestamps, cube = load_cube()

        smooth = make_smooth_cube(timestamps, cube, x, y)
        convert_to_120month_cube(timestamps, cube, smooth)
        LAI_LAYERS.clear()

    if args.plot:
        points = [int(n) for n in args.plot]
        plot_months(points)
    if args.store_meta:
        store_meta()
    if args.plotlayer:
        plot_map(layer=args.plotlayer)


if __name__ == '__main__':
    desc = "Create LAI matrix cubes of LAI"
    inputparser = argparse.ArgumentParser(desc)

    inputparser.add_argument(
        '--nasa',
        action='store_true',
        default=False,
        help="Load raw hdf nasa data from direcotory")

    inputparser.add_argument(
        '--nasaclean',
        action='store_true',
        default=False,
        help="Clean raw nasa lai data from hdf5")

    inputparser.add_argument(
        '--smooth',
        action='store_true',
        default=False,
        help="Create smoothed cube of lai data")

    inputparser.add_argument(
        '--savemonthly',
        action='store_true',
        default=False,
        help="Convert 8 day period to month perod of CRU")

    inputparser.add_argument(
        '--store_meta',
        action='store_true',
        default=False,
        help="Store LAI meta data in hdf5")

    inputparser.add_argument(
        '--plot',
        nargs=2,
        default=None, # [1000, 1000],
        metavar=('x', 'y'),
        help="Plot LAI data at given pixel coordinates")

    inputparser.add_argument(
        '--plotlayer',
        # action='store_true',
        nargs=1,
        type=int,
        default=[100],
        metavar=('layer_n',),
        help="Plot LAI map at given layer")

    args = inputparser.parse_args()

    main(args)
