"""
Create world map of multiregression cru data with
LAI data of green areas

For every h,v location in modis grid.
create seperate hdf files for each modis square (h, v):

We prepare data for LAI prediction function calculations.

1. Extract green locations from map.
2. Collect all CRU data for green locations and store in hdf file.
    - 10 years
    - use geotransform from modis file to make selection in cru data!
    - create mapping matrix. px, py -> lon, lat
3. Collect all LAI data for green locations (done)
    - smoothed.
    - normalized.
    - 10 years. / selected years.
4. Calculate best lai predictor function for each green location.
    - draw map with colors from this.
    - maybe just a subset of these locations.
5. Improve predictions functions and run step 4 again.

"""
import os
import settings
import numpy as np
import argparse
import logging

import h5util
from settings import conf
from create_lai_cube_v006 import load_lai_from_hdf_nasa_v006
from create_lai_cube_v006 import make_hv_cube
import extract_green
import extract_CRU
import multi_regression_006


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# We store out endresults in here..
world_array_file = 'worldarray.np'

if not os.path.exists(world_array_file):
    # lon, lat, model scores
    with open(world_array_file, 'wb') as _file:
        world_array = np.zeros((720, 360, 4))
        np.save(_file, world_array)
else:
    with open(world_array_file, 'rb') as _file:
        world_array = np.load(_file)


def modis_sinodal_grid(h1, v1, h2, v2):
    log.debug(h1)
    for h in range(h1, h2):
        for v in range(v1, v2):
            yield h, v


def action(args):
    # step1
    if args.green:
        # find involved hdf green landuse file
        path = extract_green.modis_hv()
        if path:
            extract_green.modis_hv_extract(path)
    # step 2
    if args.extract_cru:
        extract_CRU.collect_cru_v006()

    # step 3
    if args.nasa:
        if h5util.layer_exists('lai/smooth_month'):
            log.debug('smooth month Alrady there..')
            # return

        load_lai_from_hdf_nasa_v006()
        make_hv_cube()

    # Victory!
    if args.multi_regression:
        plotlocation = False
        if args.plotlocation:
            plotlocation = True
        multi_regression_006.main_world(plotlocation)


def setup_conf(h, v, args):

    hdf5_path = os.path.join(
        settings.PROJECT_ROOT,
        # "landuse_german_forest",
        f"world_modis_hdf5_2",
        # "MCD12Q1.A2011001.h18v03.051.2014288191624.hdf"
        f"h{h:02d}v{v:02d}.hdf"
    )

    loc_key = f'h{h:02d}v{v:02d}'
    conf['h'] = h
    conf['v'] = v
    conf['location'] = loc_key
    conf['groupname'] = loc_key
    conf['hdf5storage'] = hdf5_path
    conf['v006'] = True
    log.debug('conf setup for %s %s', h, v)


def main(args):
    """
    """
    if args.monthsrange:
        start, end = args.location
        conf['start_month'] = int(start)
        conf['end_month'] = int(end)

    if args.plotpickle:
        multi_regression_006.plot_pickle()
        return

    if args.location:
        h, v = args.location
        setup_conf(h, v, args)
        action(args)
        return

    boxlabel = 'all'
    if args.box:

        assert args.box[0] < args.box[2]
        assert args.box[1] < args.box[3]

        boxlabel = "%d_%d_%d_%d" % (
            args.box[0],
            args.box[1],
            args.box[2],
            args.box[3],
        )

    if args.greenrange:
        conf['greenrange'] = args.greenrange
    g1, g2 = conf['greenrange']

    conf['csv'] = f'result-run-auto-{boxlabel}-g{g1}-{g2}.csv'
    for h, v in modis_sinodal_grid(*args.box):
        log.info('TILE  h %d v %d', h, v)
        setup_conf(h, v, args)
        action(args)


if __name__ == '__main__':
    desc = "Create WORLD map"
    inputparser = argparse.ArgumentParser(desc)

    inputparser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help="debug stuff")

    inputparser.add_argument(
        '--extract_cru',
        action='store_true',
        default=False,
        help="Make predictor plots of locations definded in settings")

    inputparser.add_argument(
        '--green',
        action='store_true',
        default=False,
        help="extract green landuse")

    inputparser.add_argument(
        '--nasa',
        action='store_true',
        default=False,
        help="Load raw hdf nasa data from direcotory")

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
        '--location',
        type=int,
        nargs=2,
        default=[],
        help="h, v modis grid location to load")

    inputparser.add_argument(
        '--box',
        type=int,
        nargs=4,
        default=[0, 0, 36, 17],
        help="min (h, v) (h v) modis boxes to load")

    inputparser.add_argument(
        '--multi_regression',
        action='store_true',
        default=False,
        help="create map which show most influential climate variables")

    inputparser.add_argument(
        '--plotlocation',
        action='store_true',
        default=False,
        help="Plot data on pixel location")

    inputparser.add_argument(
        '--monthsrange',
        type=int,
        nargs=2,
        default=[],
        help="Give up from and to months [0, 120] 118-120 would mean 2010")

    inputparser.add_argument(
        '--greenrange',
        type=int,
        nargs=2,
        default=[],
        help="Give up from and to months [0, 120] 118-120 would mean 2010")

    inputparser.add_argument(
        '--plotpickle',
        action='store_true',
        default=False,
        help="Give up from and to months [0, 120] 118-120 would mean 2010")

    args = inputparser.parse_args()
    main(args)
