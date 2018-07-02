"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu.

We extract locations of green area.
"""

import os
import argparse
import gdal
import glob
import numpy as np

import settings
from matplotlib import pyplot as plt
import matplotlib as mpl
import logging
import numpy.ma as ma

from read_modis import load_modis_data
import h5util

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def plot_green(data, title='green'):
    cmap = plt.get_cmap('BuGn', 6-np.min(data)+1)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)
    cmaplist[-1] = (.7, .7, .7, 1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=6, clip=True)
    mat = plt.imshow(
        data,
        norm=norm,
        cmap=cmap, vmin = np.min(data)-.5, vmax=6 + .5)
    plt.colorbar(mat, ticks=np.arange(np.min(data), 6+1))
    # plt.imshow(data, norm=norm, cmap=cmap)
    # plt.colorbar()
    plt.title(title)
    plt.show()


def extract_green(dataset, _geotransform, _projection, verb=True):
    """"Extract green locations from map.
    """

    band = dataset.GetRasterBand(1)
    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    if verb:
        log.debug('Data type %s', bandtype)

    data = band.ReadAsArray()
    raw_data = np.copy(data)

    if verb:
        log.debug('Bytes: %s Size %.5d kb', data.nbytes, float(data.nbytes) / 1024)

    passer = np.logical_and(data > 0, data <= 1000)

    if verb:
        log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
            data[passer].min(), data[passer].max(),
            data[passer].mean(), data[passer].std())
        )

    if verb:
        plot_green(data)
    # create green mask
    green = ma.masked_inside(data, 1, 9)
    xarr, yarr = np.where(green.mask)
    # test if we extracted green
    data[green.mask] = 0
    # check if all green is gone
    if verb:
        plot_green(data, title='no green')

    # log.info('Converting xy to lon lat Locations')
    # lons, lats = determine_lonlat(geotransform, projection, xarr[:], yarr[:])
    # log.info('Extracted %d Locations', len(lons))
    return dataset, raw_data, green, xarr, yarr


def resize_green(land_use):
    """Resize 2400*2400 to 1200*1200

    We only take 2*2 squares with all green type values
    """
    # the new matrix we are going to fill
    resize = np.zeros((1200, 1200))
    lu = np.copy(land_use)

    # create 4. 1200*1200 samples
    g1 = lu[0::2, 0::2]
    g2 = lu[1::2, 0::2]
    g3 = lu[0::2, 1::2]
    g4 = lu[1::2, 1::2]

    # check if they have green values
    # in all 4 locations
    a = np.logical_and(g1 > 0, g1 < 5)
    b = np.logical_and(g2 > 0, g2 < 5)
    c = np.logical_and(g3 > 0, g3 < 5)
    d = np.logical_and(g4 > 0, g4 < 5)

    tg1 = np.bitwise_and(a, b)
    tg2 = np.bitwise_and(c, d)
    all_green = np.bitwise_and(tg1, tg2)

    # plot orignal 1/4 sample
    # plot_green(g1[370:850, 320:800], title='g1')
    # plot_green(g2[370:850, 320:800], title='g2')
    # plot_green(g3[370:850, 320:800], title='g3')
    # plot_green(g4[370:850, 320:800], title='g4')

    plot_green(g1[:, :], title='g1')
    plot_green(g2[:, :], title='g2')
    plot_green(g3[:, :], title='g3')
    plot_green(g4[:, :], title='g4')
    #
    # all not similar is not interesting
    # whats left should be green stuff thats
    # green everywhere (1mk2)
    # gg = np.copy(g2)
    # above 6 is gray on map
    g1[~all_green] = 0
    g2[~all_green] = 0
    g3[~all_green] = 0
    g4[~all_green] = 0
    resize = (resize + g1 + g2 + g3 + g4) / 4 # + g2[all_green] + g3[all_green] + g4[all_green]
    # np.divide(resize, 4)

    for q in [g1, g2, g3, g4, resize]:
        log.debug('stats:')
        log.debug('1: %d', np.count_nonzero([q == 1]))
        log.debug('2: %d', np.count_nonzero([q == 2]))
        log.debug('3: %d', np.count_nonzero([q == 3]))
        log.debug('4: %d', np.count_nonzero([q == 4]))
        log.debug('5: %d', np.count_nonzero([q == 5]))

    # plot_green(resize[770:850, 320:400], title='results - wadden')
    plot_green(resize, title='results')

    return resize


ground_usage = os.path.join(
    settings.PROJECT_ROOT,
    # "landuse_german_forest",
    f"Landuse_{settings.conf['groupname']}",
    # "MCD12Q1.A2011001.h18v03.051.2014288191624.hdf"
    "*.hdf"
)


def get_landuse():
    options = glob.glob(ground_usage)
    return options[0]


def extract():
    ground_usage = get_landuse()
    log.info(ground_usage)

    if not ground_usage:
        raise ValueError('Landuse file not found')

    ground_usage = f'HDF4_EOS:EOS_GRID:"{ground_usage}":MOD12Q1:Land_Cover_Type_5'   # noqa
    dataset, geotransform, projection = \
        load_modis_data(ground_usage, verb=False)
    return extract_green(dataset, geotransform, projection)


def extract_v006(filepath):
    """extract v006 ground usage data
    """
    log.info(filepath)
    ground_usage = f'HDF4_EOS:EOS_GRID:"{filepath}":MCD12Q1:LC_Type5'   # noqa
    dataset, geotransform, projection =  \
        load_modis_data(ground_usage, verb=False)
    return extract_green(dataset, geotransform, projection, verb=False)


def print_hdf_info():
    """Extract hdf layer info

    with extracted layer infromation we can
    fill configuration in extract function

    :return: log information.
    """
    from settings import PROJECT_ROOT
    # hdf LAI directory data
    landuse_path = os.path.join(PROJECT_ROOT, 'Landuse_usa', '*.hdf')
    hdf_files = glob.glob(landuse_path)

    if not hdf_files:
        raise ValueError('Directory hdf4 source wrong.')

    for hdf_name in hdf_files:
        load_modis_data(hdf_name)


def get_hv_landuse():
    h = settings.conf['h']
    v = settings.conf['v']

    ground_usage_path = os.path.join(
        settings.PROJECT_ROOT,
        # "landuse_german_forest",
        f"worldlanduse_2010",
        # "MCD12Q1.A2011001.h18v03.051.2014288191624.hdf"
        f"*h{h:02d}v{v:02d}*.hdf"
    )
    return ground_usage_path


def modis_hv():
    """
    Extract green of location

    return dataset, raw_data, green, xarr, yarr
    """

    landuse_path = get_hv_landuse()
    log.debug(landuse_path)
    hdf_files = glob.glob(landuse_path)
    if not hdf_files:
        log.debug(
            'No landuse for h %d v %d',
            settings.conf['h'], settings.conf['v'])
        return

    assert len(hdf_files) == 1

    return hdf_files[0]


def modis_hv_extract(hdf_file):
    # log.debug(hdf_file)
    load_modis_data(hdf_file)
    try:
        ds, lu, g, xarr, yarr = extract_v006(hdf_file)
    except ValueError:
        log.error('No green %s', hdf_file)
        return
    h5util.save('green', lu)


def main(args):

    if args.info:
        print_hdf_info()
        return

    if args.extract:
        ds, lu, g, xarr, yarr = extract()
        if args.resize:
            rs = resize_green(lu)
            if args.save:
                h5util.save('green', rs)


if __name__ == '__main__':
    # print_hdf_info()
    desc = "Create GREEN matrix cubes of LAI"
    inputparser = argparse.ArgumentParser(desc)

    inputparser.add_argument(
        '--info',
        action='store_true',
        default=False,
        help="Print raw hdf landuse data from direcotory")

    inputparser.add_argument(
        '--extract',
        action='store_true',
        default=False,
        help="Extract natural area hdf landuse data from direcotory")

    inputparser.add_argument(
        '--resize',
        action='store_true',
        default=False,
        help="resize 2400*2400 to 1200*1200")

    inputparser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help="save data in hdf5 as green")

    args = inputparser.parse_args()
    main(args)
