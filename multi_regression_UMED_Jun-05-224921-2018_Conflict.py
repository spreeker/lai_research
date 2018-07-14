"""
- Predict LAI based on the climatic variables.

- Fit a line, y = ax + by + cz + dq + constant
  for each cru lon, lat location on a modis map

- Plot the end result on a map
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import logging
import numpy as np
import os

import math
import modis_map
import read_modis
from datetime import datetime

import settings
from settings import conf
import extract_CRU
import create_lai_cube
import plot_predictors

from plot_map_progress import plot_lai

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


CRU_IDX = ('tmp', 'vap', 'pet', 'pre')

OPTIONS = {
    'debug': False
}


def solver_function_multi(
        lcru, llai, timestamps, predictors=('tmp', 'vap', 'pet', 'pre'),
        label='all', showplot=True):
    """
    Fit a line, y = ax + by + cz + dq + constant,
    through some noisy data-points

    :param lcru:  cru at location (raw , option to still do calculations)
    :param llai:  lai at location (normalized)
    :param timestamps datetimes of time period. used to make graph.
    :param predictors  the dataset names we use to predict.
    :param label  we store this predicted lai under pred_{label}
    :return:  best symbol values and rmse for prediction function in settings.
    """
    measurements = []
    plot_predictors = []

    for ds_key in predictors:
        if type(ds_key) is str:
            input_ar = lcru[:, CRU_IDX.index(ds_key)]
            plot_predictors.append(ds_key)
        else:
            input_ar = ds_key(lcru)
            plot_predictors.append(ds_key.__name__)

        input_ar = normalize(input_ar)
        measurements.append(input_ar)

    y = llai

    measurements.append(np.ones(len(y)))

    # We can rewrite the line equation as y = Ap,
    # where A = [[x 1]] and p = [[m], [c]].
    # Now use lstsq to solve for p:
    A = np.vstack(measurements).T  # [[x  y z q 1]]

    try:
        parameters = np.linalg.lstsq(A, y, rcond=None)[0]
    except ValueError:
        # log.error('missing cru?')
        return

    # log.info(f'parameters: %s', list(zip(predictors, parameters)))

    m = measurements

    y_pred = np.zeros(120)

    for i, p in enumerate(parameters[:-1]):   # we skip K.
        # log.debug('p %s', p)
        y_pred += p * m[i]

    rmse = calc_rmse(y, y_pred)
    log.info('%s RMSE: %s', label, calc_rmse(y, y_pred))

    if not showplot:
        return label, rmse

    # datasets[f'pred_{label}'] = y_pred
    plot_predictors.plot(
        timestamps, y, y_pred,
        measurements, plot_predictors, p_label=label)


def make_local_plot(grid, lai, cru, timestamps, geotransform, projection):
    """
    """
    group = conf['groupname']
    lon = settings.locations[group]['lon']
    lat = settings.locations[group]['lat']
    x, y = read_modis.determine_xy(geotransform, projection, lon, lat)
    pass


def normalize(arr):
    mean = np.mean
    std = np.std
    normalized_data = (arr - mean(arr, axis=0)) / std(arr, axis=0)
    return normalized_data


def calc_rmse(predictions, targets):
    differences = predictions - targets      # the DIFFERENCES.
    differences_squared = differences ** 2   # the SQUARES of ^
    mean_of_differences_squared = differences_squared.mean()  # the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)        # ROOT of ^
    if np.isnan(rmse_val):
        log.error('P %s', predictions)
        log.error('T %s', targets)
    return rmse_val


def valid_box(box):
    """Check if box is valid
    """

    if not box:
        return False

    if any([c is None for c in box]):
        return False

    if len(box) != 4:
        return False

    # if box[1][1] == box[2][1]:
    #     log.debug('same y')
    #     return False

    # if box[0][0] == box[3][0]:
    #     log.debug('same x')
    #     return False

    return True


def extract_grid_data(box, lai, green_m, grid, debug=False):
    """For grid location extract relavant data
    """

    l, r, d, u = box

    log.info('%d:%d %d:%d', u[1], d[1], l[0], r[0])
    cube_lai_at_location = lai[:, u[1]:d[1], l[0]:r[0]]
    # p4grid.extend([pr, pl, pu, pd])
    g_at_loc = green_m[u[1]:d[1], l[0]:r[0]]

    def plot(data):
        valid_px = grid_to_pixels(grid)
        # plot the cru pixel we are working on
        for j, (x, y) in enumerate(valid_px):
            plt.plot(x, y, 'r+')

        plt.imshow(data)
        geotransform, projection, bbox = create_lai_cube.extract_lai_meta()
        plt.show()

    if debug:
        # debug indexing.
        # plot what we are doing.
        # and check if sliceing is going ok
        dlai = np.copy(lai[20, :, :])
        dlai[u[1]:d[1], l[0]:r[0]] = 30
        plot(dlai)

        green_c = np.copy(green_m)
        green_c[u[1]:d[1], l[0]:r[0]] = 30
        plot(green_c)

        plt.imshow(g_at_loc)
        # plt.colorbar()
        plt.show()
        # green_mask = np.logical_and(g_at_loc > 0
        plt.imshow(g_at_loc)
        # plt.colorbar()
        plt.show()

    return cube_lai_at_location, g_at_loc


def _rmse_one_location(
        models, grid, i, g, box,
        cru, lai, green_m, timestamps,
        grid_model_rmse, invalid):
    """
    Find lowest rmse for models of one cru grid location in map/dataset
    """

    cru_at_location = cru[i, :, :]

    cube_lai_at_location, green_mask = extract_grid_data(
        box, lai, green_m, grid, debug=OPTIONS['debug'])

    # validate green mask
    if not np.any(green_mask):
        invalid.append((g, 'nogreen'))
        return

    # we want at least more then 10km²
    if np.sum(green_mask) < 10:
        invalid.append((g, 'nogreen'))
        return

    # create a 3d / cube green mask
    green_mask3d = np.zeros(cube_lai_at_location.shape, dtype=bool)
    green_mask3d[:, :, :] = green_mask[np.newaxis, :, :]
    # set ALL LAI values not in green mask are ZERO
    cube_lai_at_location[~green_mask3d] = 0
    assert cube_lai_at_location.shape == green_mask3d.shape
    # Sum lai values in each layer (month) to array of 120 values
    sum_colums_lai = cube_lai_at_location.sum(axis=1)
    sum_array_lai = sum_colums_lai.sum(axis=1)
    # we have summed up lai values of 120 months
    assert sum_array_lai.size == 120

    # print(sum_array_lai)
    # normalize input data

    # cru_at_location = normalize(cru_at_location)
    avg_lai_at_location = normalize(sum_array_lai)

    # print(avg_lai_at_location)
    if np.isnan(avg_lai_at_location).any():
        log.error('%s %s', g, sum_array_lai)
        invalid.append((g, 'nolai'))
        return

    for label, p_labels in models.items():
        # label = '%s %s' % (label, g)
        answer = solver_function_multi(
            cru_at_location, avg_lai_at_location,
            timestamps, p_labels, label=label)

        if not answer:
            log.error(answer)
            invalid.append((g, 'normse'))
            return

        m, rmse = answer

        grid_model_rmse[g].append((rmse, m))


def calculate_models_for_grid(
        models, cru, lai, green, timestamps, grid, boxes):
    """
    For location in box. which is a cru location, find cru data,

    Parameters

    models:       Dict of models evaluate for each location
    cru:          925*120*4 of cru information
    lai:          1200*1200*120 cube of lai information
    green:        1km² locations of green
    timestamp:    10 years in months (120)
    param grid:   all cru lon, lat of current lai data location of world
    param boxes:  4 points in pixel location, up, down, left, right.
    """
    invalid = []
    grid_model_rmse = {}
    green_m = np.logical_and(green > 0, green < 5)

    for i, g in enumerate(grid):
        # add default value.
        g = tuple(g)
        # values should be between 0, 1, 2 will be masked
        grid_model_rmse[g] = [(2, '')]
        box = boxes[i*4:i*4+4]
        # print(box)

        if not valid_box(box):
            invalid.append((g, 'bbox'))
            continue

        log.error('%d %d %s', i, len(grid), box)

        log.debug('OK %s', box)

        _rmse_one_location(
            models, grid, i, g, box, cru,
            lai, green_m, timestamps,
            grid_model_rmse, invalid)

    return grid_model_rmse, invalid


def aic_criterion(models_to_make, datasets):
    # load hdf5 measurement data.
    lai = datasets['lai']
    for p, ds_label in models_to_make.items():
        p_label = f'pred_{p}'
        predicted_lai = datasets[p_label]
        R = np.square(lai - predicted_lai).sum()
        # print(R)
        m = len(ds_label) # len variables
        n = len(lai)      # measurements
        A = n * math.log((2*math.pi)/n) + n + 2 + n * math.log(R) + 2 * m
        print('%s %.4f' % (p, A))


def tmp_gdd(lcru):
    """
    temperature below 5.
    """
    tmp = np.copy(lcru[:, CRU_IDX.index('tmp')])
    tmp[tmp < 5] = 0
    return tmp


def tmp_one(lcru):
    tmp = np.copy(lcru[:, CRU_IDX.index('tmp')])
    tmpone = np.roll(tmp, 1)
    return tmpone


def pet_one(lcru):
    tmp = np.copy(lcru[:, CRU_IDX.index('pet')])
    tmpone = np.roll(tmp, 1)
    return tmpone


def vap_one(lcru):
    tmp = np.copy(lcru[:, CRU_IDX.index('vap')])
    tmpone = np.roll(tmp, 1)
    return tmpone


def pre_one(lcru):
    pre = np.copy(lcru[:, CRU_IDX.index('pre')])
    preone = np.roll(pre, 1)
    return preone


def find_green_location_mask(green):
    g = green
    m = np.logical_and(g > 0, g < 5)
    return m


def make_box_grid(grid):
    """
    For each point find 4 neighboring points to calculate ~exact area to
    take green points from.
    """
    # for grid item find 4 neigbouring points in middle.
    p4grid = []

    for (lon, lat) in grid:
        pr = (lon + 0.15, lat)
        pl = (lon - 0.15, lat)
        pu = (lon, lat + 0.20)
        pd = (lon, lat - 0.20)
        p4grid.extend([pl, pr, pd, pu])
        # log.debug(f'{lon}:{lat}: {pr}{pl}{pu}{pd}')

    return p4grid


def make_basemap(extent):
    # create map using BASEMAP
    m = Basemap(
        llcrnrlon=extent[0], llcrnrlat=extent[3],
        urcrnrlon=extent[1], urcrnrlat=extent[2],
        # lat_0=(lat_max - lat_min) / 2,
        # lon_0=(lon_max - lon_min) / 2,
        projection='cyl',
        # projection='cyl',
        resolution='h',
        # area_thresh=10000.,
    )

    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    parallels = np.arange(49., 79., .5)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 49., .5)
    m.drawmeridians(meridians, labels=[False, True, True, False])

    plt.tight_layout()

    return m


def plot_errors(m, invalid):
    """Plot data errors on map
    """

    for (lon, lat), reason in invalid:
        if reason == 'bbox':
            m.scatter(lon, lat, c='yellow', latlon=True, marker='x')
        elif reason == 'nogreen':
            m.scatter(lon, lat, c='green', latlon=True, marker='8')
        elif reason == 'nolai':
            m.scatter(lon, lat, c='blue', latlon=True, marker='v')
        elif reason == 'normse':
            m.scatter(lon, lat, c='red', latlon=True, marker='>')
        else:
            raise ValueError('unknown reason')


def _plot_and_save(m, cmap, data_g, title):
    """Plot world map and save output to a location
    """
    data_g = data_g.ReadAsArray()
    d = np.flipud(data_g)
    m.imshow(d, vmin=1, vmax=5, cmap=cmap)
    plt.title(title)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    fig1 = plt.gcf()
    plt.show()
    d = datetime.now()
    date = f'{d.year}-{d.month}-{d.day}'
    imgtarget = os.path.join('imgs', conf['groupname'], f'{date}-{title}.png')
    fig1.savefig(imgtarget)


def plot_models(models, extent, lons, lats, data_g, invalid, title):

    m = make_basemap(extent)

    model_keys = list(MODEL_OPTIONS.keys())
    model_keys.sort()

    data = []

    for mdl in models:
        if mdl in model_keys:
            data.append(model_keys.index(mdl))
        else:
            data.append(-1)

    models_data = np.array(data)
    mdata = models_data.reshape(len(lons), len(lats))

    cmap = plt.get_cmap('RdBu', 4)
    cmap.set_under('0.8')
    cmap.set_bad('0.8')
    cmap.set_over('0.8')

    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(mdata), clip=True)

    valid = np.ma.masked_where(mdata == -1, mdata)

    im = m.pcolormesh(
        # make sure suqares are over the crosses
        lons - 0.25, lats - 0.25, valid, vmin=-.5, vmax=len(model_keys)-.5,
        norm=norm,
        # edgecolors='None',
        latlon=True, cmap=cmap, alpha=0.6)

    cbar = m.colorbar(im)
    cbar.ax.set_ylabel(" ".join(model_keys))

    plot_errors(m, invalid)

    _plot_and_save(m, cmap, data_g, title)


def plot_scores(scores, extent, lons, lats, data_g, invalid, title):

    m = make_basemap(extent)

    mscores = np.ma.masked_where(scores > 1, scores)

    im = m.pcolormesh(
        # make sure suqares are over the crosses
        lons - 0.25, lats - 0.25, mscores, vmin=0, vmax=1,
        latlon=True, cmap='RdBu_r', alpha=0.4)

    m.colorbar(im)

    cmap = plt.cm.Greens
    cmap.set_under('0.8')
    cmap.set_bad('0.8')
    cmap.set_over('0.8')

    plot_errors(m, invalid)

    scores[scores > 1] = 0
    mean_rmse = scores.mean()

    _plot_and_save(m, cmap, data_g, f'{title} - mean {mean_rmse}')



def plot_model_map(green, grid_model_rmse, invalid, title='rmse'):
    geotransform, projection, bbox = create_lai_cube.extract_lai_meta()

    grid = grid_model_rmse.keys()
    grid_model_rmse.values()

    data_g = modis_map.reproject_dataset(
        green, geotransform=geotransform, x_size=1200, y_size=1200)

    geo, pro = read_modis.get_meta_geo_info(data_g)

    lons = []
    lats = []

    for x, y in grid:
        lons.append(x)
        lats.append(y)

    lons, lats = np.meshgrid(lons, lats)
    scores = []
    models = []

    for x, y in zip(lons.flatten(), lats.flatten()):
        rmse_model_list = grid_model_rmse[tuple((x, y))]
        # order by score
        rmse_model_list.sort()
        # lowest rmse
        scores.append(rmse_model_list[0][0])
        # extract the bet model
        models.append(rmse_model_list[0][1])

    scores = np.array(scores)
    scores = scores.reshape(len(lons), len(lats))

    extent = [
        geo[0], geo[0] + data_g.RasterXSize*geo[1],
        geo[3], geo[3] + data_g.RasterYSize*geo[5]]

    # plot_scores(scores, extent, lons, lats, data_g, invalid, "rmse scores")
    plot_models(models, extent, lons, lats, data_g, invalid, "models")


def grid_to_pixels(grid):

    geotransform, projection, bbox = create_lai_cube.extract_lai_meta()

    valid_px = extract_CRU.find_xy_cru_grid(
        geotransform, projection, 1200, 1200, grid)

    return valid_px


def plot_layer(x, lai, green, grid):
    """
    Plot layer fromm lai cube with 2 projections
    each with lon, lat grid of cru data
    """
    onelayer = lai[x, :, :]

    geotransform, projection, bbox = create_lai_cube.extract_lai_meta()

    data = modis_map.reproject_dataset(
        onelayer, geotransform=geotransform, x_size=1200, y_size=1200)

    data_g = modis_map.reproject_dataset(
        green, geotransform=geotransform, x_size=1200, y_size=1200)

    geo, pro = read_modis.get_meta_geo_info(data)

    points4326 = [read_modis.coord2pixel(geo, lon, lat) for lon, lat in grid]

    boxgrid = make_box_grid(grid)

    # valid = []

    # for i in range(len(grid)):
    #     x, y = points4326[i]
    #     if 0 > x  or x > data.RasterXSize:
    #         continue
    #     if 0 > y or y > data.RasterYSize:
    #         continue
    #     valid.append(grid[i])

    valid_px = extract_CRU.find_xy_cru_grid(
        geotransform, projection, 1200, 1200, grid)

    box_px = extract_CRU.find_xy_cru_grid(
        geotransform, projection, 1200, 1200, boxgrid)

    # no projection raw data
    # plot_lai(onelayer, green, valid_px, title='one layer of cube')

    points4326px = extract_CRU.cru_filter(
        points4326, data.RasterXSize, data.RasterYSize)

    extent = [
        geo[0], geo[0] + data.RasterXSize*geo[1],
        geo[3], geo[3] + data.RasterYSize*geo[5]]

    # create map using BASEMAP
    m = Basemap(
        llcrnrlon=extent[0], llcrnrlat=extent[3],
        urcrnrlon=extent[1], urcrnrlat=extent[2],
        # lat_0=(lat_max - lat_min) / 2,
        # lon_0=(lon_max - lon_min) / 2,
        projection='cyl',
        # projection='cyl',
        resolution='h',
        # area_thresh=10000.,
    )

    cmap = plt.cm.gist_rainbow
    cmap.set_under('0.8')
    cmap.set_bad('0.8')
    cmap.set_over('0.8')

    d = data.ReadAsArray()
    data_g = data_g.ReadAsArray()
    control = np.copy(d)
    d = np.flipud(d)
    m.imshow(d, vmin=0, vmax=40, cmap=cmap)

    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    parallels = np.arange(49., 79., .5)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 49., .5)
    m.drawmeridians(meridians, labels=[False, True, True, False])

    plt.tight_layout()
    plt.show()

    plot_lai(control, data_g, points4326px)

    return box_px, boxgrid


def _plot_rmse_each_model(model_options, cru, lai, green, timestamps, grid, box_px):

    for k, v in model_options.items():
        title = k
        model_option = {k: v}
        locations_model_rmse, invalid = calculate_models_for_grid(
            model_option, cru, lai, green, timestamps, grid, box_px)

        # print(len(locations_model_rmse))
        plot_model_map(green, locations_model_rmse, invalid, title=title)


#MODEL_COLOR_MAP = {
#    'tmp': 'red',
#    'vap': 'orange',
#    'pre': 'green',
#    'pet': 'blue',
#}

# load hdf5 measurement data.
MODEL_OPTIONS = {
    # 'p4': ['pre', 'pet', 'vap', 'tmp'],
    # 'p3_tmp-vap-pet': ['tmp', 'vap', 'pet'],
    # 'p2_vap_pre': ['vap', 'pre',],
    # 'p2_vap_pet': ['vap', 'pet',],
    # 'p2_pet_pre': ['pet', 'pre',],
    # 'p2_tv': ['tmp', 'vap'],
    # 'p3_pre_pre_pet': [pre_one, 'pre', 'pet'],
    # 'gdd2': [tmp_gdd, pre_one],
    # 'tmp_pet_vap_tmp5': ['vap', 'tmp', 'pet', pet_one, tmp_one, tmp_gdd],
    # 'p6': ['pre', 'vap', 'pet', pre_one, pet_one, vap_one],
    # 'gdd_tmp': [tmp_gdd, 'tmp'],
    # 'gdd_tmp_vap': [tmp_gdd, 'tmp', 'vap'],
    'tmp': ['tmp'],
    'vap': ['vap'],
    'pre': ['pre'],
    'pet': ['pet'],
}


def main(args):

    if args.debug:
        OPTIONS['debug'] = True

    model_options = MODEL_OPTIONS
    import h5util

    h5util.print_paths()
    grid = h5util.load_dataset('grid')
    green = h5util.load_dataset('green')
    lai = h5util.load_dataset('lai/smooth_month')
    # lai = h5util.load_dataset('lai/month')
    cru = h5util.load_dataset('cru')
    timestamps = [
        datetime.fromtimestamp(t) for t in h5util.load_dataset('months')]

    geotransform, projection, bbox = create_lai_cube.extract_lai_meta()

    if OPTIONS['debug']:
        box_px, box_lon_lat = plot_layer(20, lai, green, grid)

    boxgrid = make_box_grid(grid)
    box_px = []

    if OPTIONS['plotlocation']:
        make_local_plot(
            grid,
            lai, cru, timestamps,
            geotransform, projection)
        return


    # plot_layer(20, lai, green, grid)
    # return
    for lon, lat in boxgrid:
        x, y = read_modis.determine_xy(geotransform, projection, lon, lat)
        box_px.append((int(x), int(y)))

    for i, (x, y) in enumerate(list(box_px)):
        if 0 > x or x > 1199:
            box_px[i] = None
            continue
        if 0 > y or y > 1199:
            box_px[i] = None
            continue

    assert len(grid)*4 == len(boxgrid), f'{len(grid)} {len(boxgrid)}'

    # make sure grid and boxgrid match
    for g, b in zip(grid, boxgrid[::4]):
        log.debug('%s %s', g, b)

    # aic_criterion(model_options, datasets)
    # _plot_rmse_each_model(model_options, cru, lai, green, timestamps, grid, box_px)

    # return

    locations_model_rmse, invalid = calculate_models_for_grid(
        model_options, cru, lai, green, timestamps, grid, box_px)

    # print(len(locations_model_rmse))
    plot_model_map(green, locations_model_rmse, invalid, title='lowest rmse')


if __name__ == '__main__':
    desc = "Create GREEN matrix cubes of LAI"
    inputparser = argparse.ArgumentParser(desc)

    inputparser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help="Print raw hdf landuse data from direcotory")

    inputparser.add_argument(
        '--plotlocation',
        action='store_true',
        default=False,
        help="Make predictor plots of locations definded in settings")

    args = inputparser.parse_args()
    main(args)
