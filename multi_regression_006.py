"""
- Predict LAI based on the climatic variables.

- Fit a line, y = ax + by + cz + dq + constant
  for each cru lon, lat location on a modis map

- Plot the end result on a map
"""
import os
# import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import logging
import numpy as np

import math
import modis_map
import read_modis
from datetime import datetime

import settings
from settings import conf
import extract_CRU
import create_lai_cube_v006
import plot_predictors_v006

from plot_map_progress import plot_lai

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


LOCATIONS_MODEL_RMSE = {}

CRU_IDX = ('tmp', 'vap', 'pet', 'pre')

OPTIONS = {
    'debug': False
    # 'debug': True
}

MODEL_OPTIONS = {
    # 'p4': ['pre', 'pet', 'vap', 'tmp'],
    # 'p3_tmp-vap-pet': ['tmp', 'vap', 'pet'],
    # 'p2_vap_pre': ['vap', 'pre'],
    # 'p3_vap_pre_tmp_one': ['vap', 'pre', tmp_one],
    # 'p2_vap_pet': ['vap', 'pet',],
    # 'p2_pet_pre': ['pet', 'pre',],
    # 'p2_tv': ['tmp', 'vap'],
    # 'p3_pre_pre_pet': [pre_one, 'pre', 'pet'],
    # 'gdd2': [tmp_gdd, pre_one],
    # 'tmp_pet_vap_tmp5': ['vap', 'tmp', 'pet', pet_one, tmp_one, tmp_gdd],
    # 'p6': ['tmp', 'pre', 'vap', 'pet', pre_one],
    # 'gdd_tmp': [tmp_gdd, 'tmp'],
    # 'gdd_tmp_vap': [tmp_gdd, 'tmp', 'vap'],
    'tmp': ['tmp'],
    'vap': ['vap'],
    'pre': ['pre'],
    'pet': ['pet'],
}


model_keys = list(MODEL_OPTIONS.keys())
model_keys.sort()

log.debug(model_keys)
# pickle_file = 'europe_21658.pickle'


def save_result_to_csv(locations_model_rmse):
    # size = len(LOCATIONS_MODEL_RMSE)
    filename = f'result-run.csv'
    if conf.get('csv'):
        filename = conf['csv']

    addheader = True
    if os.path.exists(filename):
        addheader = False

    with open(filename, 'a') as csv:
        if addheader:
            csv.write('lon,lat,%s\n' % (" ".join(model_keys)))
        for (lon, lat), models in locations_model_rmse.items():
            models.sort()
            mdl = models[0][1]
            if mdl not in model_keys:
                continue
            idx = model_keys.index(mdl)
            csv.write('%.2f,%.2f,%s\n' % (lon, lat, idx))

    log.info('saved %s', filename)


def solver_function_multi(
        lcru, llai, timestamps, predictors=('tmp', 'vap', 'pet', 'pre'),
        label='all', showplot=False, valid=None):
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
    plot_predictor_labels = []

    if not valid:
        # incase not everythin is downloaded and we have
        # time gap
        raise ValueError('valid idx locations are needed')

    y = llai  # alreadu filtered with [valid]

    for ds_key in predictors:
        if type(ds_key) is str:
            input_ar = lcru[:, CRU_IDX.index(ds_key)]
            plot_predictor_labels.append(ds_key)
        else:
            input_ar = ds_key(lcru)
            plot_predictor_labels.append(ds_key.__name__)

        input_ar = normalize(input_ar, valid)
        assert len(input_ar) == len(y)
        measurements.append(input_ar)

    measurements.append(np.ones(len(y)))

    # We can rewrite the line equation as y = Ap,
    # where A = [[x 1]] and p = [[m], [c]].
    # Now use lstsq to solve for p:
    A = np.vstack(measurements).T  # [[x  y z q 1]]

    try:
        parameters = np.linalg.lstsq(A, y, rcond=None)[0]
    except ValueError:
        log.error('missing cru?')
        log.error('missing lai?')
        log.error(y)
        log.error(measurements[0])
        return

    predictor_params = "parameters: "
    for l, p in zip(plot_predictor_labels, parameters):
        log.debug(l)
        log.debug(p)
        predictor_params += ' %s %.4f ' % (l, p)

    m = measurements
    y_pred = np.zeros(len(y))

    # for i, p in enumerate(parameters[:-1]):   # we skip K.
    for i, p in enumerate(parameters):
        # log.debug('p %s', p)
        y_pred += p * m[i]

    rmse = calc_rmse(y, y_pred)
    log.info('%s RMSE: %s', label, calc_rmse(y, y_pred))

    if not showplot:
        return label, rmse

    timestamps = np.array(timestamps)
    timestamps = timestamps[valid]
    # datasets[f'pred_{label}'] = y_pred
    plot_predictors_v006.plot(
        timestamps, y, y_pred,
        measurements, predictors=plot_predictor_labels, p_label=label,
        text=predictor_params,
    )

    calculate_ss(y, y_pred, measurements, plot_predictor_labels, parameters)


def calculate_ss(y, y_pred, measurements, plot_predictor_labels, parameters):
    """Calculate SSerr SStot, SSreg, R2,

    not means are always zero since we have standardized the data.
    """
    n = len(y)
    log.info(conf['groupname'])
    log.info(plot_predictor_labels)
    log.info(parameters)
    ss_err = np.power(y - y_pred, 2).sum() / n
    ss_tot = np.power(y, 2).sum() / n
    ss_tot_p = np.power(y_pred, 2).sum() / n

    log.info('ss_err %.3f', ss_err)
    log.info('ss_tot %.3f', ss_tot)
    log.info('ss_tot_p %.3f', ss_tot_p)

    sum_r = 0
    for m, l, p in zip(measurements, plot_predictor_labels, parameters):
        # mss_tot = np.power(m, 2).sum() / n
        mss_reg = np.power(m * p, 2).sum() / n
        # mss_reg = np.power(mp, 2).sum() / n
        log.info('%s %.2f ss_reg %.3f', l, p, mss_reg)
        sum_r += mss_reg

    log.info('sum regression %s', sum_r)
    log.info('sum regression + err %.2f', sum_r + ss_err)
    # fraction of vaiance unexplained.
    fvu = ss_err / ss_tot

    log.info('fvu %.3f', fvu)
    log.info('R2 %.3f', 1 - fvu)


def make_local_plot(grid, lai, cru, timestamps, geotransform, projection):
    """Plot LAI, predicted LAI and predictors (temp, vap, pre, pet)
    of lon lat location defined in settings for current group
    """
    group = conf['groupname']
    lon = settings.locations[group]['lon']
    lat = settings.locations[group]['lat']
    x, y = read_modis.determine_xy(geotransform, projection, lon, lat)

    log.debug('%s %s', x, y)
    lai_at_location = lai[:, int(y), int(x)]
    # valid = np.nonzero(lai_at_location)
    valid = np.where(lai_at_location > -1)

    lai_at_location = normalize(lai_at_location, valid)

    # log.debug(lai_at_location)
    # find closest cru data
    min_i = -1
    min_d = 999999999999

    for i, (glon, glat) in enumerate(grid):
        distance = (lon-glon)**2 + (lat-glat)**2
        if distance < min_d:
            min_d = distance
            min_i = i
            # log.debug(f'Distance**2 = {min_d} {glon}, {glat} -> {lon} {lat}')

    cru_at_location = cru[min_i, :, :]

    for label, predictors in MODEL_OPTIONS.items():
        solver_function_multi(
            cru_at_location, lai_at_location, timestamps,
            predictors=predictors,
            label=f'{group}-{label}', showplot=True,
            valid=valid
        )


def normalize(arr_source, valid):
    """standardize arr with average around 0
    """
    # only use time-measurments where we have
    # valid lai data. Sometimes we only downloaded part of
    # the timeseries. and we have gaps / zeros.
    arr = arr_source[valid]
    mean = np.mean
    std = np.std
    normalized_data = (arr - mean(arr, axis=0)) / std(arr, axis=0)
    # normalized_data = normalized_data / abs(normalized_data).max()
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
    """For grid location extract relevant data
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
        geotransform, projection, bbox = \
            create_lai_cube_v006.extract_lai_meta_v006()
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


def _collect_lai_data_location(grid, i, g, box, lai, green_m, invalid):
    """
    For one location extract data from LAI, green, CRU to do calculations.
    """

    cube_lai_at_location, green_mask = extract_grid_data(
        box, lai, green_m, grid, debug=OPTIONS['debug'])

    # validate green mask
    if not np.any(green_mask):
        invalid.append((g, 'nogreen'))
        return

    # we want at least more then 10km²
    if np.sum(green_mask) < 40:
        invalid.append((g, 'nogreen'))
        return

    return cube_lai_at_location, green_mask


def _normalized_lai_at_location(green_mask, cube_lai_at_location, g, invalid):
    """
    Given green mask. create masked lai cube for location
    """
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
    # in case of missing lai data.

    valid = np.where(sum_array_lai > 0)
    # cru_at_location = normalize(cru_at_location)
    # sum_array_lai[valid] / valid[0].shape[0]

    avg_lai_at_location = normalize(sum_array_lai, valid)

    # print(avg_lai_at_location)
    if np.isnan(avg_lai_at_location).any():
        log.error('%s %s', g, sum_array_lai)
        invalid.append((g, 'nolai'))
        return None, None

    # print(avg_lai_at_location)
    if avg_lai_at_location.size == 0:
        log.error('%s %s', g, sum_array_lai)
        invalid.append((g, 'nolai'))
        return None, None

    return avg_lai_at_location, valid


def _rmse_one_location(
        cru_at_location,
        avg_lai_at_location,
        valid, g,
        models, timestamps,
        grid_model_rmse, invalid):
    """
    Find lowest rmse for models of one cru grid
    location in map/dataset
    """

    for label, p_labels in models.items():
        # label = '%s %s' % (label, g)
        answer = solver_function_multi(
            cru_at_location, avg_lai_at_location,
            timestamps, predictors=p_labels, label=label,
            valid=valid   # non zero values
        )

        if answer is None:
            log.error('ANSWER %s', answer)
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
    lai:          2400*2400*120 cube of lai information
    green:        1km² locations of green
    timestamp:    10 years in months (120)
    param grid:   all cru lon, lat of current lai data location of world
    param boxes:  4 points in pixel location, up, down, left, right.

    :grid_model_rmse {
        (lon,lat): [(score, model)..],
        ...
    }
    :invalid {
        (lon,lat): errorcode
    }
    """
    invalid = []
    grid_model_rmse = {}

    min_g, max_g = conf['greenrange']

    green_m = np.logical_and(green > min_g, green < max_g)

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

        cru_at_location = cru[i, :, :]

        lai_cube, green_mask = _collect_lai_data_location(
            grid, i, g, box, lai, green_m, invalid)

        avg_lai_at_location, valid = _normalized_lai_at_location(
            green_mask, lai_cube, g, invalid)

        _rmse_one_location(
            cru_at_location,
            avg_lai_at_location, valid, g,
            models,
            timestamps,
            grid_model_rmse, invalid)

    del green_m

    return grid_model_rmse, invalid


def aic_criterion(models_to_make, datasets):
    # load hdf5 measurement data.
    lai = datasets['lai']
    for p, ds_label in models_to_make.items():
        p_label = f'pred_{p}'
        predicted_lai = datasets[p_label]
        R = np.square(lai - predicted_lai).sum()
        # print(R)
        m = len(ds_label)  # len variables
        n = len(lai)       # measurements
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
        pr = (lon + 0.19, lat)
        pl = (lon - 0.19, lat)
        pu = (lon, lat + 0.24)
        pd = (lon, lat - 0.24)
        p4grid.extend([pl, pr, pd, pu])
        # log.debug(f'{lon}:{lat}: {pr}{pl}{pu}{pd}')

    return p4grid


def make_basemap(_extent):
    # create map using BASEMAP
    m = Basemap(
        # llcrnrlon=extent[0], llcrnrlat=extent[3],
        # llcrnrlon=-25, llcrnrlat=30,
        # urcrnrlon=extent[1], urcrnrlat=extent[2],
        # urcrnrlon=20, urcrnrlat=70,
        # lat_0=(lat_max - lat_min) / 2,
        # lon_0=(lon_max - lon_min) / 2,
        projection='cyl',
        # projection='cyl',
        # resolution='h',
        # area_thresh=10000.,
    )
    m.drawcoastlines(linewidth=0.5)
    # m.drawcountries(linewidth=0.5)
    # parallels = np.arange(-90, 90, .5)
    # labels = [left,right,top,bottom]
    # m.drawparallels(parallels, labels=[False, True, True, False])
    # meridians = np.arange(-180, 180, .5)
    # m.drawmeridians(meridians, labels=[False, True, True, False])
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


def _plot_and_save(m, cmap, title, i=''):
    """Plot world map and save output to a location
    """
    # data_g = data_g.ReadAsArray()
    # d = np.flipud(data_g)
    # m.imshow(d, vmin=1, vmax=5, cmap=cmap)
    plt.title(title)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    fig1 = plt.gcf()
    plt.show()
    d = datetime.now()
    date = f'{d.year}-{d.month}-{d.day}'
    tile = conf['groupname']
    imgtarget = os.path.join('imgs', 'modis', f'{tile}-{date}-{title}{i}.png')
    fig1.savefig(imgtarget)


def plot_models(data, _extent, lons, lats, invalid, title):

    assert data.size
    # we no longer use extent
    m = make_basemap(_extent)

    mdata = data.reshape(len(lons), len(lats))

    cmap = plt.get_cmap('RdBu', 4)
    cmap.set_under('0.8')
    cmap.set_bad('0.8')
    cmap.set_over('0.8')

    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=True)

    valid = np.ma.masked_where(mdata == -1, mdata)

    im = m.pcolormesh(
        # make sure squares are over the crosses
        lons - 0.25, lats - 0.25,
        valid,    # makes model data
        vmin=-.5, vmax=3.5,
        norm=norm,
        # edgecolors='None',
        latlon=True, cmap=cmap, alpha=0.8)

    cbar = m.colorbar(im)
    cbar.ax.set_ylabel(" ".join(model_keys))

    # plot_errors(m, invalid)

    _plot_and_save(m, cmap, title, i=data.size)
    # help python cleanup
    del m
    del im
    del valid


def plot_scores(scores, extent, lons, lats, _data_g, invalid, title):

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

    _plot_and_save(m, cmap, f'{title} - mean {mean_rmse}')


def plot_model_map(green, grid_model_rmse, invalid, title=None):

    grid = grid_model_rmse.keys()
    grid_model_rmse.values()
    lons = []
    lats = []

    for x, y in grid:
        lons.append(x)
        lats.append(y)

    lons = np.array(lons)
    lats = np.array(lats)

    lons.sort()
    lats.sort()

    lons, lats = np.meshgrid(lons, lats)

    model_keys = list(MODEL_OPTIONS.keys())
    model_keys.sort()

    data = []

    log.debug('create/fill grid..')

    for x, y in zip(lons.flatten(), lats.flatten()):
        rmse_model_list = grid_model_rmse.get((x, y))
        if not rmse_model_list:
            # invalid clipped values
            data.append(-1)
            continue
        rmse_model_list.sort()
        # lowest rmse
        # extract the bet model
        mdl = rmse_model_list[0][1]
        if mdl == '':
            data.append(-1)
            continue

        data.append(model_keys.index(mdl))

    data = np.array(data)

    log.debug('starting plot..')

    plot_models(data, None, lons, lats, invalid, f"{title} models")


def green_data(green):

    x_size = 2400
    y_size = 2400

    geotransform, projection, bbox =  \
        create_lai_cube_v006.extract_lai_meta_v006()

    data_g = modis_map.reproject_dataset(
        green, geotransform=geotransform, x_size=x_size, y_size=y_size)

    geo, pro = read_modis.get_meta_geo_info(data_g)

    extent = [
        geo[0], geo[0] + data_g.RasterXSize*geo[1],
        geo[3], geo[3] + data_g.RasterYSize*geo[5]]

    return extent


def grid_to_pixels(grid):

    geotransform, projection, bbox = \
        create_lai_cube_v006.extract_lai_meta_v006()

    valid_px = extract_CRU.find_xy_cru_grid(
        geotransform, projection, 2400, 2400, grid)

    return valid_px


def plot_layer(x, lai, green, grid):
    """
    Plot layer fromm lai cube with 2 projections
    each with lon, lat grid of cru data
    """
    x_size = 2400
    y_size = 2400

    onelayer = lai[x, :, :]

    geotransform, projection, bbox =  \
        create_lai_cube_v006.extract_lai_meta_v006()

    data = modis_map.reproject_dataset(
        onelayer, geotransform=geotransform, x_size=x_size, y_size=y_size)

    data_g = modis_map.reproject_dataset(
        green, geotransform=geotransform, x_size=x_size, y_size=y_size)

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
        geotransform, projection, x_size, y_size, grid)

    box_px = extract_CRU.find_xy_cru_grid(
        geotransform, projection, x_size, y_size, boxgrid)

    # no projection raw data
    plot_lai(onelayer, green, valid_px, title='one layer of cube')

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

    del data_g

    return box_px, boxgrid


def _plot_rmse_each_model(
        model_options, cru, lai, green, timestamps, grid, box_px):

    for k, v in model_options.items():
        title = k
        model_option = {k: v}

        locations_model_rmse, invalid = calculate_models_for_grid(
            model_option, cru, lai, green, timestamps, grid, box_px)

        print(len(locations_model_rmse))
        plot_model_map(green, locations_model_rmse, invalid, title=title)


def create_extraction_areas(boxgrid, geotransform, projection):
    """We determine the boundaties for data extraction of lai / green near grid.

    translate lon, lat to pixel bboxes
    """

    box_px = []
    # plot_layer(20, lai, green, grid)
    # return
    for lon, lat in boxgrid:
        x, y = read_modis.determine_xy(geotransform, projection, lon, lat)
        box_px.append((int(x), int(y)))

    # make sure we extract from within matrix
    for i, (x, y) in enumerate(list(box_px)):
        if 0 > x:
            box_px[i] = (0, y)

        if x > 2399:
            box_px[i] = (2399, y)

        if 0 > y:
            x = box_px[i][0]
            box_px[i] = (x, 0)

        if y > 2399:
            x = box_px[i][0]
            box_px[i] = (x, 2399)

    return box_px


def main_world(plotlocation=False):
    """world mapping entry point. return grid with best models.
    """
    model_options = MODEL_OPTIONS
    import h5util

    # h5util.print_paths()
    try:
        grid = h5util.load_dataset('grid')
    except (OSError, KeyError):
        log.error('hdf5 file missing..')
        return

    green = h5util.load_dataset('green')
    try:
        lai = h5util.load_dataset('lai/smooth_month')
    except (KeyError):
        log.error('no lai data')
        return
    # lai = h5util.load_dataset('lai/month')
    cru = h5util.load_dataset('cru')
    timestamps = [
        datetime.fromtimestamp(t) for t in h5util.load_dataset('months')]

    geotransform, projection, bbox = \
        create_lai_cube_v006.extract_lai_meta_v006()

    boxgrid = make_box_grid(grid)

    # plot graph of single location
    if plotlocation:
        make_local_plot(
            grid,
            lai, cru, timestamps,
            geotransform, projection)
        return
    assert len(grid)*4 == len(boxgrid), f'{len(grid)} {len(boxgrid)}'

    # make sure grid and boxgrid match
    for g, b in zip(grid, boxgrid[::4]):
        log.debug('%s %s', g, b)

    box_px = create_extraction_areas(boxgrid, geotransform, projection)

    locations_model_rmse, invalid = calculate_models_for_grid(
        model_options, cru, lai, green, timestamps, grid, box_px)

    # print(len(locations_model_rmse))
    # plot_model_map(
    #     green, locations_model_rmse, invalid, title='lowest rmse', )

    # update global set
    update_locations_model_rmse(locations_model_rmse)

    # plot global data
    # plot_model_map(green, LOCATIONS_MODEL_RMSE, [], title='world')

    # now we can plot this in qgis
    log.debug('SQUARES %s', len(LOCATIONS_MODEL_RMSE))
    save_result_to_csv(locations_model_rmse)

    # help python cleanup
    del box_px
    del boxgrid
    del grid
    del green
    del lai
    del cru
    del geotransform
    del projection
    del bbox


def update_locations_model_rmse(locations_model_rmse):
    """Update global model result with current models.

    Make sure we have correct edges. Overlapping locations with
    no model information should not overwrite valid data.

    Otherwise we end up with 'lines' in our map.
    """
    keys = list(locations_model_rmse.keys())
    for k in keys:
        # find overlapping locations
        if k in LOCATIONS_MODEL_RMSE:
            # order model scores
            locations_model_rmse[k].sort()
            rmse = locations_model_rmse[k]
            # containes no information if score == 2
            if rmse[0][0] == 2:
                locations_model_rmse.pop(k)
                # nothing was found. do not overwrite possible neighbours.

    LOCATIONS_MODEL_RMSE.update(locations_model_rmse)
