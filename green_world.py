"""
Create some LAI / green stats csv of the globe
"""

import os
import logging
import numpy as np
import h5util
import settings

from multi_regression_006 import _load_current_granule
from multi_regression_006 import make_box_grid
from multi_regression_006 import create_extraction_areas
from multi_regression_006 import _collect_lai_data_location
from multi_regression_006 import valid_box


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


GREEN_TYPES = {
    1: "1-Evergreen_Needleleaf_Trees",
    2: "2-Evergreen_Broadleaf_Trees",
    3: "3-Deciduous_Needleleaf_Trees",
    4: "4-Deciduous_Broadleaf_Trees",
    5: "5-Shrub",
    6: "6-Grass",
    7: "7-Cereal_Crop",
    8: "8-Broadleaf_Crop",
}

STATS = ['green_count', 'min', 'max', 'std', 'mean']


def store_lai_stats_for_location(lai_stats: dict, greentype: int):
    """
    store lai stats for location
    """
    # size = len(LOCATIONS_MODEL_RMSE)
    green_title = GREEN_TYPES[greentype]
    filename = f'laistats/green-stats-{green_title}.csv'

    addheader = True

    if os.path.exists(filename):
        addheader = False

    with open(filename, 'a') as csv:
        if addheader:
            value_header = ",".join(STATS)
            csv.write('lon,lat,%s\n' % (value_header))

        for (lon, lat), values in lai_stats.items():
            csv_values = ",".join(map(str, values))
            csv.write('%.2f,%.2f,%s\n' % (lon, lat, csv_values))

    log.info('saved %s %d', filename, len(lai_stats))


def store_lai_stats_in_world(lai_stats: dict, greentype: int):
    """
    """

    green_title = GREEN_TYPES[greentype]

    grid_idx, lons, lats, empty = h5util.world_grid()

    for i, stat in enumerate(STATS):

        groupname = f'{green_title}-{stat}'

        world_data = h5util.world_data_load(groupname)

        if world_data is None:
            world_data = empty.copy()

        for (lon, lat), values in lai_stats.items():
            x, y = grid_idx[(lon, lat)]
            # the biggest value wins.
            # this takes care of artifacts near the edges of
            # each modis tile whare a square is on both granules.
            # we pick the best one.
            # if world_data[y][x] < values[i]:
            world_data[y][x] = values[i]

        # the amount of non fill values should greater equal
        # to the locations we updated
        assert np.sum(world_data > -1) >= len(lai_stats)
        h5util.save_netcdf(groupname, world_data)


def array_stats(arr_source, valid):
    """ retrun min, max, std and mean of valid values of array source
    """
    # only use time-measurments where we have
    # valid lai data. Sometimes we only downloaded part of
    # the timeseries and we have gaps / zeros.
    arr = arr_source[valid]
    if arr.size == 0:
        return
    # return arr
    min_v = arr.min()
    max_v = arr.max()
    std_v = arr.std()
    mean_v = arr.mean()
    return min_v, max_v, std_v, mean_v


def stats_lai_at_location(green_mask, cube_lai_at_location):
    """
    Given green mask. create masked lai cube for location
    and calculate stats.

    returns

    -min lai
    -max lai
    -mean lai
    -std lai
    -number of valid green pixels
    """
    # create a 3d / cube green mask
    green_mask3d = np.zeros(cube_lai_at_location.shape, dtype=bool)
    green_mask3d[:, :, :] = green_mask[np.newaxis, :, :]
    # set ALL LAI values not in green mask are ZERO
    cube_lai_at_location[~green_mask3d] = 0
    # set all values outside greenvalues to 0
    # > 100 can happen in case of fillvalues
    cube_lai_at_location[cube_lai_at_location > 100] = 0
    assert cube_lai_at_location.shape == green_mask3d.shape
    # Sum lai values in each layer (month) to array of 120 values
    sum_colums_lai = cube_lai_at_location.sum(axis=1)
    sum_array_lai = sum_colums_lai.sum(axis=1)
    # we have summed up lai values of 120 months
    assert sum_array_lai.size == 120
    # in case of missing lai data.
    valid = np.where(sum_array_lai > 0)
    # number of green locations
    g_count = np.count_nonzero(green_mask)
    lai_at_location = sum_array_lai / g_count

    agg_stats = array_stats(lai_at_location, valid)

    if not agg_stats:
        return

    _min, _max, std, mean = agg_stats
    return g_count, _min, _max, std, mean


def _mask_stats(
        g: tuple, masks: list, grid: list, i: int,
        box: list, lai, stat_results: list):
    """For every plant functional type determine stats
    """
    # log.error('%d %d %s', i, len(grid), box)
    for mi, green_m in enumerate(masks):

        # for every green type make stats
        lai_cube_l, green_mask_l = _collect_lai_data_location(
            grid, i, g, box, lai, green_m.copy(), [])

        if lai_cube_l is None:
            continue

        # log.error('%d %d', gi+1, green_mask_l.sum())
        stats_l = stats_lai_at_location(green_mask_l.copy(), lai_cube_l)
        if stats_l is not None:
            stat_results[mi][g] = list(stats_l)


def calculate_laistats_for_grid(lai, green, grid: list, boxes: list):
    """
    For location in box which is a 0.5 cru location, find cru data,

    Parameters

    lai:          2400*2400*120 cube of lai information
    green:        .5 km² locations of green
    timestamp:    10 years in months (120)
    param grid:   all cru lon, lat of current lai data location of world
    param boxes:  4 points in pixel location, up, down, left, right.

    """

    # 8 green masks
    masks = []
    # 8 stats collection
    stat_results = []

    for i in range(1, 9):
        m = green == i
        masks.append(m.copy())
        stat_results.append(dict())

        # for x in range(1, 9):
        #     c = (green == x).sum()
        #     log.debug(f'g{x} {c}')

    for i, g in enumerate(grid):
        # add default value.
        g = tuple(g)
        # values should be between 0, 1, 2 will be masked
        box = boxes[i*4:i*4+4]

        if not valid_box(box):
            continue

        _mask_stats(g, masks, grid, i, box, lai, stat_results)

    return stat_results


def main_green_world():
    """Create green / lai averages
    """

    relevant_data = _load_current_granule()

    if not relevant_data:
        log.debug('not all data could be collected..')
        return

    grid, green, lai, cru, timestamps, geotransform, projection, bbox = \
        relevant_data

    boxgrid = make_box_grid(grid)
    box_px = create_extraction_areas(boxgrid, geotransform, projection)

    for x in range(1, 9):
        c = (green == x).sum()
        log.debug(f'g{x} {c}')

    stat_results = calculate_laistats_for_grid(
        lai, green, grid, box_px)

    # for i, results in enumerate(stat_results):
    for gidx in range(0, 8):
        results = stat_results[gidx]
        # for each green type store results
        store_lai_stats_in_world(results, gidx+1)
        # store_lai_stats_for_location(results, i+1)


def green_type_map(stat):
    """Given 8 green landuse types. Build a map showing the dominant
    land use type for each gid cell. using stat. [min, max, mean, green_count]
    """
    grid_idx, lons, lats, empty = h5util.world_grid()

    green_worlds = []
    # load current maps
    for idx in range(1, 9):
        green_title = GREEN_TYPES[idx]
        groupname = f'{green_title}-{stat}'
        print(groupname)
        world_data = h5util.world_data_load(groupname)
        assert world_data is not None
        green_worlds.append(world_data)

    # start with empty world
    # will contain vegetation type [1..8]
    target = empty
    # will contain stat values for
    # popular vegetantion
    recorded = np.copy(empty)

    for i, g_values in enumerate(green_worlds):
        g_value = i+1
        # where lai counts are greater then recorded.
        # set lai type in target
        # remove invalid value
        g_values[g_values > 100] = 0
        target[g_values > recorded] = g_value
        # update the recorded scores.
        recorded[g_values > recorded] = g_values[g_values > recorded]

    h5util.save_netcdf(f'green_types_{stat}', target)
    h5util.save_netcdf(f'green_values_{stat}', recorded)


def cru_best_map():

    grid_idx, lons, lats, empty = h5util.world_grid()

    green_source = 'green_maps_19.nc'

    settings.conf['world'] = green_source

    best_type_map = h5util.world_data_load('green_types_mean')

    assert best_type_map is not None

    settings.conf['world'] = 'world_rmse_all.nc'

    green_worlds = []

    for idx in range(1, 9):
        plant_layer = []
        for cru in ['tmp', 'vap', 'pet', 'pre']:
            # green_title = GREEN_TYPES[idx]
            groupname = f'{cru}_{idx-1}_{idx+1}'
            print(groupname)
            world_data = h5util.world_data_load(groupname)
            assert world_data is not None
            plant_layer.append(world_data)

        assert len(plant_layer) == 4
        green_worlds.append(plant_layer)

    assert len(green_worlds) == 8

    target = np.copy(empty)

    for idx in range(1, 9):
        empty_x = np.copy(empty)
        empty_x.fill(2)
        cru_x = np.copy(empty)

        for i, cru in enumerate(['tmp', 'vap', 'pet', 'pre']):
            w = green_worlds[idx-1][i]
            valid = np.logical_and(w > 0, w < empty_x)
            empty_x[valid] = w[valid]
            cru_x[valid] = i

        target[best_type_map == idx] = cru_x[best_type_map == idx]

    settings.conf['world'] = green_source
    h5util.save_netcdf(f'green_types_max_mean', target)


def best_cru_variable():
    grid_idx, lons, lats, empty = h5util.world_grid()

    # CRU_VARS = ['pet', 'pre', 'tmp', 'vap']
    CRU_VARS = ['Jolly Formula', 'ANPI', 'SFormula']
    # load current maps
    for g0 in range(8):
        gidx = g0 + 1
        green_title = GREEN_TYPES[gidx]
        target = empty.copy()
        recorded = empty.copy()
        # change default value to higher then 1.
        recorded.fill(2)
        target.fill(-1)

        for ti, cru_var in enumerate(CRU_VARS):
            layer_name = f'{cru_var}_{g0}_{g0+2}'
            cru_rmse = h5util.world_data_load(layer_name)
            target[(cru_rmse > -1) & (cru_rmse < recorded)] = ti
            recorded[cru_rmse < recorded] = cru_rmse[cru_rmse < recorded]

        # Put back fill value for unchanged values
        recorded[recorded == 2] = -1
        type_groupname = f'model-type-{gidx}-{green_title}'
        h5util.save_netcdf(type_groupname, target)
        rmse_groupname = f'model-rmse-{gidx}-{green_title}'
        h5util.save_netcdf(rmse_groupname, recorded)


if __name__ == '__main__':
    """
    """
    #for stat in ['max', 'min', 'mean', 'std']:
    #    green_type_map(stat)
    cru_best_map()

    # green_type_map()
    # best_cru_variable()
