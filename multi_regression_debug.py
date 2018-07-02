"""
Predict LAI based on the climatic variables.
Fit a line, y = ax + by + cz + dq + constant
"""

import matplotlib.pyplot as plt
import logging
import osr
import numpy as np
from mpl_toolkits.basemap import Basemap

import h5util
import modis_map
import extract_CRU
import create_lai_cube
from plot_map_progress import plot_lai
import read_modis

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def main():

    h5util.print_paths()
    grid = h5util.load_dataset('grid')
    green = h5util.load_dataset('green')
    lai = h5util.load_dataset('lai/smooth_month')
    geotransform, projection, bbox = create_lai_cube.extract_lai_meta()

    # print(geotransform)
    # print(projection)
    # print(bbox)

    points = extract_CRU.find_xy_cru_grid(
        geotransform, projection, 1200, 1200, grid)

    d = np.copy(lai[1, :, :])

    # plot x by x squares
    for x, y in extract_CRU.find_xy_cru_grid(
            geotransform, projection, 1200, 1200, grid):
        d[y-2:y+2, x-2:x+2] = 30

    data = modis_map.reproject_dataset(
        d, geotransform=geotransform, x_size=1200, y_size=1200)

    data_g = modis_map.reproject_dataset(
        green, geotransform=geotransform, x_size=1200, y_size=1200)

    geo, pro = read_modis.get_meta_geo_info(data)
    # convert lon lat to pixel location
    points4326 = [read_modis.coord2pixel(geo, lon, lat) for lon, lat in grid]
    points4326 = extract_CRU.cru_filter(
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

    # for x, y in grid:
    #     m.scatter(x, y, latlon=True)

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

    # cb = plt.colorbar(orientation='horizontal', fraction=0.10, shrink=0.8)
    plt.tight_layout()
    plt.show()

    plot_lai(lai[1, :, :], green, points)

    plot_lai(control, data_g, points4326)

    # timestamps, datasets = load_data()
    # make_models(models_options, datasets, timestamps)
    # aic_criterion(models_options, datasets)


if __name__ == '__main__':
    main()
