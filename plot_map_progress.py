#!/usr/bin/env python
#
import numpy as np
import time
import extract_green
import numpy
import numpy.ma as ma
import random
import logging
import matplotlib as mpl
from sympy.utilities.lambdify import MATH

import create_lai_cube
import modis_map
# mpl.use('GTk3Agg')
# mpl.use('Qt5Agg')
mpl.rc('figure', figsize=(8, 8))

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

matrix_size = 500

from matplotlib import pyplot as plt

CMAP = mpl.colors.ListedColormap([
    'blue',     # water.
    'green',    # to calculate with
    'red',      # done / loaded
    'darkgray', # not interesting.
])

"""
"MOD15A2 FILL VALUE LEGEND
      255 = _Fillvalue, assigned when:
            * the MODAGAGG suf. reflectance for channel VIS, NIR was assigned its _Fillvalue, or
            * land cover pixel itself was assigned _Fillvalus 255 or 254.
      254 = land cover assigned as perennial salt or inland fresh water.
      253 = land cover assigned as barren, sparse vegetation (rock, tundra, desert.)
      252 = land cover assigned as perennial snow, ice.
      251 = land cover assigned as \"permanent\" wetlands/inundated marshlands.
      250 = land cover assigned as urban/built-up.
      249 = land cover assigned as \"unclassified\" or not able to determine.";
"""


def normalize_fill_values(lai):
    lai[lai == 255] = 13
    lai[lai == 254] = 13
    lai[lai == 253] = 13
    lai[lai == 252] = 14
    lai[lai == 251] = 14
    lai[lai == 250] = 12
    lai[lai == 249] = 12


def _plot_lai(data, points, title):

    cmap = plt.get_cmap('Greens', 14)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.8, .8, .8, 1.0)
    # force the last color entry to be blue
    cmaplist[-3] = (.5, .5, .5, 1.0)
    cmaplist[-2] = (.3, .3, .9, 1.0)
    cmaplist[-1] = (.3, .3, .3, 1.0)

    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=14, clip=True)

    mat = plt.imshow(
        data,
        norm=norm,
        cmap=cmap, vmin = 0-.5, vmax=14 + .5)
    plt.colorbar(mat, ticks=np.arange(0, 14+1))

    # plt.imshow(data, norm=norm, cmap=cmap)
    for x, y in points:
        plt.plot(x, y, 'r+')

    plt.title(title)
    plt.show()


def plot_lai(lai, green, points, title='LAI'):
    """
    Debug what we are doing.
    """
    lai = np.copy(lai)

    measurements = np.where(lai < 100, True, False)

    normalize_fill_values(lai)
    # divide all values below 100 with 100
    np.place(lai, measurements, lai[measurements] / 10)

    g = green
    green_mask = np.logical_and(g > 0, g < 5)

    laif = np.copy(lai)
    laic = np.copy(lai)

    goodlocations = np.logical_and(green_mask, measurements)

    plt.imshow(goodlocations)
    plt.title('boolean ok green & lai locations')
    # plt.colorbar()
    plt.show()

    # plot normalized lai
    _plot_lai(lai, points, title)
    # _plot_lai(
    np.place(laic, ~goodlocations & measurements, 0)

    _plot_lai(laic, points, 'usefull')

    # set all not usefull lai locations  to 0
    np.place(laif, ~goodlocations, 0)

    # plot lai to calculate with
    plt.imshow(laif)
    plt.show()

    plt.imshow(green)
    plt.title('greens')
    plt.show()


def plot_control(control, points):
    """
    """
    pass


def plot_stuff(lai, green, grid):
    """
    debug some plotting stuff
    """
    plot_lai(lai, green)
    return
    # lai[green] = 1
    green[green > 0]
    lai[lai > 255] = 10
    # lai[] = 3
    plt.tight_layout()
    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=True)
    # plt.imshow(lai, norm=norm, cmap=CMAP)
    plt.imshow(lai, vmin=0, vmax=86)
    plt.colorbar()
    plt.show()


def run_map(value_generator, background_data, green, grids, modulo=5):
    """Display the simulation using matplotlib
    """

    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    ax.set_aspect('equal')
    rw = value_generator()
    plt.show(False)
    # plt.draw()
    # plt.figure()

    background_data[green] = 1
    background_data[background_data > 5] = 3
    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=True)

    tic = time.time()
    niter = 0

    ax.clear()
    ax.imshow(background_data, norm=norm, cmap=CMAP)

    def map_update(px, py):

        # for grd, c in zip(grids, ['ro', 'b+']):
        #     for x, y in grd:
        #         plt.plot(x, y, c)

        plt.plot(px, py, 'r+')
        # fig.canvas.draw()
        log.info(niter)
        plt.pause(0.001)  # It ain't needed!!!

    for x, y, v in rw:
        niter += 1
        # background_data[x][y] = v
        if niter % modulo != 0:
            continue

        map_update(x, y)

    plt.show()
    print("Average FPS: %.2f" % (niter / (time.time() - tic)))


def green_simulator(mask):
    def generate_dots():
        xarr, yarr = numpy.where(mask)
        for x, y in zip(xarr, yarr):
            yield x, y, 2

    return generate_dots


# test this plotting code.
if __name__ == '__main__':
    MATRIX_SIZE = 100
    # random map 11 ground types
    # background = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE))
    background = numpy.random.randint(0, 4, size=(MATRIX_SIZE, MATRIX_SIZE))
    # make 1's background data
    gray = ma.masked_inside(background, 1, 1)
    background[gray.mask] = 4
    # make 3's also 0 for water.
    water = ma.masked_inside(background, 3, 3)
    background[water.mask] = 0
    green = ma.masked_inside(background, 2, 2)
    generator = green_simulator(green)
    # Test the progress plotting of a map
    run_map(generator, background, green.mask, [], modulo=50)
