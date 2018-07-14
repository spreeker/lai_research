#Map of the dataset MODIS15A.
"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu.
"""
import gdal
import glob
import numpy
from matplotlib import pyplot
import matplotlib as mpl
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)


def process_modis(filename, call_back):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)

    log.info("Driver: {}/{}".format(
        dataset.GetDriver().ShortName,
        dataset.GetDriver().LongName))
    log.info("Size is {} x {} x {}".format(
        dataset.RasterXSize,
        dataset.RasterYSize,
        dataset.RasterCount))

    log.info("Projection is {}".format(dataset.GetProjection()))

    geotransform = dataset.GetGeoTransform()

    if geotransform:
        log.info("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        log.info("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i+1, ds)

    call_back(dataset)


def process_data(dataset):

    band = dataset.GetRasterBand(1)
    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    log.debug('Data type %s', bandtype)

    data = band.ReadAsArray()
    log.debug('Bytes: %s Size %.5d kb', data.nbytes, float(data.nbytes) / 1024)

    passer = numpy.logical_and(data > 0, data <= 1000)

    log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
        data[passer].min(), data[passer].max(),
        data[passer].mean(), data[passer].std())
    )

    #new_m = numpy.divide(lai, 10)
    #pyplot.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
    #pyplot.set_yticklabels(['< 1', 'green', '> 5'])  # vertically oriented colorbar
    #lai[lai > 7] = 7
    #data[data < 1] = 10
    #data[data > 5] = 10

    cmap = mpl.colors.ListedColormap([
        'gray',
        'lightgreen', 'green', 'green', 'darkgreen',
        'darkgray'
    ])
    bounds = numpy.array([0, 1, 2, 6, 11])

    norm = mpl.colors.Normalize(vmin=0, vmax=6, clip=True)

    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    img = pyplot.imshow(data, norm=norm, cmap=cmap)

    # cb2 = mpl.colorbar.ColorbarBase(
    #     img.axes, cmap=cmap,
    #     norm=norm,
    #     boundaries=[0] + bounds + [18],
    #     extend='both',
    #     ticks=bounds,
    #     spacing='proportional',
    #     orientation='horizontal'
    # )

    pyplot.colorbar()
    pyplot.show()

    return


if __name__ == '__main__':
    #hdf LAI directory data

    #hdf_files = glob.glob('D:/LAI_thesis/*.hdf')
    hdf_files = glob.glob('D:/LAI_thesis/Landuse_german/*.hdf')
    # Landuse.

    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    for hdf_name in hdf_files:
        process_modis(
            #hdf_name,
            #f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m',
            #f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            #f'HDF4_EOS: EOS_GRID:"{hdf_name}": MOD12Q1:Land_Cover_Type_5',
            'HDF4_EOS:EOS_GRID:"D:/LAI_thesis/Landuse_german\\MCD12Q1.A2011001.h18v03.051.2014288191624.hdf":MOD12Q1:Land_Cover_Type_5',
            process_data)