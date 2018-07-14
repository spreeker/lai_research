"""
Modis plot functions.

Reproject numpy array to different projection
"""

from osgeo import gdal
from osgeo import osr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from create_lai_cube import get_lai_gdal_modis_files

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def load_numpy_to_gdal_mem(
        array, geotransform, wtk_from, x_size=None, y_size=None):
    """Create in memory gdal object we can use for reprojection
    """
    assert x_size
    assert y_size

    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')

    dest = mem_drv.Create('', x_size, y_size, 1, gdal.GDT_Float32)

    # Set the geotransform
    dest.SetGeoTransform(geotransform)
    dest.SetProjection(wtk_from)
    outband = dest.GetRasterBand(1)
    outband.WriteArray(array)
    outband.FlushCache()
    return dest


def reproject_dataset(
        dataset,
        pixel_x_spacing=(0.4639)/60.,
        pixel_y_spacing=(0.4639)/100.,
        wkt_from="+proj=sinu +R=6371007.181 +nadgrids=@null +wktext",
        # epsg_to=3395
        epsg_to=4326,
        geotransform=None,
        x_size=None,
        y_size=None,
    ):
    """
    Derived from:

    http://jgomezdans.github.io/gdal_notes/reprojection.html

    A sample function to reproject and resample a GDAL dataset from within
    Python. The idea here is to reproject from one system to another, as well
    as to change the pixel size. The procedure is slightly long-winded, but
    goes like this:

    1. Set up the two Spatial Reference systems.
    2. Open the original dataset, and get the geotransform
    3. Calculate bounds of new geotransform by projecting the UL corners
    4. Calculate the number of pixels with the new projection & spacing
    5. Create an in-memory raster dataset
    6. Perform the projection
    """
    # Define the UK OSNG, see <http://spatialreference.org/ref/epsg/27700/>
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(epsg_to)
    modis = osr.SpatialReference()
    modis.ImportFromProj4(wkt_from)
    tx = osr.CoordinateTransformation(modis, wgs84)
    # Up to here, all  the projection have been defined, as well as a
    # transformation from the from to the  to :)
    # We now open the dataset
    if not geotransform:
        g = gdal.Open(dataset)
        # Get the Geotransform vector
        geo_t = g.GetGeoTransform()
        x_size = g.RasterXSize  # Raster xsize
        y_size = g.RasterYSize  # Raster ysize
    else:
        g = load_numpy_to_gdal_mem(
            dataset, geotransform, wkt_from, x_size, y_size)
        geo_t = geotransform

    # Work out the boundaries of the new dataset in the target projection
    (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
    (lrx, lry, lrz) = tx.TransformPoint(
        geo_t[0] + geo_t[1] * x_size, geo_t[3] + geo_t[5] * y_size)

    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')

    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. Also, setting it to store one band
    # and to use Float32 data type.
    dest = mem_drv.Create(
        '', int((lrx - ulx)/pixel_x_spacing),
        int((uly - lry)/pixel_y_spacing), 1, gdal.GDT_Float32)

    # Calculate the new geotransform
    new_geo = (ulx, pixel_x_spacing, geo_t[2],
               uly, geo_t[4], -pixel_y_spacing)

    # Set the geotransform
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(wgs84.ExportToWkt())

    # Perform the projection/resampling
    res = gdal.ReprojectImage(
        g, dest, modis.ExportToWkt(), wgs84.ExportToWkt(),
        gdal.GRA_Mode)

    log.debug(res)

    return dest


def plot_gdal_file(input_dataset, vmin=0, vmax=100):
    # plt.figure ( figsize=(11.3*0.8, 8.7*0.8), dpi=600 ) # This is A4. Sort of

    geo = input_dataset.GetGeoTransform()  # Need to get the geotransform
    data = input_dataset.ReadAsArray()
    # We need to flip the raster upside down
    data = np.flipud(data)
    # Define a cylindrical projection
    projection_opts = {'projection': 'cyl', 'resolution': 'h'}

    # These are the extents in the native raster coordinates
    extent = [
        geo[0], geo[0] + input_dataset.RasterXSize*geo[1],
        geo[3], geo[3] + input_dataset.RasterYSize*geo[5]]

    print(geo)
    print(extent)
    print(input_dataset.RasterXSize)
    print(input_dataset.RasterYSize)

    map = Basemap(
        llcrnrlon=extent[0], llcrnrlat=extent[3],
        urcrnrlon=extent[1], urcrnrlat=extent[2],  ** projection_opts)

    cmap = plt.cm.gist_rainbow
    cmap.set_under('0.8')
    cmap.set_bad('0.8')
    cmap.set_over('0.8')

    map.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')

    map.drawcoastlines(linewidth=0.5, color='k')
    map.drawcountries(linewidth=0.5, color='k')

    map.drawmeridians(
        np.arange(0, 360, 1), color='k', labels=[False, True, True, False])
    map.drawparallels(
        np.arange(-90, 90, 1), color='k', labels=[False, True, True, False]
    )
    map.drawmapboundary()

    cb = plt.colorbar(orientation='horizontal', fraction=0.10, shrink=0.8)

    cb.set_label(r'$10\cdot LAI$')

    plt.title('LAI')
    plt.show()


if __name__ == "__main__":
    # Two scenarios
    # 1.- We want to reproject using ``reproject_dataset`` above. Then the input_filename is
    # the full MODIS HDF granule
    # input_file='HDF4_EOS:EOS_GRID:"{hdf_file}":MOD_Grid_MOD15A2:Lai_1km'
    input_file = get_lai_gdal_modis_files()[60]
    reprojected_dataset = reproject_dataset(input_file)
    plot_gdal_file(reprojected_dataset, vmin=0, vmax=100)
    # (Note that in this case, the LAI product example is 1km, and
    # I reproject *and* resample to ~0.5km. Change pixel_spacing option for other values)

    # 2.- We already have the reprojected data in an eg VRT file
    # reprojected_datset = gdal.Open ( "output_raster.vrt" )
    # plot_gdal_file (reprojected_datset, vmin=1, vmax=60)
