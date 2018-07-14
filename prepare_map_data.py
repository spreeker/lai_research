"""
We prepare data for LAI prediction function calculations.

1. Extract green locations from map.
2. Collect all CRU data for green locations and store in hdf file.
    - 10 years
    - use geotransform from modis file to make selection in cru data!
    - create mapping matrix. px, py -> lon, lat
3. Collect all LAI data for green locations (done)
    - smoothed.
    - normalized.
    - 10 years.
    - resolution of LAI map is higher.
4. Calculate best lai predictor function for each green location.
    - draw map with colors from this.
    - maybe just a subset of these locations.
5. Improve predictions functions and run step 4 again.
"""

import argparse
import logging
import extract_green
import extract_CRU
import read_modis

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

CRU_LOC = {}


def collect_cru():
    """
    Collect all relavant CRU information for
    current Landuse map
    """
    # first extract green map.
    ds, values, green, xarr, yarr = extract_green.extract()
    # find out wich bouding (lai/landuse) bbox we are currently working on.
    bbox = read_modis.make_lonlat_bbox(ds)
    log.debug('BBOX: %s', bbox)
    # store CRU data in hdf5
    extract_CRU.collect_cru_in(bbox)
    # convert lat,lon to x, y
    # grid = find_xy_cru_grid(ds, lon_lat_grid)


def main(args):
    if args.collect_cru:
        collect_cru()


if __name__ == '__main__':
    # print_hdf_info()
    desc = "Collect CEU for current region"
    inputparser = argparse.ArgumentParser(desc)
    inputparser.add_argument(
        '--collect_cru',
        action='store_true',
        default=False,
        help="Extract CRU data for currenct configured map")

    args = inputparser.parse_args()
    main(args)
