"""Create a netcdf file. for nice raster plotting.
"""

import argparse
import datetime
import numpy
import read_modis
import logging
from netCDF4 import Dataset as netcdf


