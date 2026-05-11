#!/home/db3n/Documents/Ph.D./Courses/Directed_Studies/hybrid_ntn_tn/venv_ntn_tn/bin/python3.12

import sys

from osgeo.gdal import deprecation_warn

# import osgeo_utils.gdal_pansharpen as a convenience to use as a script
from osgeo_utils.gdal_pansharpen import *  # noqa
from osgeo_utils.gdal_pansharpen import main

deprecation_warn("gdal_pansharpen")
sys.exit(main(sys.argv))
