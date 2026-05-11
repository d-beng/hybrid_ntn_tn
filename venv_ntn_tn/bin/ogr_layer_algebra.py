#!/home/db3n/Documents/Ph.D./Courses/Directed_Studies/hybrid_ntn_tn/venv_ntn_tn/bin/python3.12

import sys

from osgeo.gdal import deprecation_warn

# import osgeo_utils.ogr_layer_algebra as a convenience to use as a script
from osgeo_utils.ogr_layer_algebra import *  # noqa
from osgeo_utils.ogr_layer_algebra import main

deprecation_warn("ogr_layer_algebra")
sys.exit(main(sys.argv))
