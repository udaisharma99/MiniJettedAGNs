# merge FR0 and CoreG catalogues into a basic catalogue of mini-jetted AGNs
# find counterparts in X-ray catalogues
# - basic imports
import logging
from astroquery.vizier import Vizier
# - local imports
from data_structures2 import Source

# set up logging
log = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# initial catalogues:
# - CoreG Catalogs
ho_1997 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/112/315")
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
# - FR0 Catalog
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")

source1 = Source("NGC 1275")
print(source1.ned_flux_table)

source2 = Source("NGC 4261")
print(source2.ned_flux_table)

import IPython; IPython.embed()
