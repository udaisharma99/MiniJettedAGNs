# merge FR0 and CoreG catalogues into a basic catalogue of mini-jetted AGNs
# find counterparts in X-ray catalogues
# - basic imports
import logging
from astroquery.vizier import Vizier
# - local imports
from data_structures import Source

# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(level)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# initial catalogues:
# - CoreG Catalogs
ho_1997 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/112/315")
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
# - FR0 Catalog
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")

import IPython; IPython.embed()
