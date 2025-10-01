# gather all X-ray information available on CoreG and FR0 sources
import numpy as np
from astroquery.vizier import Vizier

# initial catalogues:
# - CoreG Catalogs
ho_1997 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/112/315")
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
# - FR0 Catalog
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")


import IPython; IPython.embed()