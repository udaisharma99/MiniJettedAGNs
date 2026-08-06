from data_structures2 import Source
from data_structures import coreG_catalogue, fr0_catalogue
import logging
from astropy.table import vstack
from astroquery.vizier import Vizier
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

#Importing Source Catalogues
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")

for name, _type in zip(
    nagar_2005[0]["Name"],
    nagar_2005[0]["AType"],
):
    if 'L' in _type:
        source = Source(name)
        source.write_catalogue_row(coreG_catalogue)

for row in fr0cat[0]:
    name = row['SimbadName']
    source = Source(name)
    source.write_catalogue_row(fr0_catalogue)

fr0_catalogue = fr0_catalogue.table 
path1 = Path("./")
path1.mkdir(exist_ok=True, parents=True)
fr0_catalogue.write(
    path1 / "fr0_catalogue.fits", overwrite=True
)
coreG_catalogue = coreG_catalogue.table
path2 = Path("./")
path2.mkdir(exist_ok=True, parents=True)
coreG_catalogue.write(
    path2 / "coreG_catalogue.fits", overwrite=True
)
fr0_coreG_catalogue = vstack([fr0_catalogue,coreG_catalogue])
path = Path("./")
path.mkdir(exist_ok=True, parents=True)
fr0_coreG_catalogue.write(
    path / "fr0_coreG_catalogue.fits", overwrite=True
)

import IPython; IPython.embed()
