# we import all the catalogues from here
import logging
from pathlib import Path
from astroquery.vizier import Vizier


# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)
# path of this module
main_dir_path = Path(__file__).parent.parent


# clear vizier cache, in case of repeatedly getting failed queries
# or bad/out-of-date results
Vizier.clear_cache()


# principal catalogues, these we want to load all
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")

# counterpart catalogues, in these we just want to search
# - lines
ho_1997 = Vizier(
    catalog="J/ApJS/112/315",
    columns=["Name", "AType", "logL(Ha)", "[OIII]"],
    row_limit=-1,
)

# - radio surveys
# -- NVSS
nvss = Vizier(catalog="VIII/65/nvss", columns=["NVSS", "S1.4", "e_S1.4"], row_limit=-1)
# -- FIRST
first = Vizier(
    catalog="VIII/92/first14", columns=["FIRST", "Fint", "Rms"], row_limit=-1
)

# - MORX
cols_morx = [
    "Name",
    "RAJ2000",
    "DEJ2000",
    "Type",
    "z",
    "RXpct",
    "Xraypct",
    "XMM-ID",
    "CX-ID",
    "Swift-ID",
    "radiopct",
    "NVSS-ID",
    "FIRST-ID",
    "LoTSS-ID",
    "VLASS-ID",
    "Lobe1",
    "Lobe2",
    "Lobedist",
]
morx = Vizier(catalog="V/158/morxv2", columns=cols_morx, row_limit=-1)

# - X-ray catalogues
# -- Chandra catalogues
csc1 = Vizier(catalog="IX/45/csc11", columns=["**"], row_limit=-1)
csc2 = Vizier(catalog="	IX/57/csc2master", columns=["**"], row_limit=-1)
csc_acis = Vizier(catalog="J/ApJS/224/40", columns=["**"], row_limit=-1)

# -- XMM catalogues

# 4XMM DR14 is not available from Vizier, let use DR13 for the moment
_2xmm = Vizier(catalog="IX/69/xmm4d13s", columns=["**"], row_limit=-1)
_2xmmi = Vizier(catalog="IX/69/xmm4d13s", columns=["**"], row_limit=-1)
_4xmm = Vizier(catalog="IX/69/xmm4d13s", columns=["**"], row_limit=-1)


torresi_sources = [
    "SDSS J004150.47-091811.2",
    "SDSS J010101.12-002444.4",
    "SDSS J011515.78+001248.4",
    "SDSS J015127.10-083019.3",
    "SDSS J080624.94+172503.7",
    "SDSS J092405.30+141021.5",
    "SDSS J093346.08+100909.0",
    "SDSS J094319.15+361452.1",
    "SDSS J104028.37+091057.1",
    "SDSS J114232.84+262919.9",
    "SDSS J115954.66+302726.9",
    "SDSS J122206.54+134455.9",
    "SDSS J125431.43+262040.6",
    "Tol 1326-379",
    "SDSS J135908.74+280121.3",
    "SDSS J153901.66+353046.0",
    "SDSS J160426.51+174431.1",
    "SDSS J171522.97+572440.2",
    "SDSS J235744.10-001029.9",
]
