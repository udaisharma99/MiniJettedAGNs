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
# - lines
ho_1997 = Vizier(
    catalog="J/ApJS/112/315",
    columns=["Name", "AType", "logL(Ha)", "[OIII]"],
    row_limit=-1
)

# - radio surveys
# -- NVSS
nvss = Vizier(
    catalog="VIII/65/nvss",
    columns=["NVSS", "S1.4", "e_S1.4"],
    row_limit=-1
)
# -- FIRST
first = Vizier(
    catalog="VIII/92/first14",
    columns=["FIRST", "Fint", "Rms"],
    row_limit=-1
)

# - X-ray catalogues
# -- MORX
cols_morx = [
    "Name",
    "RAJ2000",
    "DEJ2000",
    "Type",
    "NVSS-ID",
    "XMM-ID",
    "CX-ID",
    "Swift-ID",
    "Lobedist",
]
morx = Vizier(
    catalog="V/158/morxv2",
    columns=cols_morx,
    row_limit=-1
)

"""
## - X-ray catalogues
# NOTE: for 4XMM there is already a DR14 that required to be ingested as a file
# let use DR13 from Vizier for now
cols_xmm = [
    "iauname",
    "ra",
    "dec",
    "sc_ep_4_flux",
    "sc_ep_4_flux_err",
    "sc_ep_5_flux",
    "sc_ep_5_flux_err",
    "sc_hr3",
    "sc_hr4",
    "sc_hr3_err",
    "sc_hr4_err",
    "sc_var_flag",
]
fourxmm = Vizier(columns=cols_xmm, row_limit=-1).get_catalogs("IX/69/xmm4d13s")
log.info("loaded 4XMM-DR13 catalogue from Vizier")

cols_csc = [
    "RAICRS",
    "DEICRS",
    "2CXO",
    "FPL0.5-7",
    "b_FPL0.5-7",
    "B_FPL0.5-7",
    "GamPL",
    "b_GamPL",
    "B_GamPL",
    "HRhm",
    "b_HRhm",
    "B_HRhm",
    "fv",
]
cxotwo = Vizier(columns=cols_csc, row_limit=-1).get_catalogs("IX/70/csc21mas")
log.info("loaded CSC2 catalogue from Vizier")

cols_twosxps = [
    "RAJ2000",
    "DEJ2000",
    "IAUName",
    "FPCO0",
    "e_FPCO0",
    "E_FPCO0",
    "Gamma",
    "e_Gamma",
    "E_Gamma",
    "HR2",
    "e_HR2",
    "E_HR2",
]
twosxps_swift = Vizier(columns=cols_twosxps, row_limit=-1).get_catalogs("IX/58/2sxps")
log.info("loaded 2SXPS catalogue from Vizier")

bat157 = Table.read(
    f"{main_dir_path}/data/catalogues/BAT_157m.txt",
    format="ascii",
    delimiter="|"
)

x_ray_catalogs = [morx[0], fourxmm, cxotwo[0], twosxps_swift[0], bat157]

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

## - Gamma-ray catalogues
fermi_4fgl = Table.read(f"{main_dir_path}/data/catalogues/4fgl-dr4.fit", format="fits")

fermi_transient = Table.read(f"{main_dir_path}/data/catalogues/1FLT_final_V23.fits", format="fits")

gamma_ray_catalogs = [fermi_4fgl, fermi_transient]
"""