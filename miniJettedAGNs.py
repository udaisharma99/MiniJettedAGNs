import time
import logging
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import Distance
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from utils import (
    get_source_identifier,
    insert_space_source_ids,
    convert_F_nu_to_luminosity,
)


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

def find_xmatch_id_in_catalog(
    xmatch_id, xmatch_id_column, catalog, catalog_id_column, id_strip=None
):
    """Given a catalogue with cross-matched sources, return the sources
    corresponding to a given counterpart."""
    if id_strip is None:
        this_source = catalog[xmatch_id_column] == xmatch_id
    else:
        if catalog[xmatch_id_column].dtype == np.int64:
            this_source = catalog[xmatch_id_column] == np.int64(
                xmatch_id.strip(id_strip)
            )
        else:
            this_source = catalog[xmatch_id_column] == xmatch_id.strip(id_strip)

    xmatches = catalog[this_source]

    if len(xmatches) == 0:
        log.info(f"{xmatch_id} not matched in the catalogue")
    else:
        log.info(f"{xmatch_id} matched with {xmatches[catalog_id_column].data}")
        return xmatches

def convert_flux_to_luminosity(flux, flux_unit, distance):
    
    flux = flux * u.Unit(flux_unit)
    D_L = distance.to(u.cm) # Convert Mpc to cm for consistent units
    flux_in_cgs = flux.to(u.erg / (u.s * u.cm**2))
    luminosity = 4 * np.pi * D_L**2 * flux_in_cgs
    
    return luminosity.to(u.erg / u.s)

#CoreG Catalogs
ho_1997 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/112/315")
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
#FR0 Catalog
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")
#MORX Catalog
morx = Vizier(columns=["**"], row_limit=-1).get_catalogs("V/158/morxv2")
torresi_sources = [
    "SDSS J004150.47−091811.2",
    "SDSS J010101.12−002444.4",
    "SDSS J011515.78+001248.4",
    "SDSS J015127.10−083019.3",
    "SDSS J080624.94+172503.7",
    "SDSS J092405.30+141021.5",
    "SDSS J093346.08+100909.0",
    "SDSS J094319.15+361452.1",
    "SDSS J104028.37+091057.1",
    "SDSS J114232.84+262919.9",
    "SDSS J115954.66+302726.9",
    "SDSS J122206.54+134455.9",
    "SDSS J125431.43+262040.6",
    "Tol 1326−379",
    "SDSS J135908.74+280121.3",
    "SDSS J153901.66+353046.0",
    "SDSS J160426.51+174431.1",
    "SDSS J171522.97+572440.2",
    "SDSS J235744.10−001029.9",
]

coreG_catalogue = Table(
    names=(
        "SOURCE_NAME",
        "SOURCE_TYPE",
        "SIMBAD SDSS-ID",
        "SIMBAD FERMI-ID",
        "SIMBAD NVSS-ID",
        "NVSS-XMATCH-ID",
        "NVSS-MORX-ID",
        "SIMBAD FIRST-ID",
        "FIRST-XMATCH-ID",
        "FIRST-MORX-ID",
        "XMM-MORX-ID",
        "CXO-MORX-ID",
        "SWIFT-MORX-ID",
        "LoTSS-MORX-ID",
        "VLASS-MORX",
        "LOBE EXTENSION",
        "DISTANCE",
        "Log10(L_OIII)",
        "TORESSI DETECTION",
        "NVSS-FLUX-XMATCH",
        "NVSS-FLUX-ERROR-XMATCH",
        "FIRST-FLUX-XMATCH",
        "FIRST-FLUX-ERROR-XMATCH",
    ),
    dtype=[
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        np.int16,
        np.float32,
        np.float32,
        bool,
        np.float64,
        np.float64,
        np.float64,
        np.float64
    ],
    units=[
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "mas",
        "Mpc",
        "erg s-1",
        "",
        "erg s-1",
        "erg s-1",
        "erg s-1",
        "erg s-1",
        
    ],
)

fr0_catalogue = Table(
    names=(
        "SOURCE_NAME",
        "SOURCE_TYPE",
        "SIMBAD SDSS-ID",
        "SIMBAD FERMI-ID",
        "SIMBAD NVSS-ID",
        "NVSS-XMATCH-ID",
        "NVSS-MORX-ID",
        "SIMBAD FIRST-ID",
        "FIRST-XMATCH-ID",
        "FIRST-MORX-ID",
        "XMM-MORX-ID",
        "CXO-MORX-ID",
        "SWIFT-MORX-ID",
        "LoTSS-MORX-ID",
        "VLASS-MORX",
        "LOBE EXTENSION",
        "DISTANCE",
        "Log10(L_OIII)",
        "TORESSI DETECTION",
        "NVSS-FLUX-XMATCH",
        "NVSS-FLUX-ERROR-XMATCH",
        "FIRST-FLUX-XMATCH",
        "FIRST-FLUX-ERROR-XMATCH",
    ),
    dtype=[
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        np.int16,
        np.float32,
        np.float32,
        bool,
        np.float64,
        np.float64,
        np.float64,
        np.float64
    ],
    units=[
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "mas",
        "Mpc",
        "erg s-1",
        "",
        "erg s-1",
        "erg s-1",
        "erg s-1",
        "erg s-1",
        
    ],
)

first_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:VIII/92/first14",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=10 * u.arcsec,
)

nvss_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:VIII/65/nvss",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=10 * u.arcsec,
)

morx_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:V/158/morxv2",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=1 * u.arcsec,
)

for name, _type, distance, F_15GHz in zip(
    nagar_2005[0]["Name"],
    nagar_2005[0]["AType"],
    nagar_2005[0]["Dist"],
    nagar_2005[0]["St15GHz"],
):
    if "L" in _type:
        log.info(f"considering {name}")
        sdss_id = get_source_identifier(name,"SDSS")
        fermi_id = get_source_identifier(name, "4FGL")
        nvss_id = get_source_identifier(name, "NVSS")
        first_id = get_source_identifier(name, "FIRST")
        distancee =   distance * u.Mpc
        
        match_ho = ho_1997[1]["Name"] == insert_space_source_ids(name)
        if np.any(match_ho):  # Check if there is at least one True
            _log_L_alpha = ho_1997[1]["logL(Ha)"][match_ho][0]
            _OIII = ho_1997[1]["[OIII]"][match_ho][0]
            L_OIII = np.power(10, _log_L_alpha) * _OIII * u.Unit("erg s-1")
        else:
            L_OIII = 0 * u.Unit("erg s-1")
        
        # NVSS cross match with Nagar 2005 measurement
        this_source_nvss_xmatch = nvss_xmatches["Name"] == name
        if this_source_nvss_xmatch.any():
            nvss_xmatch_name = (
                "NVSS J" + nvss_xmatches["NVSS"][this_source_nvss_xmatch][0]
            )
            nvss_xmatch_flux = nvss_xmatches["S1.4"][this_source_nvss_xmatch][0]
            nvss_xmatch_flux_err = nvss_xmatches["e_S1.4"][this_source_nvss_xmatch][0]
        else:
            nvss_xmatch_name = ""
            nvss_xmatch_flux = 0
            nvss_xmatch_flux_err = 0

        # NVSS Cross matched with Nagar and then with MORX 
        nvss_morx_match = nvss_xmatch_name == morx_nagar_xmatches["NVSS-ID"]
        if nvss_morx_match.any():
            morx_nvss_name = morx_nagar_xmatches["NVSS-ID"][nvss_morx_match][0]
        else:
            morx_nvss_name = ""

        # FIRST measurement
        this_source_first_xmatch = first_xmatches["Name"] == name
        if this_source_first_xmatch.any():
            first_xmatch_name = (
                "FIRST " + first_xmatches["FIRST"][this_source_first_xmatch][0]
            )
            first_xmatch_name_nospace = (
                "FIRST" + first_xmatches["FIRST"][this_source_first_xmatch][0]
            )
            first_xmatch_flux = first_xmatches["Fint"][this_source_first_xmatch][0]
            first_xmatch_flux_err = first_xmatches["Rms"][this_source_first_xmatch][0]
            
        else:
            first_xmatch_name = ""
            first_xmatch_name_nospace = ""
            first_xmatch_flux = 0
            first_xmatch_flux_err = 0
            
        first_morx_match = first_xmatch_name_nospace == morx_nagar_xmatches["FIRST-ID"]
        if first_morx_match.any():
            morx_first_name = morx_nagar_xmatches["FIRST-ID"][first_morx_match][0]
        else:
            morx_first_name = ""

        morx_matches = morx_nagar_xmatches["Name"] == name
        if morx_matches.any():
            morx_xmm = morx_nagar_xmatches["XMM-ID"][morx_matches]
            morx_cxo = morx_nagar_xmatches["CX-ID"][morx_matches]
            morx_swift = morx_nagar_xmatches["Swift-ID"][morx_matches]
            morx_first = morx_nagar_xmatches["FIRST-ID"][morx_matches]
            morx_nvss = morx_nagar_xmatches["NVSS-ID"][morx_matches]
            morx_lotss = morx_nagar_xmatches["LoTSS-ID"][morx_matches]
            morx_vlass = morx_nagar_xmatches["VLASS-ID"][morx_matches]
            morx_lobedist = morx_nagar_xmatches["Lobedist"][morx_matches]
        else:
            morx_xmm = ""
            morx_cxo = ""
            morx_swift = ""
            morx_first = ""
            morx_nvss = ""
            morx_lotss = ""
            morx_vlass = ""
            morx_lobedist = 0
        
        # check if the source is in the list of sources detected by Torresi et al. 2018
        torresi_detection = sdss_id in torresi_sources

        coreG_catalogue.add_row(
            [
                name,
                _type,
                sdss_id,
                fermi_id,
                nvss_id,
                nvss_xmatch_name,
                morx_nvss_name,
                first_id,
                first_xmatch_name,
                morx_first_name,
                morx_xmm,
                morx_cxo,
                morx_swift,
                morx_lotss,
                morx_vlass,
                morx_lobedist,
                distancee,
                np.log10(L_OIII.to_value("erg s-1")),
                torresi_detection,
                convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux, u.mJy, distancee),
                convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux_err, u.mJy, distancee),
                convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux, u.mJy, distancee),
                convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distancee),
            ])

first_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:VIII/92/first14",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=10 * u.arcsec,
)

nvss_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:VIII/65/nvss",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=5 * u.arcsec,
)

morx_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:V/158/morxv2",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=1 * u.arcsec,
)

# let us do the same for FR0 galaxies
# we do not need to XMatch with NVSS as there is already a NVSS flux measurement for the FR0 sources
# we will not search for FIRST counterparts as there is already the NVSS flux measurement at 1.4 GHz


for row in fr0cat[0]:
    sdss_id = row["SimbadName"]
    ngc_id = get_source_identifier(sdss_id, "NGC")
    first_id = get_source_identifier(sdss_id, "FIRST")
    ic_id = get_source_identifier(sdss_id, "IC")
    nvss_id = get_source_identifier(sdss_id, "NVSS")
    name = ngc_id if ngc_id else ic_id
    fermi_id = get_source_identifier(sdss_id, "4FGL")
    distance = Distance(z=row["z"]).to("Mpc")
    L_OIII_FR0 = row["logL[OIII]"]
    L_NVSS = np.power(10, row["logLr"])
   
    # FIRST measurement
    this_source_first_xmatch = first_fr0_xmatches["SimbadName"] == sdss_id
    if this_source_first_xmatch.any():
        first_xmatch_name = (
            "FIRST " + first_fr0_xmatches["FIRST"][this_source_first_xmatch][0]
        )
        first_xmatch_name_nospace = (
                "FIRST" + first_fr0_xmatches["FIRST"][this_source_first_xmatch][0]
            )
        first_xmatch_flux = first_fr0_xmatches["Fint"][this_source_first_xmatch][0]
        first_xmatch_flux_err = first_fr0_xmatches["Rms"][this_source_first_xmatch][0]
    else:
        first_xmatch_name = ""
        first_xmatch_flux = 0
        first_xmatch_name_nospace = ""
        first_xmatch_flux_err = 0
            
    first_morx_match = first_xmatch_name_nospace == morx_fr0_xmatches["FIRST-ID"]
    if first_morx_match.any():
        morx_first_name = morx_fr0_xmatches["FIRST-ID"][first_morx_match][0]
    else:
        morx_first_name = ""   

    # NVSS cross match with Nagar 2005 measurement
    this_source_nvss_xmatch = nvss_fr0_xmatches["SimbadName"] == sdss_id
    if this_source_nvss_xmatch.any():
        nvss_xmatch_name = (
            "NVSS J" + nvss_fr0_xmatches["NVSS"][this_source_nvss_xmatch][0]
        )
        nvss_xmatch_flux = nvss_fr0_xmatches["S1.4"][this_source_nvss_xmatch][0]
        nvss_xmatch_flux_err = nvss_fr0_xmatches["e_S1.4"][this_source_nvss_xmatch][0]
    else:
        nvss_xmatch_name = ""
        nvss_xmatch_flux = 0
        nvss_xmatch_flux_err = 0

        # NVSS Cross matched with Nagar and then with MORX 
    nvss_morx_match = nvss_xmatch_name == morx_fr0_xmatches["NVSS-ID"]
    if nvss_morx_match.any():
        morx_nvss_name = morx_fr0_xmatches["NVSS-ID"][nvss_morx_match][0]
    else:
        morx_nvss_name = ""
    
    morx_matches = morx_fr0_xmatches["SimbadName"] == sdss_id
    if morx_matches.any():
        morx_xmm = morx_fr0_xmatches["XMM-ID"][morx_matches]
        morx_cxo = morx_fr0_xmatches["CX-ID"][morx_matches]
        morx_swift = morx_fr0_xmatches["Swift-ID"][morx_matches]
        morx_first = morx_fr0_xmatches["FIRST-ID"][morx_matches]
        morx_nvss = morx_fr0_xmatches["NVSS-ID"][morx_matches]
        morx_lotss = morx_fr0_xmatches["LoTSS-ID"][morx_matches]
        morx_vlass = morx_fr0_xmatches["VLASS-ID"][morx_matches]
        morx_lobedist = morx_fr0_xmatches["Lobedist"][morx_matches]
    else:
        morx_xmm = ""
        morx_cxo = ""
        morx_swift = ""
        morx_first = ""
        morx_nvss = ""
        morx_lotss = ""
        morx_vlass = ""
        morx_lobedist = 0
        
        # check if the source is in the list of sources detected by Torresi et al. 2018
    torresi_detection = sdss_id in torresi_sources

    fr0_catalogue.add_row(
         [
            name,
            "FR0",
            sdss_id,
            fermi_id,
            nvss_id,
            nvss_xmatch_name,
            morx_nvss_name,
            first_id,
            first_xmatch_name,
            morx_first_name,
            morx_xmm,
            morx_cxo,
            morx_swift,
            morx_lotss,
            morx_vlass,
            morx_lobedist,
            distance.to_value("Mpc"),
            L_OIII_FR0,
            torresi_detection,
            convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux, u.mJy, distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux_err, u.mJy, distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux, u.mJy, distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distance),
        ])

from astropy.table import vstack

fr0_coreG_catalogue = vstack([fr0_catalogue,coreG_catalogue])
