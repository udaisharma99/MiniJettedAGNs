#!/usr/bin/env python
# coding: utf-8

# In[48]:


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
    remove_space_source_ids,
    convert_F_nu_to_luminosity,
)


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


# In[49]:


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


# # Load the CoreG and FR0 catalogues

# In[52]:


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
#Swift-BAT Catalogs of 105 month survey, Detected AGN in 60 months, 
swift_bat_105_months = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/235/4")


# In[53]:


morx[0]


# In[64]:


catalogue = Table(
    names=(
        "SOURCE_NAME",
        "SOURCE_TYPE",
        "SDSS-ID",
        "FERMI-ID",
        "NVSS-XMATCH-ID",
        "NVSS-MORX-ID",
        "FIRST-XMATCH-ID",
        "FIRST-MORX-ID",
        "XMM-MORX-ID",
        "CXO-MORX-ID",
        "SWIFT-MORX-ID",
        "LoTSS-MORX-ID",
        "VLASS-MORX",
        "LOBE EXTENSION",
        "DISTANCE",
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
        np.float64,
        np.float64,
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
        "mas",
        "Mpc",
        "",
        "erg s-1",
        "erg s-1",
        "erg s-1",
        "erg s-1",
        
    ],
)


# In[65]:


first_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:VIII/92/first14",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=10 * u.arcsec,
)

first_xmatches


# In[66]:


nvss_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:VIII/65/nvss",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=10 * u.arcsec,
)


# In[67]:


nvss_xmatches


# In[68]:


morx_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:V/158/morxv2",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=1 * u.arcsec,
)


# In[76]:


nagar_2005[0]


# In[72]:


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
        distance *= u.Mpc
        
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
        nvss_space = insert_space_source_ids(nvss_xmatch_name)
        nvss_morx_match = nvss_space == morx[0]["NVSS-ID"]
        if nvss_morx_match.any():
            morx_nvss_name = morx[0]["NVSS-ID"][nvss_morx_match][0]
        else:
            morx_nvss_name = ""

        # FIRST measurement
        this_source_first_xmatch = first_xmatches["Name"] == name
        if this_source_first_xmatch.any():
            first_xmatch_name = (
                "FIRST " + first_xmatches["FIRST"][this_source_first_xmatch][0]
            )
            first_xmatch_flux = first_xmatches["Fint"][this_source_first_xmatch][0]
            first_xmatch_flux_err = first_xmatches["Rms"][this_source_first_xmatch][0]
        else:
            first_xmatch_name = ""
            first_xmatch_flux = 0
            first_xmatch_flux_err = 0
            
        first_morx_match = first_xmatch_name == morx[0]["FIRST-ID"]
        if first_morx_match.any():
            morx_first_name = morx[0]["FIRST-ID"][first_morx_match][0]
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


        catalogue.add_row(
            [
                name,
                _type,
                sdss_id,
                fermi_id,
                nvss_xmatch_name,
                morx_nvss_name,
                first_xmatch_name,
                morx_first_name,
                morx_xmm,
                morx_cxo,
                morx_swift,
                morx_lotss,
                morx_vlass,
                morx_lobedist,
                torresi_detection,
                distance.to_value("Mpc"),
                convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux, u.mJy, distance),
                convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux_err, u.mJy, distance),
                convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux, u.mJy, distance),
                convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distance),
            ])


# In[73]:


catalogue


# In[78]:


np.array(catalogue['LOBE EXTENSION'])


# In[31]:


# let us do the same for FR0 galaxies
# we do not need to XMatch with NVSS as there is already a NVSS flux measurement for the FR0 sources
# we will not search for FIRST counterparts as there is already the NVSS flux measurement at 1.4 GHz

first_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:VIII/92/first14",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=10 * u.arcsec,
)

for row in fr0cat[0]:
    sdss_id = row["SimbadName"]
    log.info(f"considering {sdss_id}")
    ngc_id = get_source_identifier(sdss_id, "NGC")
    ic_id = get_source_identifier(sdss_id, "IC")
    _2mass_id = get_source_identifier(sdss_id, "2MASX")
    nvss_id = get_source_identifier(sdss_id, "NVSS")
    first_id = get_source_identifier(sdss_id, "FIRST")
    gaia_id = get_source_identifier(sdss_id, "Gaia DR3")
    fermi_id = get_source_identifier(name, "4FGL")

    distance = Distance(z=row["z"]).to("Mpc")

    L_OIII = np.power(10, row["logL[OIII]"])
    L_NVSS = np.power(10, row["logLr"])

    # FIRST measurement
    this_source_first_xmatch = first_fr0_xmatches["SimbadName"] == sdss_id
    if this_source_first_xmatch.any():
        first_xmatch_name = (
            "FIRST " + first_fr0_xmatches["FIRST"][this_source_first_xmatch][0]
        )
        first_xmatch_flux = first_fr0_xmatches["Fint"][this_source_first_xmatch][0]
        first_xmatch_flux_err = first_fr0_xmatches["Rms"][this_source_first_xmatch][0]
    else:
        first_xmatch_name = ""
        first_xmatch_flux = 0
        first_xmatch_flux_err = 0

     # XMM cross match with FR0cat measurement
        this_source_xmm_xmatch = xmm_xmatches_fr0["Name"] == name
        if this_source_xmm_xmatch.any():
            xmm_xmatch_name = (
                "4XMM " + xmm_xmatches["4XMM"][this_source_xmm_xmatch][0]
            )
            xmm_xmatch_flux = xmm_xmatches["Flux8"][this_source_xmm_xmatch][0]
            xmm_xmatch_flux_err = xmm_xmatches["e_Flux8"][this_source_xmm_xmatch][0]
        else:
            xmm_xmatch_name = ""
            xmm_xmatch_flux = 0
            xmm_xmatch_flux_err = 0

    # eventual CSC2 xmatches identified through the 2MASS or GAIA ID
    if _2mass_id:
        _2mass_xmatches = find_xmatch_id_in_catalog(
            _2mass_id,
            "2MASS21P_designation",
            csc2_xmatched,
            "CSC21P_name",
            "2MASX J",
        )

    if gaia_id:
        gaia_xmatches = find_xmatch_id_in_catalog(
            gaia_id, "GAIA21P_source_id", csc2_xmatched, "CSC21P_name", "Gaia DR3 "
        )

    # "common" (NGC or IC) name to be saved in the catalogue
    name = ngc_id if ngc_id else ic_id

    # search for an eventual swift counterpart
    # try both with the "common" name and with the 2MAS counterpart
    swift_bat_name_xmatch = find_xmatch_id_in_catalog(
        name, "CName", swift_bat_105_months[0], "Swift"
    )
    if swift_bat_name_xmatch is None:
        swift_bat_name_xmatch = find_xmatch_id_in_catalog(
            _2mass_id, "CName", swift_bat_105_months[0], "Swift"
        )

    # check if the source is in the list of sources detected by Torresi et al. 2018
    torresi_detection = sdss_id in torresi_sources

    if _2mass_id and _2mass_xmatches is not None:
        for _2mass_xmatch in _2mass_xmatches:
            coreg_fr0_expanded_catalogue.add_row(
                [
                    name,
                    "FR0",
                    sdss_id,
                    gaia_id,
                    _2mass_id,
                    nvss_id,
                    first_id,
                    "",
                    first_xmatch_name,
                    xmm_xmatch_name,
                    _2mass_xmatch["CSC21P_name"],
                    swift_name,
                    fermi_id,
                    torresi_detection,
                    distance.to_value("Mpc"),
                    L_OIII,
                    0,
                    L_NVSS,
                    0,
                    convert_F_nu_to_luminosity(
                        1.4 * u.GHz, first_xmatch_flux, u.mJy, distance
                    ),
                    convert_F_nu_to_luminosity(
                        1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distance
                    ),
                    _2mass_xmatch["flux_aper_b"] * distance.to_value("cm") ** 2,
                    _2mass_xmatch["flux_aper_lolim_b"] * distance.to_value("cm") ** 2,
                    _2mass_xmatch["flux_aper_hilim_b"] * distance.to_value("cm") ** 2,
                    convert_flux_to_luminosity(xmm_xmatch_flux,'mW / m**2', distance),
                    convert_flux_to_luminosity(xmm_xmatch_flux_err,'mW / m**2', distance),
                ]
            )

    if gaia_id and gaia_xmatches is not None:
        for gaia_xmatch in gaia_xmatches:
            coreg_fr0_expanded_catalogue.add_row(
                [
                    name,
                    "FR0",
                    sdss_id,
                    gaia_id,
                    _2mass_id,
                    nvss_id,
                    first_id,
                    "",
                    first_xmatch_name,
                    xmm_xmatch_name,
                    gaia_xmatch["CSC21P_name"],
                    swift_name,
                    fermi_id,
                    torresi_detection,
                    distance.to_value("Mpc"),
                    L_OIII,
                    0,
                    L_NVSS,
                    0,
                    convert_F_nu_to_luminosity(
                        1.4 * u.GHz, first_xmatch_flux, u.mJy, distance
                    ),
                    convert_F_nu_to_luminosity(
                        1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distance
                    ),
                    gaia_xmatch["flux_aper_b"] * distance.to_value("cm") ** 2,
                    gaia_xmatch["flux_aper_lolim_b"] * distance.to_value("cm") ** 2,
                    gaia_xmatch["flux_aper_hilim_b"] * distance.to_value("cm") ** 2,
                    convert_flux_to_luminosity(xmm_xmatch_flux,'mW / m**2', distance),
                    convert_flux_to_luminosity(xmm_xmatch_flux_err,'mW / m**2', distance),
                ]
            )

    else:
        coreg_fr0_expanded_catalogue.add_row(
            [
                name,
                "FR0",
                sdss_id,
                gaia_id,
                _2mass_id,
                nvss_id,
                first_id,
                "",
                first_xmatch_name,
                xmm_xmatch_name,
                "",
                swift_name,
                fermi_id,
                torresi_detection,
                distance.to_value("Mpc"),
                L_OIII,
                0,
                L_NVSS,
                0,
                convert_F_nu_to_luminosity(
                    1.4 * u.GHz, first_xmatch_flux, u.mJy, distance
                ),
                convert_F_nu_to_luminosity(
                    1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distance
                ),
                0,
                0,
                0,
                convert_flux_to_luminosity(xmm_xmatch_flux,'mW / m**2', distance),
                convert_flux_to_luminosity(xmm_xmatch_flux_err,'mW / m**2', distance),
            ]
        )


# In[32]:


coreg_fr0_expanded_catalogue


# In[36]:


np.array(coreg_fr0_expanded_catalogue['XMATCH_XMM_ID'][-109:-1])


# In[34]:


path = Path("./")
path.mkdir(exist_ok=True, parents=True)
coreg_fr0_expanded_catalogue.write(
    path / "coreg_fr0_expanded_catalogue.fits", overwrite=True
)


# In[ ]:




