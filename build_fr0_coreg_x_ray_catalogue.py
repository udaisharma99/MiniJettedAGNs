# gather all X-ray information available on CoreG and FR0 sources
import time
import logging
import numpy as np
from pathlib import Path
import astropy.units as u
from astropy.table import Table,  vstack 
from astropy.coordinates import Distance
from astropy.coordinates import SkyCoord, FK5
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from utils import (
    get_source_identifier,
    insert_space_source_ids,
    convert_F_nu_to_luminosity,
)
from table_structure import CatalogBuilder, table_coreG, table_fr0
  
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

def convert_flux_to_luminosity(flux, distance):

    flux = flux * u.mW/u.m**2
    D_L = distance.to(u.cm) # Convert Mpc to cm for consistent units
    flux_in_cgs = flux.to(u.erg / (u.s * u.cm**2))
    luminosity = 4 * np.pi * D_L**2 * flux_in_cgs
    
    return luminosity.to(u.erg / u.s)

# initial catalogues:
# - CoreG Catalogs
ho_1997 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/112/315")
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
# - FR0 Catalog
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")
#MORX Catalog
morx = Vizier(columns=["**"], row_limit=-1).get_catalogs("V/158/morxv2")

#SWIFT 2SXPS
twosxps_swift = Vizier(columns=["**"], row_limit=-1).get_catalogs("IX/58/2sxps")
twosxps_swift[0]['row_id'] = np.arange(len(twosxps_swift[0]))
twosxps_swift = twosxps_swift[0]['row_id','IAUName','RAJ2000','DEJ2000','HR1','E_HR1','e_HR1','HR2','E_HR2','e_HR2','FPO0','E_FPO0','e_FPO0','FPCU0','E_FPCU0','e_FPCU0']
twosxps_swift['FPCU0'] = np.where(twosxps_swift['FPCU0'] == '---', 0, twosxps_swift['FPCU0']).astype('float64')
twosxps_swift['E_FPCU0'] = np.where(twosxps_swift['E_FPCU0'] == '---', 0, twosxps_swift['E_FPCU0']).astype('float64')
twosxps_swift['e_FPCU0'] = np.where(twosxps_swift['e_FPCU0'] == '---', 0, twosxps_swift['e_FPCU0']).astype('float64')
twosxps_swift['FPCU0'] = twosxps_swift['FPCU0'] * u.Unit('mW/m**2')
twosxps_swift['E_FPCU0'] = twosxps_swift['E_FPCU0'] * u.Unit('mW/m**2')
twosxps_swift['e_FPCU0'] = twosxps_swift['e_FPCU0'] * u.Unit('mW/m**2')

#Swift BAT 105 month Catalog
bat_105 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/235/4/table3")
bat_105 = bat_105[0]

#4XMM Catalog
fourxmm = Table.read('4XMM_DR14cat_v1.0.csv', format='csv')
fourxmm['row_id'] = np.arange(len(fourxmm))
#Assigning units to the 4XMM table columns
fourxmm['ra'] = fourxmm['ra'] * u.deg
fourxmm['dec'] = fourxmm['dec'] * u.deg
cols = ['sc_ep_1_flux','sc_ep_1_flux_err',
        'sc_ep_2_flux','sc_ep_2_flux_err',
        'sc_ep_3_flux','sc_ep_3_flux_err',
        'sc_ep_4_flux','sc_ep_4_flux_err',
        'sc_ep_5_flux','sc_ep_5_flux_err',
        'sc_ep_8_flux','sc_ep_8_flux_err',
        'sc_ep_9_flux','sc_ep_9_flux_err']
for c in cols:
    fourxmm[c].unit = u.mW / u.m**2

fourxmm = fourxmm['row_id','iauname','ra','dec','sc_ep_1_flux','sc_ep_1_flux_err','sc_ep_2_flux','sc_ep_2_flux_err','sc_ep_3_flux','sc_ep_3_flux_err','sc_ep_4_flux','sc_ep_4_flux_err','sc_ep_5_flux','sc_ep_5_flux_err','sc_ep_8_flux','sc_ep_8_flux_err','sc_ep_9_flux','sc_ep_9_flux_err','sc_hr1','sc_hr2','sc_hr3','sc_hr4','sc_var_flag','n_detections']
fourxmm.rename_column('ra','RAJ2000')
fourxmm.rename_column('dec','DEJ2000')

#2CXO Catalog
cxotwo = Vizier(columns=["**"], row_limit=-1).get_catalogs("IX/70/csc21mas")
cxotwo[0]['row_id'] = np.arange(len(cxotwo[0]))
cxotwo = cxotwo[0]['row_id','2CXO','RAICRS','DEICRS','fv','Favgb','b_Favgb','B_Favgb','Favgh','b_Favgh','B_Favgh','Favgm','b_Favgm','B_Favgm','Favgs','b_Favgs','B_Favgs','Favgu','b_Favgu','B_Favgu','HRhm','b_HRhm','B_HRhm','HRhs','b_HRhs','B_HRhs','HRms','b_HRms','B_HRms','FPL0.5-7','b_FPL0.5-7','B_FPL0.5-7','GamPL','b_GamPL','B_GamPL']
#transforming ICRS to J2000 coordinates in 2CXO
coords_icrs = SkyCoord(ra=cxotwo['RAICRS'] ,
                       dec=cxotwo['DEICRS'] ,
                       frame="icrs")
coords_j2000 = coords_icrs.transform_to(FK5(equinox="J2000"))
cxotwo['RAICRS'] = coords_j2000.ra.deg
cxotwo['DEICRS'] = coords_j2000.dec.deg
cxotwo['RAICRS'] = cxotwo['RAICRS'] * u.deg
cxotwo['DEICRS'] = cxotwo['DEICRS'] * u.deg
cxotwo.rename_column('RAICRS', 'RAJ2000')
cxotwo.rename_column('DEICRS', 'DEJ2000')

#Fermi Transient Catalog - 1FLT
fermi_transient = Table.read('1FLT_final_V23.fits',format='fits')
fermi_transient['row_id'] = np.arange(len(fermi_transient))

#Making a coordinate system table for XMATCH since there are several masked columns in original table causing xmatch to crash!
twosxps_coords = twosxps_swift['row_id','IAUName','RAJ2000','DEJ2000']
fourxmm_coords = fourxmm['row_id','iauname','RAJ2000','DEJ2000']
cxotwo_coords = cxotwo['row_id','2CXO','RAJ2000','DEJ2000']

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
#Xmatch b/w Nagar 2005 and NVSS/FIRST/Swift 2SXPS/4XMM-DR14/2CXO
first_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:VIII/92/first14",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)
nvss_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:VIII/65/nvss",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)
morx_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:V/158/morxv2",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)

fourxmm_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2=fourxmm_coords,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 2 * u.arcsec,
)

twosxps_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2=twosxps_coords,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 3 * u.arcsec,
)
bat_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2=bat_105,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)

cxotwo_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2=cxotwo_coords,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 1 * u.arcsec,
)

ft_nagar_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2=fermi_transient,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 3 * u.arcmin,
)
nustar_fr0_xmatches = XMatch.query(
    cat1=nagar_2005[0],
    cat2="vizier:J/ApJ/836/99/table5",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=180 * u.arcsec,
)

#Xmatch b/w FR0CAT and NVSS/FIRST/Swift 2SXPS/4XMM-DR14/2CXO

first_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:VIII/92/first14",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)
nvss_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:VIII/65/nvss",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)
morx_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:V/158/morxv2",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)
fourxmm_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2=fourxmm_coords,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 3 * u.arcsec,
)
twosxps_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2=twosxps_coords,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 3 * u.arcsec,
)

bat_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2=bat_105,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=3 * u.arcsec,
)
cxotwo_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2=cxotwo_coords,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 1 * u.arcsec,
)
ft_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2=fermi_transient,
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance= 3 * u.arcmin,
)
nustar_fr0_xmatches = XMatch.query(
    cat1=fr0cat[0],
    cat2="vizier:J/ApJ/836/99/table5",
    colRA1="_RA",
    colDec1="_DE",
    colRA2="RAJ2000",
    colDec2="DEJ2000",
    max_distance=180 * u.arcsec,
)

coreG_catalogue = CatalogBuilder(table_coreG)

for name, _type, distance, F_15GHz in zip(
    nagar_2005[0]["Name"],
    nagar_2005[0]["AType"],
    nagar_2005[0]["Dist"],
    nagar_2005[0]["St15GHz"],
):
    if "L" in _type:
        log.info(f"considering {name}")
        
        #SIMBAD Identifiers
        sdss_id = get_source_identifier(name,"SDSS")
        nvss_id = get_source_identifier(name, "NVSS")
        first_id = get_source_identifier(name, "FIRST")
        fermi_id = get_source_identifier(name, "4FGL")
        distancee =   distance * u.Mpc
        
        #[OIII] Luminosity
        match_ho = ho_1997[1]["Name"] == insert_space_source_ids(name)
        if np.any(match_ho):  # Check if there is at least one True
            _log_L_alpha = ho_1997[1]["logL(Ha)"][match_ho][0]
            _OIII = ho_1997[1]["[OIII]"][match_ho][0]
            L_OIII = np.power(10, _log_L_alpha) * _OIII * u.Unit("erg s-1")
        else:
            L_OIII = 0 * u.Unit("erg s-1")

        # Fermi Transient 1FTL Name
        fermi_transient_name = ft_nagar_xmatches['Source_Name'] == name
        if fermi_transient_name.any():
            transient_name = ft_nagar_xmatches[fermi_transient_name][0]
        else:
            transient_name = ""
        
        # NVSS Measurement
        this_source_nvss_xmatch = nvss_nagar_xmatches["Name"] == name
        if this_source_nvss_xmatch.any():
            nvss_xmatch_name = (
                "NVSS J" + nvss_nagar_xmatches["NVSS"][this_source_nvss_xmatch][0]
            )
            nvss_xmatch_flux = nvss_nagar_xmatches["S1.4"][this_source_nvss_xmatch][0]
            nvss_xmatch_flux_err = nvss_nagar_xmatches["e_S1.4"][this_source_nvss_xmatch][0]
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
        this_source_first_xmatch = first_nagar_xmatches["Name"] == name
        if this_source_first_xmatch.any():
            first_xmatch_name = (
                "FIRST " + first_nagar_xmatches["FIRST"][this_source_first_xmatch][0]
            )
            first_xmatch_name_nospace = (
                "FIRST" + first_nagar_xmatches["FIRST"][this_source_first_xmatch][0]
            )
            first_xmatch_flux = first_nagar_xmatches["Fint"][this_source_first_xmatch][0]
            first_xmatch_flux_err = first_nagar_xmatches["Rms"][this_source_first_xmatch][0]
            
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
            morx_xmm = morx_nagar_xmatches["XMM-ID"][morx_matches][0]
            xmm4_name = morx_xmm.removeprefix("4XMM ")
            morx_cxo = morx_nagar_xmatches["CX-ID"][morx_matches][0]
            cx0_name = morx_cxo.removeprefix("CXOG ")
            morx_swift = morx_nagar_xmatches["Swift-ID"][morx_matches][0]
            swift_name = insert_space_source_ids(morx_swift)
            morx_first = morx_nagar_xmatches["FIRST-ID"][morx_matches][0]
            morx_nvss = morx_nagar_xmatches["NVSS-ID"][morx_matches][0]
            morx_lotss = morx_nagar_xmatches["LoTSS-ID"][morx_matches][0]
            morx_vlass = morx_nagar_xmatches["VLASS-ID"][morx_matches][0]
            morx_lobedist = morx_nagar_xmatches["Lobedist"][morx_matches][0]
        else:
            morx_xmm = ""
            morx_cxo = ""
            xmm4_name = ""
            cx0_name = ""
            morx_swift = ""
            swift_name = ""
            morx_first = ""
            morx_nvss = ""
            morx_lotss = ""
            morx_vlass = ""
            morx_lobedist = 0
        
        
        #4XMM-DR14 
        fourxmm_xmatch = fourxmm_nagar_xmatches['Name'] == name
        if fourxmm_xmatch.any():
            index_xmm = fourxmm_nagar_xmatches['row_id'][fourxmm_xmatch][0]
            index_match_xmm = fourxmm['row_id'] == index_xmm
            fourxmm_name = fourxmm['iauname'][index_match_xmm]
            fourxmm_var = fourxmm['sc_var_flag'][index_match_xmm][0]
            fourxmm_flux_1 = fourxmm['sc_ep_1_flux'][index_match_xmm][0]
            fourxmm_flux_1_err = fourxmm['sc_ep_1_flux_err'][index_match_xmm][0]
            fourxmm_flux_2 =fourxmm['sc_ep_2_flux'][index_match_xmm][0]
            fourxmm_flux_2_err = fourxmm['sc_ep_2_flux_err'][index_match_xmm][0]
            fourxmm_flux_3 = fourxmm['sc_ep_3_flux'][index_match_xmm][0]
            fourxmm_flux_3_err = fourxmm['sc_ep_3_flux_err'][index_match_xmm][0]
            fourxmm_flux_4 = fourxmm['sc_ep_4_flux'][index_match_xmm][0]
            fourxmm_flux_4_err = fourxmm['sc_ep_4_flux_err'][index_match_xmm][0]
            fourxmm_flux_5 = fourxmm['sc_ep_5_flux'][index_match_xmm][0]
            fourxmm_flux_5_err = fourxmm['sc_ep_5_flux_err'][index_match_xmm][0]
            fourxmm_flux_8 = fourxmm['sc_ep_8_flux'][index_match_xmm][0]
            fourxmm_flux_8_err = fourxmm['sc_ep_8_flux_err'][index_match_xmm][0]
            fourxmm_flux_9 = fourxmm['sc_ep_9_flux'][index_match_xmm][0]
            fourxmm_flux_9_err = fourxmm['sc_ep_9_flux_err'][index_match_xmm][0]
            fourxmm_hr1 = fourxmm['sc_hr1'][index_match_xmm][0]
            fourxmm_hr2 = fourxmm['sc_hr2'][index_match_xmm][0]
            fourxmm_hr3 = fourxmm['sc_hr3'][index_match_xmm][0]
            fourxmm_hr4 = fourxmm['sc_hr4'][index_match_xmm][0]
            xmm_detections = fourxmm['n_detections'][index_match_xmm][0]
            
        else:
            fourxmm_name = ""
            fourxmm_var = ""
            fourxmm_flux_1 = 0
            fourxmm_flux_1_err = 0
            fourxmm_flux_2 = 0
            fourxmm_flux_2_err = 0
            fourxmm_flux_3 = 0
            fourxmm_flux_3_err = 0
            fourxmm_flux_4 = 0
            fourxmm_flux_4_err = 0
            fourxmm_flux_5 = 0
            fourxmm_flux_5_err = 0
            fourxmm_flux_8 = 0
            fourxmm_flux_8_err = 0
            fourxmm_flux_9 = 0
            fourxmm_flux_9_err = 0
            fourxmm_hr1 = 0
            fourxmm_hr2 = 0
            fourxmm_hr3 = 0
            fourxmm_hr4 = 0
            xmm_detections = 0
        
        cxotwo_xmatch = cxotwo_nagar_xmatches['Name'] == name
        if cxotwo_xmatch.any():
            index_cxotwo = cxotwo_nagar_xmatches['row_id'][cxotwo_xmatch][0]
            index_match_cxotwo = cxotwo['row_id'] == index_cxotwo
            cxotwo_id = cxotwo['2CXO'][index_match_cxotwo][0]
            cxotwo_var = cxotwo['fv'][index_match_cxotwo][0]
            cxotwo_fpl_flux = cxotwo['FPL0.5-7'][index_match_cxotwo][0]
            cxotwo_fpl_flux_lerr = cxotwo['b_FPL0.5-7'][index_match_cxotwo][0]
            cxotwo_fpl_flux_uerr = cxotwo['B_FPL0.5-7'][index_match_cxotwo][0]
            cxotwo_fpl_phoindex = cxotwo['GamPL'][index_match_cxotwo][0]
            cxotwo_fpl_phoindex_lerr = cxotwo['b_GamPL'][index_match_cxotwo][0]
            cxotwo_fpl_phoindex_uerr = cxotwo['B_GamPL'][index_match_cxotwo][0]
            cxotwo_broad_flux = cxotwo['Favgb'][index_match_cxotwo][0]
            cxotwo_broad_flux_lerr= cxotwo['b_Favgb'][index_match_cxotwo][0]
            cxotwo_broad_flux_uerr = cxotwo['B_Favgb'][index_match_cxotwo][0]
            cxotwo_hard_flux = cxotwo['Favgh'][index_match_cxotwo][0]
            cxotwo_hard_flux_lerr = cxotwo['b_Favgh'][index_match_cxotwo][0]
            cxotwo_hard_flux_uerr = cxotwo['B_Favgh'][index_match_cxotwo][0]
            cxotwo_medium_flux = cxotwo['Favgm'][index_match_cxotwo][0]
            cxotwo_medium_flux_lerr = cxotwo['b_Favgm'][index_match_cxotwo][0]
            cxotwo_medium_flux_uerr = cxotwo['B_Favgm'][index_match_cxotwo][0]
            cxotwo_soft_flux = cxotwo['Favgs'][index_match_cxotwo][0]
            cxotwo_soft_flux_lerr = cxotwo['b_Favgs'][index_match_cxotwo][0]
            cxotwo_soft_flux_uerr = cxotwo['B_Favgs'][index_match_cxotwo][0]
            cxotwo_ultrasoft_flux = cxotwo['Favgu'][index_match_cxotwo][0]
            cxotwo_ultrasoft_flux_lerr = cxotwo['b_Favgu'][index_match_cxotwo][0]
            cxotwo_ultrasoft_flux_uerr = cxotwo['B_Favgu'][index_match_cxotwo][0]
            cxotwo_hr_hm = cxotwo['HRhm'][index_match_cxotwo][0]
            cxotwo_hr_hm_lerr = cxotwo['b_HRhm'][index_match_cxotwo][0]
            cxotwo_hr_hm_uerr = cxotwo['B_HRhm'][index_match_cxotwo][0]
            cxotwo_hr_hs = cxotwo['HRhs'][index_match_cxotwo][0]
            cxotwo_hr_hs_lerr = cxotwo['b_HRhs'][index_match_cxotwo][0]
            cxotwo_hr_hs_uerr = cxotwo['B_HRhs'][index_match_cxotwo][0]
            cxotwo_hr_ms = cxotwo['HRms'][index_match_cxotwo][0]
            cxotwo_hr_ms_lerr = cxotwo['b_HRms'][index_match_cxotwo][0]
            cxotwo_hr_ms_uerr = cxotwo['B_HRms'][index_match_cxotwo][0]
            
        else:
            cxotwo_id = ""
            cxotwo_var = "" 
            cxotwo_fpl_flux = 0
            cxotwo_fpl_flux_lerr = 0
            cxotwo_fpl_flux_uerr = 0
            cxotwo_fpl_phoindex = 0
            cxotwo_fpl_phoindex_lerr = 0
            cxotwo_fpl_phoindex_uerr = 0
            cxotwo_broad_flux = 0
            cxotwo_broad_flux_lerr = 0
            cxotwo_broad_flux_uerr = 0
            cxotwo_hard_flux = 0
            cxotwo_hard_flux_lerr = 0
            cxotwo_hard_flux_uerr = 0
            cxotwo_medium_flux = 0
            cxotwo_medium_flux_lerr = 0
            cxotwo_medium_flux_uerr = 0
            cxotwo_soft_flux = 0
            cxotwo_soft_flux_lerr = 0
            cxotwo_soft_flux_uerr = 0
            cxotwo_ultrasoft_flux = 0
            cxotwo_ultrasoft_flux_lerr = 0
            cxotwo_ultrasoft_flux_uerr = 0
            cxotwo_hr_hm = 0
            cxotwo_hr_hm_lerr = 0
            cxotwo_hr_hm_uerr = 0
            cxotwo_hr_hs = 0
            cxotwo_hr_hs_lerr = 0
            cxotwo_hr_hs_uerr = 0
            cxotwo_hr_ms = 0
            cxotwo_hr_ms_lerr = 0
            cxotwo_hr_ms_uerr = 0

        twosxps_match = twosxps_nagar_xmatches['Name'] == name
        if twosxps_match.any():
            index_twosxps = twosxps_nagar_xmatches['row_id'][twosxps_match][0]
            index_match_twosxps = twosxps_swift['row_id'] == index_twosxps
            twosxps_id = twosxps_swift['IAUName'][index_match_twosxps][0]
            twosxps_fpl_broadflux = twosxps_swift['FPCU0'][index_match_twosxps][0]
            twosxps_fpl_broadflux_lerr = twosxps_swift['e_FPCU0'][index_match_twosxps][0]
            twosxps_fpl_broadflux_uerr = twosxps_swift['E_FPCU0'][index_match_twosxps][0]
            twosxps_broadflux = twosxps_swift['FPO0'][index_match_twosxps][0]
            twosxps_broadflux_lerr = twosxps_swift['e_FPO0'][index_match_twosxps][0]
            twosxps_broadflux_uerr = twosxps_swift['E_FPO0'][index_match_twosxps][0]
            twosxps_hr_1 = twosxps_swift['HR1'][index_match_twosxps][0]
            twosxps_hr_1_lerr = twosxps_swift['e_HR1'][index_match_twosxps][0]
            twosxps_hr_1_uerr = twosxps_swift['E_HR1'][index_match_twosxps][0]
            twosxps_hr_2 = twosxps_swift['HR2'][index_match_twosxps][0]
            twosxps_hr_2_lerr = twosxps_swift['e_HR2'][index_match_twosxps][0]
            twosxps_hr_2_uerr = twosxps_swift['E_HR2'][index_match_twosxps][0]
        else:
            twosxps_id = ""
            twosxps_fpl_broadflux = 0
            twosxps_fpl_broadflux_lerr = 0
            twosxps_fpl_broadflux_uerr = 0
            twosxps_hr_1 = 0
            twosxps_hr_1_lerr = 0
            twosxps_hr_1_uerr = 0
            twosxps_hr_2 = 0
            twosxps_hr_2_lerr = 0
            twosxps_hr_2_uerr = 0
            twosxps_broadflux = 0
            twosxps_broadflux_lerr = 0
            twosxps_broadflux_uerr = 0

        bat_xmatch = bat_nagar_xmatches["Name"] == name
        if bat_xmatch.any():
            bat_id = (
                "BAT " + bat_nagar_xmatches["Swift"][bat_xmatch][0]
            )
        else:
            bat_id = ""

        # check if the source is in the list of sources detected by Torresi et al. 2018
        torresi_detection = sdss_id in torresi_sources

        coreG_catalogue.add_source(
            name, 
            _type, 
            morx_lotss, 
            morx_vlass, 
            nvss_id, 
            first_id,
            sdss_id,
            morx_xmm,
            fourxmm_name,
            morx_cxo,
            cxotwo_id,
            morx_swift,
            twosxps_id,
            bat_id,
            fermi_id,
            transient_name,
            torresi_detection,
            fourxmm_var,
            cxotwo_var,
            morx_lobedist,
            distancee, 
            np.log10(L_OIII.to_value("erg s-1")),
            convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux, u.mJy, distancee),
            convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux_err, u.mJy, distancee),
            convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux, u.mJy, distancee),
            convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distancee),
            convert_flux_to_luminosity(fourxmm_flux_1, distancee),
            convert_flux_to_luminosity(fourxmm_flux_1_err, distancee),
            convert_flux_to_luminosity(fourxmm_flux_2, distancee),
            convert_flux_to_luminosity(fourxmm_flux_2_err, distancee),
            convert_flux_to_luminosity(fourxmm_flux_3, distancee),
            convert_flux_to_luminosity(fourxmm_flux_3_err, distancee), 
            convert_flux_to_luminosity(fourxmm_flux_4, distancee),
            convert_flux_to_luminosity(fourxmm_flux_4_err, distancee),
            convert_flux_to_luminosity(fourxmm_flux_5, distancee),
            convert_flux_to_luminosity(fourxmm_flux_5_err, distancee),
            convert_flux_to_luminosity(fourxmm_flux_8, distancee),
            convert_flux_to_luminosity(fourxmm_flux_8_err, distancee),
            convert_flux_to_luminosity(fourxmm_flux_9, distancee),
            convert_flux_to_luminosity(fourxmm_flux_9_err, distancee),
            fourxmm_hr1,
            fourxmm_hr2,
            fourxmm_hr3,
            fourxmm_hr4,
            xmm_detections,
            convert_flux_to_luminosity(cxotwo_fpl_flux, distancee),
            convert_flux_to_luminosity(cxotwo_fpl_flux_lerr, distancee),
            convert_flux_to_luminosity(cxotwo_fpl_flux_uerr, distancee),
            cxotwo_fpl_phoindex,
            cxotwo_fpl_phoindex_lerr,
            cxotwo_fpl_phoindex_uerr,
            convert_flux_to_luminosity(cxotwo_broad_flux, distancee),
            convert_flux_to_luminosity(cxotwo_broad_flux_lerr, distancee),
            convert_flux_to_luminosity(cxotwo_broad_flux_uerr, distancee),
            convert_flux_to_luminosity(cxotwo_hard_flux, distancee),
            convert_flux_to_luminosity(cxotwo_hard_flux_lerr, distancee),
            convert_flux_to_luminosity(cxotwo_hard_flux_uerr, distancee),
            convert_flux_to_luminosity(cxotwo_medium_flux, distancee),
            convert_flux_to_luminosity(cxotwo_medium_flux_lerr, distancee),
            convert_flux_to_luminosity(cxotwo_medium_flux_uerr, distancee),
            convert_flux_to_luminosity(cxotwo_soft_flux, distancee),
            convert_flux_to_luminosity(cxotwo_soft_flux_lerr, distancee),
            convert_flux_to_luminosity(cxotwo_soft_flux_uerr, distancee),
            convert_flux_to_luminosity(cxotwo_ultrasoft_flux, distancee),
            convert_flux_to_luminosity(cxotwo_ultrasoft_flux_lerr, distancee),
            convert_flux_to_luminosity(cxotwo_ultrasoft_flux_uerr, distancee),
            cxotwo_hr_hm,
            cxotwo_hr_hm_lerr,
            cxotwo_hr_hm_uerr,
            cxotwo_hr_hs,
            cxotwo_hr_hs_lerr,
            cxotwo_hr_hs_uerr,
            cxotwo_hr_ms,
            cxotwo_hr_ms_lerr,
            cxotwo_hr_ms_uerr,
            convert_flux_to_luminosity(twosxps_fpl_broadflux, distancee),
            convert_flux_to_luminosity(twosxps_fpl_broadflux_lerr, distancee),
            convert_flux_to_luminosity(twosxps_fpl_broadflux_uerr, distancee),
            convert_flux_to_luminosity(twosxps_broadflux, distancee),
            convert_flux_to_luminosity(twosxps_broadflux_lerr, distancee),
            convert_flux_to_luminosity(twosxps_broadflux_uerr, distancee),
            twosxps_hr_1,
            twosxps_hr_1_lerr,
            twosxps_hr_1_uerr,
            twosxps_hr_2,
            twosxps_hr_2_lerr,
            twosxps_hr_2_uerr
        )

fr0_catalogue = CatalogBuilder(table_fr0)

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
   
    # Fermi Transient 1FTL Name
    fermi_transient_name = ft_nagar_xmatches['Source_Name'] == name
    if fermi_transient_name.any():
        transient_name = ft_nagar_xmatches[fermi_transient_name][0]
    else:
        transient_name = ""
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
    
    morx_matches = morx_fr0_xmatches["SimbadName"] == sdss_id
    if morx_matches.any():
        morx_xmm = morx_fr0_xmatches["XMM-ID"][morx_matches][0]
        xmm4_name = morx_xmm.removeprefix("4XMM ")
        morx_cxo = morx_fr0_xmatches["CX-ID"][morx_matches][0]
        cx0_name = morx_cxo.removeprefix("CXOG ")
        morx_swift = morx_fr0_xmatches["Swift-ID"][morx_matches][0]
        swift_name = insert_space_source_ids(morx_swift)
        morx_first = morx_fr0_xmatches["FIRST-ID"][morx_matches][0]
        morx_nvss = morx_fr0_xmatches["NVSS-ID"][morx_matches][0]
        morx_lotss = morx_fr0_xmatches["LoTSS-ID"][morx_matches][0]
        morx_vlass = morx_fr0_xmatches["VLASS-ID"][morx_matches][0]
        morx_lobedist = morx_fr0_xmatches["Lobedist"][morx_matches][0]
    else:
        morx_xmm = ""
        morx_cxo = ""
        xmm4_name = ""
        cx0_name = ""
        morx_swift = ""
        swift_name = ""
        morx_first = ""
        morx_nvss = ""
        morx_lotss = ""
        morx_vlass = ""
        morx_lobedist = 0

        #4XMM-DR14 
    fourxmm_xmatch = fourxmm_fr0_xmatches['SimbadName'] == sdss_id
    if fourxmm_xmatch.any():
        index_xmm = fourxmm_fr0_xmatches['row_id'][fourxmm_xmatch][0]
        index_match_xmm = fourxmm['row_id'] == index_xmm
        fourxmm_name = fourxmm['iauname'][index_match_xmm][0]
        fourxmm_var = fourxmm['sc_var_flag'][index_match_xmm][0]
        fourxmm_flux_1 = fourxmm['sc_ep_1_flux'][index_match_xmm][0]
        fourxmm_flux_1_err = fourxmm['sc_ep_1_flux_err'][index_match_xmm][0]
        fourxmm_flux_2 =fourxmm['sc_ep_2_flux'][index_match_xmm][0]
        fourxmm_flux_2_err = fourxmm['sc_ep_2_flux_err'][index_match_xmm][0]
        fourxmm_flux_3 = fourxmm['sc_ep_3_flux'][index_match_xmm][0]
        fourxmm_flux_3_err = fourxmm['sc_ep_3_flux_err'][index_match_xmm][0]
        fourxmm_flux_4 = fourxmm['sc_ep_4_flux'][index_match_xmm][0]
        fourxmm_flux_4_err = fourxmm['sc_ep_4_flux_err'][index_match_xmm][0]
        fourxmm_flux_5 = fourxmm['sc_ep_5_flux'][index_match_xmm][0]
        fourxmm_flux_5_err = fourxmm['sc_ep_5_flux_err'][index_match_xmm][0]
        fourxmm_flux_8 = fourxmm['sc_ep_8_flux'][index_match_xmm][0]
        fourxmm_flux_8_err = fourxmm['sc_ep_8_flux_err'][index_match_xmm][0]
        fourxmm_flux_9 = fourxmm['sc_ep_9_flux'][index_match_xmm][0]
        fourxmm_flux_9_err = fourxmm['sc_ep_9_flux_err'][index_match_xmm][0]
        fourxmm_hr1 = fourxmm['sc_hr1'][index_match_xmm][0]
        fourxmm_hr2 = fourxmm['sc_hr2'][index_match_xmm][0]
        fourxmm_hr3 = fourxmm['sc_hr3'][index_match_xmm][0]
        fourxmm_hr4 = fourxmm['sc_hr4'][index_match_xmm][0]
        xmm_detections = fourxmm['n_detections'][index_match_xmm][0]
            
    else:
        fourxmm_name = ""
        fourxmm_var = ""
        fourxmm_flux_1 = 0
        fourxmm_flux_1_err = 0
        fourxmm_flux_2 = 0
        fourxmm_flux_2_err = 0
        fourxmm_flux_3 = 0
        fourxmm_flux_3_err = 0
        fourxmm_flux_4 = 0
        fourxmm_flux_4_err = 0
        fourxmm_flux_5 = 0
        fourxmm_flux_5_err = 0
        fourxmm_flux_8 = 0
        fourxmm_flux_8_err = 0
        fourxmm_flux_9 = 0
        fourxmm_flux_9_err = 0
        fourxmm_hr1 = 0
        fourxmm_hr2 = 0
        fourxmm_hr3 = 0
        fourxmm_hr4 = 0
        xmm_detections = 0
    
    cxotwo_xmatch = cxotwo_fr0_xmatches['SimbadName'] == sdss_id
    if cxotwo_xmatch.any():
        index_cxotwo = cxotwo_fr0_xmatches['row_id'][cxotwo_xmatch][0]
        index_match_cxotwo = cxotwo['row_id'] == index_cxotwo
        cxotwo_id = cxotwo['2CXO'][index_match_cxotwo][0]
        cxotwo_var = cxotwo['fv'][index_match_cxotwo][0]
        cxotwo_fpl_flux = cxotwo['FPL0.5-7'][index_match_cxotwo][0]
        cxotwo_fpl_flux_lerr = cxotwo['b_FPL0.5-7'][index_match_cxotwo][0]
        cxotwo_fpl_flux_uerr = cxotwo['B_FPL0.5-7'][index_match_cxotwo][0]
        cxotwo_fpl_phoindex = cxotwo['GamPL'][index_match_cxotwo][0]
        cxotwo_fpl_phoindex_lerr = cxotwo['b_GamPL'][index_match_cxotwo][0]
        cxotwo_fpl_phoindex_uerr = cxotwo['B_GamPL'][index_match_cxotwo][0]
        cxotwo_broad_flux = cxotwo['Favgb'][index_match_cxotwo][0]
        cxotwo_broad_flux_lerr= cxotwo['b_Favgb'][index_match_cxotwo][0]
        cxotwo_broad_flux_uerr = cxotwo['B_Favgb'][index_match_cxotwo][0]
        cxotwo_hard_flux = cxotwo['Favgh'][index_match_cxotwo][0]
        cxotwo_hard_flux_lerr = cxotwo['b_Favgh'][index_match_cxotwo][0]
        cxotwo_hard_flux_uerr = cxotwo['B_Favgh'][index_match_cxotwo][0]
        cxotwo_medium_flux = cxotwo['Favgm'][index_match_cxotwo][0]
        cxotwo_medium_flux_lerr = cxotwo['b_Favgm'][index_match_cxotwo][0]
        cxotwo_medium_flux_uerr = cxotwo['B_Favgm'][index_match_cxotwo][0]
        cxotwo_soft_flux = cxotwo['Favgs'][index_match_cxotwo][0]
        cxotwo_soft_flux_lerr = cxotwo['b_Favgs'][index_match_cxotwo][0]
        cxotwo_soft_flux_uerr = cxotwo['B_Favgs'][index_match_cxotwo][0]
        cxotwo_ultrasoft_flux = cxotwo['Favgu'][index_match_cxotwo][0]
        cxotwo_ultrasoft_flux_lerr = cxotwo['b_Favgu'][index_match_cxotwo][0]
        cxotwo_ultrasoft_flux_uerr = cxotwo['B_Favgu'][index_match_cxotwo][0]
        cxotwo_hr_hm = cxotwo['HRhm'][index_match_cxotwo][0]
        cxotwo_hr_hm_lerr = cxotwo['b_HRhm'][index_match_cxotwo][0]
        cxotwo_hr_hm_uerr = cxotwo['B_HRhm'][index_match_cxotwo][0]
        cxotwo_hr_hs = cxotwo['HRhs'][index_match_cxotwo][0]
        cxotwo_hr_hs_lerr = cxotwo['b_HRhs'][index_match_cxotwo][0]
        cxotwo_hr_hs_uerr = cxotwo['B_HRhs'][index_match_cxotwo][0]
        cxotwo_hr_ms = cxotwo['HRms'][index_match_cxotwo][0]
        cxotwo_hr_ms_lerr = cxotwo['b_HRms'][index_match_cxotwo][0]
        cxotwo_hr_ms_uerr = cxotwo['B_HRms'][index_match_cxotwo][0]
            
    else:
        cxotwo_id = ""
        cxotwo_var = "" 
        cxotwo_fpl_flux = 0
        cxotwo_fpl_flux_lerr = 0
        cxotwo_fpl_flux_uerr = 0
        cxotwo_fpl_phoindex = 0
        cxotwo_fpl_phoindex_lerr = 0
        cxotwo_fpl_phoindex_uerr = 0
        cxotwo_broad_flux = 0
        cxotwo_broad_flux_lerr = 0
        cxotwo_broad_flux_uerr = 0
        cxotwo_hard_flux = 0
        cxotwo_hard_flux_lerr = 0
        cxotwo_hard_flux_uerr = 0
        cxotwo_medium_flux = 0
        cxotwo_medium_flux_lerr = 0
        cxotwo_medium_flux_uerr = 0
        cxotwo_soft_flux = 0
        cxotwo_soft_flux_lerr = 0
        cxotwo_soft_flux_uerr = 0
        cxotwo_ultrasoft_flux = 0
        cxotwo_ultrasoft_flux_lerr = 0
        cxotwo_ultrasoft_flux_uerr = 0
        cxotwo_hr_hm = 0
        cxotwo_hr_hm_lerr = 0
        cxotwo_hr_hm_uerr = 0
        cxotwo_hr_hs = 0
        cxotwo_hr_hs_lerr = 0
        cxotwo_hr_hs_uerr = 0
        cxotwo_hr_ms = 0
        cxotwo_hr_ms_lerr = 0
        cxotwo_hr_ms_uerr = 0

    twosxps_match = twosxps_fr0_xmatches['SimbadName'] == sdss_id
    if twosxps_match.any():
        index_twosxps = twosxps_fr0_xmatches['row_id'][twosxps_match][0]
        index_match_twosxps = twosxps_swift['row_id'] == index_twosxps
        twosxps_id = twosxps_swift['IAUName'][index_match_twosxps][0]
        twosxps_fpl_broadflux = twosxps_swift['FPCU0'][index_match_twosxps][0]
        twosxps_fpl_broadflux_lerr = twosxps_swift['e_FPCU0'][index_match_twosxps][0]
        twosxps_fpl_broadflux_uerr = twosxps_swift['E_FPCU0'][index_match_twosxps][0]
        twosxps_broadflux = twosxps_swift['FPO0'][index_match_twosxps][0]
        twosxps_broadflux_lerr = twosxps_swift['e_FPO0'][index_match_twosxps][0]
        twosxps_broadflux_uerr = twosxps_swift['E_FPO0'][index_match_twosxps][0]
        twosxps_hr_1 = twosxps_swift['HR1'][index_match_twosxps][0]
        twosxps_hr_1_lerr = twosxps_swift['e_HR1'][index_match_twosxps][0]
        twosxps_hr_1_uerr = twosxps_swift['E_HR1'][index_match_twosxps][0]
        twosxps_hr_2 = twosxps_swift['HR2'][index_match_twosxps][0]
        twosxps_hr_2_lerr = twosxps_swift['e_HR2'][index_match_twosxps][0]
        twosxps_hr_2_uerr = twosxps_swift['E_HR2'][index_match_twosxps][0]
    else:
        twosxps_id = ""
        twosxps_fpl_broadflux = 0
        twosxps_fpl_broadflux_lerr = 0
        twosxps_fpl_broadflux_uerr = 0
        twosxps_hr_1 = 0
        twosxps_hr_1_lerr = 0
        twosxps_hr_1_uerr = 0
        twosxps_hr_2 = 0
        twosxps_hr_2_lerr = 0
        twosxps_hr_2_uerr = 0
        twosxps_broadflux = 0
        twosxps_broadflux_lerr = 0
        twosxps_broadflux_uerr = 0

    bat_xmatch = bat_fr0_xmatches["SimbadName"] == sdss_id
    if bat_xmatch.any():
        bat_id = (
            "BAT " + bat_fr0_xmatches["Swift"][bat_xmatch][0]
        )
    else:
            bat_id = ""    
  
        # check if the source is in the list of sources detected by Torresi et al. 2018
    torresi_detection = sdss_id in torresi_sources
    
    fr0_catalogue.add_source(
            name, 
            "FR0", 
            morx_lotss, 
            morx_vlass, 
            nvss_id, 
            first_id,
            sdss_id,
            morx_xmm,
            fourxmm_name,
            morx_cxo,
            cxotwo_id,
            morx_swift,
            twosxps_id,
            bat_id,
            fermi_id,
            transient_name,
            torresi_detection,
            fourxmm_var,
            cxotwo_var,
            morx_lobedist,
            distancee, 
            np.log10(L_OIII.to_value("erg s-1")),
            convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux, u.mJy, distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, nvss_xmatch_flux_err, u.mJy, distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux, u.mJy, distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, first_xmatch_flux_err, u.mJy, distance),
            convert_flux_to_luminosity(fourxmm_flux_1, distance),
            convert_flux_to_luminosity(fourxmm_flux_1_err, distance),
            convert_flux_to_luminosity(fourxmm_flux_2, distance),
            convert_flux_to_luminosity(fourxmm_flux_2_err, distance),
            convert_flux_to_luminosity(fourxmm_flux_3, distance),
            convert_flux_to_luminosity(fourxmm_flux_3_err, distance), 
            convert_flux_to_luminosity(fourxmm_flux_4, distance),
            convert_flux_to_luminosity(fourxmm_flux_4_err, distance),
            convert_flux_to_luminosity(fourxmm_flux_5, distance),
            convert_flux_to_luminosity(fourxmm_flux_5_err, distance),
            convert_flux_to_luminosity(fourxmm_flux_8, distance),
            convert_flux_to_luminosity(fourxmm_flux_8_err, distance),
            convert_flux_to_luminosity(fourxmm_flux_9, distance),
            convert_flux_to_luminosity(fourxmm_flux_9_err, distance),
            fourxmm_hr1,
            fourxmm_hr2,
            fourxmm_hr3,
            fourxmm_hr4,
            xmm_detections,
            convert_flux_to_luminosity(cxotwo_fpl_flux, distance),
            convert_flux_to_luminosity(cxotwo_fpl_flux_lerr, distance),
            convert_flux_to_luminosity(cxotwo_fpl_flux_uerr, distance),
            cxotwo_fpl_phoindex,
            cxotwo_fpl_phoindex_lerr,
            cxotwo_fpl_phoindex_uerr,
            convert_flux_to_luminosity(cxotwo_broad_flux, distance),
            convert_flux_to_luminosity(cxotwo_broad_flux_lerr, distance),
            convert_flux_to_luminosity(cxotwo_broad_flux_uerr, distance),
            convert_flux_to_luminosity(cxotwo_hard_flux, distance),
            convert_flux_to_luminosity(cxotwo_hard_flux_lerr, distance),
            convert_flux_to_luminosity(cxotwo_hard_flux_uerr, distance),
            convert_flux_to_luminosity(cxotwo_medium_flux, distance),
            convert_flux_to_luminosity(cxotwo_medium_flux_lerr, distance),
            convert_flux_to_luminosity(cxotwo_medium_flux_uerr, distance),
            convert_flux_to_luminosity(cxotwo_soft_flux, distance),
            convert_flux_to_luminosity(cxotwo_soft_flux_lerr, distance),
            convert_flux_to_luminosity(cxotwo_soft_flux_uerr, distance),
            convert_flux_to_luminosity(cxotwo_ultrasoft_flux, distance),
            convert_flux_to_luminosity(cxotwo_ultrasoft_flux_lerr, distance),
            convert_flux_to_luminosity(cxotwo_ultrasoft_flux_uerr, distance),
            cxotwo_hr_hm,
            cxotwo_hr_hm_lerr,
            cxotwo_hr_hm_uerr,
            cxotwo_hr_hs,
            cxotwo_hr_hs_lerr,
            cxotwo_hr_hs_uerr,
            cxotwo_hr_ms,
            cxotwo_hr_ms_lerr,
            cxotwo_hr_ms_uerr,
            convert_flux_to_luminosity(twosxps_fpl_broadflux, distance),
            convert_flux_to_luminosity(twosxps_fpl_broadflux_lerr, distance),
            convert_flux_to_luminosity(twosxps_fpl_broadflux_uerr, distance),
            convert_flux_to_luminosity(twosxps_broadflux, distance),
            convert_flux_to_luminosity(twosxps_broadflux_lerr, distance),
            convert_flux_to_luminosity(twosxps_broadflux_uerr, distance),
            twosxps_hr_1,
            twosxps_hr_1_lerr,
            twosxps_hr_1_uerr,
            twosxps_hr_2,
            twosxps_hr_2_lerr,
            twosxps_hr_2_uerr
        )

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
