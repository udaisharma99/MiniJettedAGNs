# structure of the table for the final catalgoue
from contextlib import ContextDecorator
from pyexpat.model import XML_CTYPE_MIXED
from tkinter import PhotoImage
import astropy
import pandas as pd
import logging
import numpy as np
import astropy.units as u
from astropy.table import Column
from astropy.coordinates import Distance
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.table import Table
from data_structures import columns
from utils import (
    insert_space_source_ids,
    get_simbad_coordinates,
    get_source_survey_identifier,
    convert_F_nu_to_luminosity,
    convert_flux_to_luminosity
)

# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)

#Generate the table structure
table = Table(columns)

# Loading tables with only required columns
##Import Source and Counterpart Catalogs
nagar_2005 = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/435/521")
fr0cat = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/A+A/609/A1")
# Import Counterpart Catalogs
ho_1997 = Vizier(columns=["Name", "AType", "logL(Ha)", "[OIII]"], row_limit=-1).get_catalogs("J/ApJS/112/315")
nvss = Vizier(columns=["NVSS", "S1.4", "e_S1.4"], row_limit=-1).get_catalogs("VIII/65/nvss")
first = Vizier(columns=["FIRST", "Fint", "Rms"], row_limit=-1).get_catalogs("VIII/92/first14")
cols_xmm = ['iauname','ra','dec','sc_ep_4_flux','sc_ep_4_flux_err','sc_ep_5_flux','sc_ep_5_flux_err','sc_hr3','sc_hr4','sc_hr3_err','sc_hr4_err','sc_var_flag']
df = pd.read_csv('./data/catalogues/4XMM_DR14cat_v1.0.csv', usecols=cols_xmm)
fourxmm = Table.from_pandas(df)
cols_csc = ["RAICRS","DEICRS","2CXO","FPL0.5-7","b_FPL0.5-7","B_FPL0.5-7","GamPL","b_GamPL","B_GamPL","HRhm","b_HRhm","B_HRhm","fv"]
cxotwo = Vizier(columns=cols_csc, row_limit=-1).get_catalogs("IX/70/csc21mas")
cols_twosxps = ["RAJ2000","DEJ2000","IAUName","FPCO0","e_FPCO0","E_FPCO0","Gamma","e_Gamma","E_Gamma","HR2","e_HR2","E_HR2"]
twosxps_swift = Vizier(columns=cols_twosxps, row_limit=-1).get_catalogs("IX/58/2sxps")
bat157 = Table.read('./data/catalogues/BAT_157m.txt', format='ascii',delimiter='|')
fermi_4fgl = Table.read('./data/catalogues/4fgl-dr4.fit',format='fits')
fermi_transient = Table.read('./data/catalogues/1FLT_final_V23.fits',format='fits')
cols_morx = ["RAJ2000", "DEJ2000","XMM-ID", "CX-ID", "Swift-ID", "LoTSS-ID", "VLASS-ID",'Lobedist']
morx = Vizier(columns=cols_morx,row_limit=-1).get_catalogs("V/158/morxv2")
x_ray_catalogs = [morx[0], fourxmm, cxotwo[0], twosxps_swift[0], bat157]
gamma_ray_catalogs = [fermi_4fgl, fermi_transient]

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


def reformat_ned_table(ned_table):
    """Reformat the uncertainties column in the NED photometry table.
    The uncertainties column contains strings like "<0.1" for upper limits.
    We will create a new boolean column "is_ul" to indicate if the measurement
    is an upper limit or not, and we will convert the uncertainties to float.
    Correct also logarithmic quantities.
    Watchout because some uncertainties are percentual!
    Nice messy table from NED...
    """
    # first correct upper limits and uncertainties
    values = []
    uncertainties = []
    units = []
    is_ul = []

    for value, uncertainty, unit in zip(
        ned_table["Photometry Measurement"],
        ned_table["Uncertainty"],
        ned_table["Units"],
    ):
        # strip possible problematic characters
        unit = unit.strip("^").replace("(", "").replace(")", "")
        # now let us go case by case:
        # - case 1.1: this is an upper limit
        if uncertainty.startswith("<"):
            uncertainty = uncertainty.replace("<", "")
            is_ul.append(True)
            uncertainties.append(float("nan"))
            if not unit.startswith("log"):
                values.append(float(uncertainty))
                units.append(unit)
            if unit.startswith("log"):
                values.append(10 ** float(uncertainty))
                units.append(unit.replace("log", ""))
        # - case 2.1: this is not an upper limit and it is not a log quantity
        elif uncertainty.startswith("+/-") and not unit.startswith("log"):
            is_ul.append(False)
            values.append(float(value))
            # check if the uncertainty is percentual
            uncertainty = uncertainty.replace("+/-", "")
            if uncertainty.endswith("%"):
                perc = float(uncertainty.replace("%", ""))
                uncertainties.append(float(value) * perc / 100)
            else:
                uncertainties.append(float(uncertainty))
            units.append(unit)
        # - case 2.2: this is not an upper limit and it is a log quantity
        elif uncertainty.startswith("+/-") and unit.startswith("log"):
            is_ul.append(False)
            values.append(10 ** float(value))
            uncertainty = uncertainty.replace("+/-", "")
            # check if the uncertainty is percentual
            uncertainty = uncertainty.replace("+/-", "")
            if uncertainty.endswith("%"):
                perc = float(uncertainty.replace("%", ""))
                uncertainties.append(10 ** float(value) * perc / 100)
            else:
                uncertainties.append(10 ** float(uncertainty))
            units.append(unit.replace("log", ""))
        # - case 3: no uncertainty provided
        else:
            # check if this is logarithmic
            is_ul.append(False)
            uncertainties.append(float("nan"))
            if unit.startswith("log"):
                values.append(10 ** float(value))
                units.append(unit.replace("log", ""))
            else:
                values.append(value)
                units.append(unit)

    ned_table.add_column(Column(values, name="flux", dtype=np.float64))
    ned_table.add_column(Column(uncertainties, name="flux_err", dtype=np.float64))
    ned_table.add_column(Column(is_ul, name="is_ul", dtype=bool))
    ned_table.add_column(Column(units, name="unit", dtype=str))
    # remove old columns
    ned_table.remove_columns(["Photometry Measurement", "Uncertainty", "Units"])
    return ned_table


def get_flux_measurements_from_ned_table(ned_table, band):
    """Get the spectral lines or flux measurement - let us use the term `band`
    to indicate both - from the NED table for a given source.
    The table has been obtained with the
    `Ned.get_table(name, table=`photometry`)` method.
    """
    mask = [_.startswith(band) for _ in ned_table["Observed Passband"]]
    # let us fetch only fundamental information: flux, its units, and uncertainty
    table = ned_table[
        "Observed Passband", "Photometry Measurement", "Uncertainty", "Units"
    ][mask]
    # let us fix the mess with the uncertainties
    return reformat_ned_table(table)


class Source:
    """
    Class to represent a single source in the catalogue.
    It will contain basic information like coordinates, name, and type.
    It will also contain basic methods to find FIRST, NVSS and SDSS names.
    We will create another structure to hold the X-ray information.
    """

    def __init__(self, name):
        self.name = name
        # already at initialisation, we find the NVSS, FIRST and SDSS counterparts
        # in principle all the sources should be in these deep surveys
        self.ra, self.dec = get_simbad_coordinates(self.name)
        self.source_coords = SkyCoord(
            ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs"
        )
        self.x_ray_catalogs = x_ray_catalogs
        self.gamma_ray_catalogs = gamma_ray_catalogs
        # let us search for radio,optical, X-ray and gamma-ray counterparts in various catalogues
        self.find_nvss_first_sdss_counterparts()
        self.search_x_ray_counterparts()
        self.search_gamma_ray_counterparts()
        source_coreG = nagar_2005[0]["Name"] == self.name
        if np.any(source_coreG):
            self.distance = nagar_2005[0]["Dist"][source_coreG][0] * u.Mpc
            self.source_type = nagar_2005[0]["AType"][source_coreG][0]
        else:
            fr0_match = self.name == fr0cat[0]["SimbadName"]
            redshift = fr0cat[0]["z"][fr0_match][0]
            self.distance = Distance(z=redshift).to("Mpc")
            self.source_type = "FR0"
        self.torresi_detection = self.sdss_id_simbad in torresi_sources
        self.get_OIII_luminosity()
        # let us load all the photometric measurements, but let us filter only
        # those of interest to us (e.g. optical lines and radio fluxes at 15 GHz)
        pass
        """
        ned_table = Ned.get_table(self.name, table="photometry")
        band_list = [
            "H{alpha}",
            "H{beta}",
            "[O III] 5007",
            "[O I] 6300",
            "[S II]",
            "1.4 GHz",
            "5 GHz",
            "15 GHz",
        ]
        tables_list = [
            get_flux_measurements_from_ned_table(ned_table, band) for band in band_list
        ]
        self.ned_flux_table = vstack(tables_list)
        #self.check_survey_ids_and_flux_measurements()
        """
    def get_OIII_luminosity(self):
        """Calculate the [OIII] luminosity from Ho et al. (1997) or FR0CAT."""
        match_ho = ho_1997[1]["Name"] == insert_space_source_ids(self.name)
        if np.any(match_ho):  # Check if there is at least one True
            _log_L_alpha = ho_1997[1]["logL(Ha)"][match_ho][0]
            _OIII = ho_1997[1]["[OIII]"][match_ho][0]
            L_OIII = np.power(10, _log_L_alpha) * _OIII * u.Unit("erg s-1")
            self.LogL_OIII = np.log10(L_OIII.to_value("erg s-1"))
        elif np.any(self.sdss_id_simbad == fr0cat[0]["SDSS"]):
            sdss_match = self.sdss_id_simbad == fr0cat[0]["SDSS"]
            self.LogL_OIII = fr0cat[0]["logL[OIII]"][sdss_match][0]
        else:
            self.LogL_OIII = 0

    def find_nvss_first_sdss_counterparts(self):
        """Find the NVSS, FIRST, and SDSS identifiers.
        In principle all the sources should be in these deep surveys.
        """
        # first search through SIMBAD
        self.sdss_id_simbad = get_source_survey_identifier(self.name, "SDSS")
        self.nvss_id_simbad = get_source_survey_identifier(self.name, "NVSS")
        nvss_stripped = self.nvss_id_simbad.lstrip("NVSS J")
        nvss_match = nvss_stripped == nvss[0]["NVSS"]
        if np.any(nvss_match):
            self.nvss_flux = nvss[0]["S1.4"][nvss_match][0] 
            self.nvss_flux_error = nvss[0]["e_S1.4"][nvss_match][0] 
        else:
            self.nvss_flux = 0
            self.nvss_flux_error = 0
        self.first_id_simbad = get_source_survey_identifier(self.name, "FIRST")
        first_stripped = self.first_id_simbad.lstrip("FIRST ")
        first_match = first_stripped == first[0]["FIRST"]
        if np.any(first_match):
            self.first_flux = first[0]["Fint"][first_match][0] 
            self.first_flux_error = first[0]["Rms"][first_match][0]
        else:
            self.first_flux = 0
            self.first_flux_error = 0

    def check_survey_ids_and_flux_measurements(self):
        """Check, once that a source has NVSS and FIRST identifiers, that it has also
        the corresponding flux measurements.
        Call after find_nvss_first_sdss_counterparts() and get_flux_measurements_from_ned_table().
        """
        if self.nvss_id_simbad != "":
            flux_mask = self.ned_flux_table["Observed Passband"] == "1.4 GHz (NVSS)"
            if not flux_mask.any():
                log.error(f"{self.name} has NVSS ID {self.nvss_id_simbad} but no flux measurement in NED!")


    def search_x_ray_counterparts(self):
        """Search for X-ray counterparts in various catalogues -- MORX, 4XMM-DR14, CSC2.1, 2SXPS, BAT 157 Month Catalog.
        Make sure x_ray_catalogs = [morx[0], fourxmm, cxotwo[0], twosxps_swift[0], bat157]
        """
        
        #Getting XMM, CXO and Swift counterparts from MORX
        coords_morx = astropy.coordinates.SkyCoord(ra=self.x_ray_catalogs[0]['RAJ2000'],dec=self.x_ray_catalogs[0]['DEJ2000'],unit=(u.deg,u.deg))
        crossmatch_morx = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_morx,nthneighbor=1)
        self.morx_xmm = self.x_ray_catalogs[0][crossmatch_morx[0].item()]['XMM-ID']
        self.morx_cxo = self.x_ray_catalogs[0][crossmatch_morx[0].item()]['CX-ID']
        self.morx_swift = self.x_ray_catalogs[0][crossmatch_morx[0].item()]['Swift-ID']
        self.morx_lotss = self.x_ray_catalogs[0][crossmatch_morx[0].item()]['LoTSS-ID']
        self.morx_vlass = self.x_ray_catalogs[0][crossmatch_morx[0].item()]['VLASS-ID']
        self.morx_seperation = crossmatch_morx[1].item()
        #Sky extent of Longer Lobe in arcseconds
        self.lobe_extension = self.x_ray_catalogs[0][crossmatch_morx[0].item()]['Lobedist']*u.mas
        #4XMM-DR14 counterpart
        coords_4xmm = astropy.coordinates.SkyCoord(ra=self.x_ray_catalogs[1]['ra'],dec=self.x_ray_catalogs[1]['dec'],unit=(u.deg,u.deg))
        crossmatch_4xmm = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_4xmm,nthneighbor=1)
        self.xmm_id = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['iauname']
        self.xmm_flux4 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_ep_4_flux']
        self.xmm_flux4_err = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_ep_4_flux_err']
        self.xmm_flux5 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_ep_5_flux']
        self.xmm_flux5_err = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_ep_5_flux_err']
        self.xmm_hr3 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_hr3']
        self.xmm_hr3e = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_hr3_err']
        self.xmm_hr4 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_hr4']
        self.xmm_hr4e = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_hr4_err']
        self.xmm_variability_flag = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]['sc_var_flag']
        self.xmm_seperation = crossmatch_4xmm[1].item()
        #CSC2.1 counterpart
        coords_cxo = astropy.coordinates.SkyCoord(ra=self.x_ray_catalogs[2]['RAICRS'],dec=self.x_ray_catalogs[2]['DEICRS'],unit=(u.deg,u.deg))
        crossmatch_cxo = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_cxo,nthneighbor=1)
        self.cxo_id = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['2CXO']
        self.cxo_flux = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['FPL0.5-7']
        self.cxo_flux_lerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['b_FPL0.5-7']
        self.cxo_flux_uerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['B_FPL0.5-7']
        self.cxo_phoindex = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['GamPL']
        self.cxo_phoindex_lerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['b_GamPL']
        self.cxo_phoindex_uerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['B_GamPL']
        self.cxo_hr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['HRhm']
        self.cxo_hr_lerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['b_HRhm']
        self.cxo_hr_uerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['B_HRhm']
        self.cxo_var_flag = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]['fv']
        self.cxo_seperation = crossmatch_cxo[1].item()
        #2SXPS counterpart
        coords_swift = astropy.coordinates.SkyCoord(ra=self.x_ray_catalogs[3]['RAJ2000'],dec=self.x_ray_catalogs[3]['DEJ2000'],unit=(u.deg,u.deg))
        crossmatch_swift = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_swift,nthneighbor=1)
        self.swift_id = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['IAUName']
        self.swift_flux = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['FPCO0']
        self.swift_flux_lerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['e_FPCO0']
        self.swift_flux_uerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['E_FPCO0']
        self.swift_phoindex = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['Gamma']
        self.swift_phoindex_lerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['e_Gamma']
        self.swift_phoindex_uerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['E_Gamma']
        self.swift_hr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['HR2']
        self.swift_hr_lerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['e_HR2']
        self.swift_hr_uerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]['E_HR2']
        self.swift_seperation = crossmatch_swift[1].item()
        #BAT 157 Month Survey Catalog counterpart
        coords_bat = astropy.coordinates.SkyCoord(ra=self.x_ray_catalogs[4]['col3'],dec=self.x_ray_catalogs[4]['col4'],unit=(u.deg,u.deg))
        crossmatch_bat = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_bat,nthneighbor=1)
        self.bat_id = self.x_ray_catalogs[4][crossmatch_bat[0].item()]['col2']
        self.bat_seperation = crossmatch_bat[1].item()
        

    def search_gamma_ray_counterparts(self):
        """Search for gamma-ray counterparts in various catalogues -- Fermi 4FGL-DR4 and Fermi Transient 1FLT Catalog.
        Make sure gamma_ray_catalogs = [fermi_4fgl, fermi_transient]"""
        
        #Fermi 4FGL-DR4 counterpart
        coords_4fgl = astropy.coordinates.SkyCoord(ra=self.gamma_ray_catalogs[0]['RAJ2000'],dec=self.gamma_ray_catalogs[0]['DEJ2000'],unit=(u.deg,u.deg))
        crossmatch_4fgl = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_4fgl,nthneighbor=1)
        self.fgl_id = self.gamma_ray_catalogs[0][crossmatch_4fgl[0].item()]['Source_Name']
        self.fgl_seperation = crossmatch_4fgl[1].item()
        #Fermi Transient 1FLT counterpart
        coords_1flt = astropy.coordinates.SkyCoord(ra=self.gamma_ray_catalogs[1]['RAJ2000'],dec=self.gamma_ray_catalogs[1]['DEJ2000'],unit=(u.deg,u.deg))
        crossmatch_1flt = astropy.coordinates.match_coordinates_sky(self.source_coords, coords_1flt,nthneighbor=1)
        self.flt_id = self.gamma_ray_catalogs[1][crossmatch_1flt[0].item()]['Source_Name']
        self.flt_seperation = crossmatch_1flt[1].item()
       

    def __repr__(self):
        _string = f"""
            name: {self.name}\n
            ra: {self.ra:.2f}\n
            dec: {self.dec:.2f}\n
            sdss_id_simbad: {self.sdss_id_simbad}\n
            nvss_id_simbad: {self.nvss_id_simbad}\n
            first_id_simbad: {self.first_id_simbad}\n
            morx_xmm: {self.morx_xmm}
            morx_cxo: {self.morx_cxo}
            morx_swift: {self.morx_swift}
            morx_source_seperation: {self.morx_seperation}/{self.morx_seperation.to(u.arcmin)}/{self.morx_seperation.to(u.arcsec)}
            4xmm_id: {self.xmm_id}
            4xmm_source_seperation: {self.xmm_seperation}/{self.xmm_seperation.to(u.arcmin)}/{self.xmm_seperation.to(u.arcsec)}
            2cxo_id: {self.cxo_id}
            2cxo_source_seperation: {self.cxo_seperation}/{self.cxo_seperation.to(u.arcmin)}/{self.cxo_seperation.to(u.arcsec)}
            2sxps_swift_id: {self.swift_id}
            2sxps_swift_source_seperation: {self.swift_seperation}/{self.swift_seperation.to(u.arcmin)}/{self.swift_seperation.to(u.arcsec)}
            bat157month_id: {self.bat_id}
            bat157month_source_seperation: {self.bat_seperation}/{self.bat_seperation.to(u.arcmin)}/{self.bat_seperation.to(u.arcsec)}
            4fgl_id: {self.fgl_id}
            4fgl_source_seperation: {self.fgl_seperation}/{self.fgl_seperation.to(u.arcmin)}/{self.fgl_seperation.to(u.arcsec)}
            1flt_id: {self.flt_id}
            1flt_source_seperation: {self.flt_seperation}/{self.flt_seperation.to(u.arcmin)}/{self.flt_seperation.to(u.arcsec)}
        """
        return _string

    def flux_information(self):
        _string = f"""
            name: {self.name}\n
            4XMM DR14 (2-4.5 keV) flux: {self.xmm_flux4} +/- {self.xmm_flux4_err}\n
            4XMM DR14 (4.5-12 keV) flux: {self.xmm_flux5} +/- {self.xmm_flux5_err}\n
            4XMM DR14 HR (1-2 keV /2-4.5 keV): {self.xmm_hr3} +/- {self.xmm_hr3e}\n
            4XMM DR14 HR (2-4.5 keV /4.5-12 keV): {self.xmm_hr4} +/- {self.xmm_hr4e}\n
            2CXO FAPL (0.5-7 keV) Flux: {self.cxo_flux} +/- {0.5*(self.cxo_flux_uerr+self.cxo_flux_lerr)}
            2CXO Best Fit Photon Index: {self.cxo_phoindex} +/- {0.5*(self.cxo_phoindex_lerr+self.cxo_flux_uerr)}
            2CXO Hardness Ratio (Hard-Medium): {self.cxo_hr} +/- {0.5*(self.cxo_hr_lerr+self.cxo_hr_uerr)}
            2SXPS Flux (0.3-10 keV): {self.swift_flux} +/- {0.5*(self.swift_flux_uerr+self.swift_flux_lerr)}
            2SXPS Best Fit Photon Index: {self.swift_phoindex} +/- {0.5*(self.swift_phoindex_lerr+self.swift_phoindex_uerr)}
            2SXPS Hardness Ratio: {self.swift_hr} +/- {0.5*(self.swift_hr_lerr+self.swift_hr_uerr)}
        """
        return _string

    def write_catalogue_row(self, table):
        """Write the source information into a catalogue row."""
        table.add_row([
            self.name,
            self.ra,
            self.dec,
            self.source_type,
            self.morx_lotss,
            self.morx_vlass,
            self.nvss_id_simbad,
            self.first_id_simbad,
            self.sdss_id_simbad,
            self.morx_xmm,
            self.xmm_id,
            self.morx_cxo,
            self.cxo_id,
            self.morx_swift,
            self.swift_id,
            self.bat_id,
            self.fgl_id,
            self.flt_id,
            self.torresi_detection,
            self.xmm_variability_flag,
            self.cxo_var_flag,
            self.lobe_extension,
            self.distance,
            self.LogL_OIII,
            convert_F_nu_to_luminosity(1.4 * u.GHz, self.nvss_flux, u.mJy, self.distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, self.nvss_flux_error, u.mJy, self.distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, self.first_flux, u.mJy, self.distance),
            convert_F_nu_to_luminosity(1.4 * u.GHz, self.first_flux_error, u.mJy, self.distance),
            convert_flux_to_luminosity(self.xmm_flux4, self.distance),
            convert_flux_to_luminosity(self.xmm_flux4_err, self.distance),
            convert_flux_to_luminosity(self.xmm_flux5, self.distance),
            convert_flux_to_luminosity(self.xmm_flux5_err, self.distance),
            self.xmm_hr3,
            self.xmm_hr3e,
            self.xmm_hr4,
            self.xmm_hr4e,
            convert_flux_to_luminosity(self.cxo_flux, self.distance),
            convert_flux_to_luminosity(self.cxo_flux_lerr, self.distance),
            convert_flux_to_luminosity(self.cxo_flux_uerr, self.distance),
            self.cxo_phoindex,
            self.cxo_phoindex_lerr,
            self.cxo_phoindex_uerr,
            self.cxo_hr,
            self.cxo_hr_lerr,
            self.cxo_hr_uerr,
            convert_flux_to_luminosity(self.swift_flux, self.distance),
            convert_flux_to_luminosity(self.swift_flux_lerr, self.distance),
            convert_flux_to_luminosity(self.swift_flux_uerr, self.distance),
            self.swift_phoindex,
            self.swift_phoindex_lerr,
            self.swift_phoindex_uerr,
            self.swift_hr,
            self.swift_hr_lerr,
            self.swift_hr_uerr,
        ])
    
