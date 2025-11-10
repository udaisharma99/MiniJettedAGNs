# structure of the table for the final catalgoue
import logging
import numpy as np
import astropy.units as u
from astropy.table import Column
from astropy.coordinates import Distance, SkyCoord, match_coordinates_sky
from astroquery.ned import Ned
from astropy.table import vstack
from .catalogues import nvss, first
from .utils import (
    get_sky_coordinates_simbad,
    get_redshift_simbad,
    get_source_survey_identifier,
    get_source_type_simbad,
    get_flux_measurements_from_ned_table,
    compile_radio_sed
)

# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)


class Source:
    """
    Class to represent a single source in the catalogue.
    It will contain basic information like coordinates, name, and type.
    It will also contain basic methods to find FIRST, NVSS and SDSS names.
    We will create another structure to hold the X-ray information.
    """

    def __init__(self, name):
        self.name = name
        self.coords = get_sky_coordinates_simbad(self.name)
        self.z = get_redshift_simbad(self.name)
        self.d_L = Distance(z=self.z).to("Mpc")
        self.source_type_simbad = get_source_type_simbad(self.name)
        # already at initialisation, we find the NVSS, FIRST and SDSS counterparts
        # in principle all the sources should be in these deep surveys
        self.find_nvss_first_sdss_counterparts()
        # self.torresi_detection = self.sdss_id_simbad in torresi_sources
        # get lines and radio fluxes measurements
        self.get_ned_lines_fluxes()
        self.get_radio_fluxes()
        self.radio_sed = compile_radio_sed(self.radio_flux_table_ned)
        """
        source_coreG = nagar_2005[0]["Name"] == self.name
        if np.any(source_coreG):
            self.distance = nagar_2005[0]["Dist"][source_coreG][0] * u.Mpc
            self.source_type = nagar_2005[0]["AType"][source_coreG][0]
        else:
            fr0_match = self.name == fr0cat[0]["SimbadName"]
            redshift = fr0cat[0]["z"][fr0_match][0]
            self.distance = Distance(z=redshift).to("Mpc")
            self.source_type = "FR0"
        """


    def get_ned_lines_fluxes(self):
        """Get the luminosities of the optical lines from the NED photometry table."""
        log.info(f"Searching NED line fluxes for source {self.name}")
        # let us load all the photometric measurements, but let us filter only
        # those of interest to us (e.g. optical lines and radio fluxes at 15 GHz)
        ned_table = Ned.get_table(self.name, table="photometry")
        lines_list = [
            "H{alpha}",
            "H{beta}",
            "[O III] 5007",
            "[O I] 6300",
            "[S II]",
        ]
        lines_flux_tables = [
            get_flux_measurements_from_ned_table(ned_table, band) for band in lines_list
        ]
        self.lines_flux_table_ned = vstack(lines_flux_tables)

    def get_radio_fluxes(self):
        """Get the radio flux measurements from the NED photometry table."""
        log.info(f"Searching NED radio fluxes for source {self.name}")
        ned_table = Ned.get_table(self.name, table="photometry")
        # search in the NED table, every band indicated by Hz or mm
        hz_mask = np.asarray(["Hz" in _ for _ in ned_table["Observed Passband"]])
        mm_mask = np.asarray([" mm" in _ for _ in ned_table["Observed Passband"]])
        # note the space before mm is due to telescopes containing 'mm' in their names (e.g. "M. Lemmon")
        radio_bands = ned_table["Observed Passband"][hz_mask | mm_mask]
        radio_flux_tables = [
            get_flux_measurements_from_ned_table(ned_table, band)
            for band in radio_bands
        ]
        self.radio_flux_table_ned = vstack(radio_flux_tables)

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

    def __repr__(self):
        _string = f"""
            name : {self.name}\n
            ra : {self.ra:.2f}\n
            dec : {self.dec:.2f}\n
            z : {self.z:.3f}\n
            source_type: {self.source_type}\n
            sdss_id_simbad: {self.sdss_id_simbad}\n
            nvss_id_simbad: {self.nvss_id_simbad}\n
            first_id_simbad: {self.first_id_simbad}\n
        """
        return _string
