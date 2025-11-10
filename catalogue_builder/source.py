# structure of the table for the final catalgoue
import logging
import numpy as np
from astropy.coordinates import Distance
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
import IPython

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

    def find_nvss_first_sdss_counterparts(self):
        """Find the NVSS, FIRST, and SDSS identifiers.
        In principle all the sources should be in these deep surveys.
        """
        # first search through SIMBAD
        self.sdss_id_simbad = get_source_survey_identifier(self.name, "SDSS")
        # search the NVSS name
        self.nvss_id_simbad = get_source_survey_identifier(self.name, "NVSS")
        if self.nvss_id_simbad != "":
            nvss_match = nvss.query_object(self.nvss_id_simbad)
            if len(nvss_match) > 0:
                self.nvss_flux = nvss_match[0]["S1.4"]
                self.nvss_flux_error = nvss_match[0]["e_S1.4"]
        else:
            self.nvss_flux = 0
            self.nvss_flux_error = 0
        # search the FIRST name
        self.first_id_simbad = get_source_survey_identifier(self.name, "FIRST")
        if self.first_id_simbad != "":
            first_match = first.query_object(self.first_id_simbad)
            if len(first_match) > 0:
                self.first_flux = first_match[0]["Fint"]
                self.first_flux_error = first_match[0]["Rms"]
        else:
            self.first_flux = 0
            self.first_flux_error = 0

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

    def __repr__(self):
        _string = f"""
            name : {self.name}
            ra : {self.coords.ra:.2f}
            dec : {self.coords.dec:.2f}
            z : {self.z:.3f}
            source_type: {self.source_type_simbad}
            sdss_id_simbad: {self.sdss_id_simbad}
            nvss_id_simbad: {self.nvss_id_simbad}
            first_id_simbad: {self.first_id_simbad}
        """
        return _string
