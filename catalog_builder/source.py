# structure of the table for the final catalgoue
import logging
from functools import cached_property
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astroquery.ned import Ned
from astropy.table import vstack
import matplotlib.pyplot as plt
from .catalog_utils import (
    get_sky_coordinates_simbad,
    get_redshift_simbad,
    get_source_survey_identifier,
    get_source_type_simbad,
)
from .flux_utils import (
    get_flux_measurements_from_ned_table,
    compile_radio_sed,
    _plot_radio_sed,
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

    def find_nvss_first_sdss_counterparts(self):
        """Find the NVSS, FIRST, and SDSS identifiers.
        In principle all the sources should be in these deep surveys.
        """
        # first search through SIMBAD
        self.sdss_id_simbad = get_source_survey_identifier(self.name, "SDSS")
        # search the NVSS name
        self.nvss_id_simbad = get_source_survey_identifier(self.name, "NVSS")
        # search the FIRST name
        self.first_id_simbad = get_source_survey_identifier(self.name, "FIRST")

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

    def _fetch_lines_fluxes_ned(self):
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
        return vstack(lines_flux_tables)

    def _fetch_radio_fluxes_ned(self):
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
        return vstack(radio_flux_tables)

    @cached_property
    def lines_flux_table(self):
        """Get the optical line fluxes from NED."""
        return self._fetch_lines_fluxes_ned()

    @cached_property
    def radio_flux_table(self):
        """Get the radio flux measurements from NED."""
        return self._fetch_radio_fluxes_ned()

    @cached_property
    def radio_sed(self):
        # accessing self.radio_flux_table triggers its own cached_property logic
        return compile_radio_sed(self.radio_flux_table)

    def plot_radio_sed(self, ax=None):
        """Plot the radio SED, both original and binned."""
        if ax is None:
            ax = plt.gca()

        if self.radio_sed is None:
            log.warning(
                f"No radio SED available for source {self.name}, cannot plot it."
            )
            return
        else:
            _plot_radio_sed(self.radio_sed, ax=ax)

    @cached_property
    def morx_xmatch_table(self):
        """Returns the MORX counterpart table.
        Lazy loads the logic from the crossmatching module on first access.
        """
        # IMPORT HERE, not at the top of the file.
        # this prevents circular dependency issues.
        from .crossmatching import search_morx_counterpart

        # call the external function passing 'self'
        return search_morx_counterpart(self, radius_cone_search=3 * u.arcmin)
