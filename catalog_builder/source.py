# structure of the table for the final catalgoue
import logging
from functools import cached_property
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astroquery.ned import Ned
from astropy.table import vstack
import matplotlib.pyplot as plt
from .catalogs import nvss
from .catalog_utils import (
    get_sky_coordinates_simbad,
    get_redshift_simbad,
    get_source_survey_identifier,
    get_source_type_simbad,
)
from .flux_utils import (
    convert_F_nu_to_luminosity,
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

    def get_L_nvss(self):
        """Get the NVSS luminosity."""
        if self.nvss_id_simbad == "" or "B" in self.nvss_id_simbad:
            # source has no NVSS ID, or some strange B in their name
            # e.g. NVSS B121650+060612
            return np.nan, np.nan
        else:
            # sources in the NVSS catalogue don't have the NVSS J prefix as in Simbad
            stripped_nvss_id = self.nvss_id_simbad.strip("NVSS J")
            _table = nvss.query_constraints(NVSS=stripped_nvss_id)
            if len(_table) > 0:
                L_nvss = convert_F_nu_to_luminosity(
                    1.4 * u.GHz, _table[0]["S1.4"][0], u.mJy, self.d_L
                )
                L_nvss_err = convert_F_nu_to_luminosity(
                    1.4 * u.GHz, _table[0]["e_S1.4"][0], u.mJy, self.d_L
                )
                return L_nvss, L_nvss_err
            else:
                log.warning(
                    f"NVSS J {stripped_nvss_id} not found in NVSS catalogue by Vizier."
                )
                return np.nan, np.nan

    def get_L_first(self):
        """TODO: add this function."""
        pass

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
        if len(radio_flux_tables) > 0:
            return vstack(radio_flux_tables)
        else:
            return None

    def _fetch_x_ray_fluxes_ned(self):
        """Get the X-ray flux measurements from the NED photometry table."""
        log.info(f"Searching NED X-ray fluxes for source {self.name}")
        ned_table = Ned.get_table(self.name, table="photometry")
        # search in the NED table, every band indicated by keV
        xray_mask = np.asarray(["keV" in _ for _ in ned_table["Observed Passband"]])
        xray_bands = ned_table["Observed Passband"][xray_mask]
        xray_flux_tables = [
            get_flux_measurements_from_ned_table(ned_table, band) for band in xray_bands
        ]
        return vstack(xray_flux_tables)

    @cached_property
    def x_ray_flux_table(self):
        """Get the X-ray flux measurements from NED."""
        return self._fetch_x_ray_fluxes_ned()

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
        if self.radio_flux_table is not None:
            return compile_radio_sed(self.radio_flux_table)
        else:
            return None

    def plot_radio_sed(self, fit, ax=None):
        """Plot the radio SED, both original and binned."""
        if ax is None:
            ax = plt.gca()

        if self.radio_sed is None:
            log.warning(
                f"No radio SED available for source {self.name}, cannot plot it."
            )
            return
        else:
            _plot_radio_sed(self.radio_sed, fit, ax=ax)

    @cached_property
    def morx_xmatch_table(self):
        """Returns the MORX counterpart table.
        Lazy loads the logic from the crossmatching module on first access.
        """
        # IMPORT HERE, not at the top of the file.
        # this prevents circular dependency issues.
        from .crossmatching_x_ray import search_morx_counterpart

        # call the external function passing 'self'
        return search_morx_counterpart(self, radius_cone_search=3 * u.arcmin)
