# structure of the table for the final catalgoue
import logging
import numpy as np
import astropy.units as u
from astropy.table import vstack, Column
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned

# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)


def get_source_survey_identifier(source_name, survey_id):
    """Get the source identifier from SIMBAD starting with a given string
    e.g. "NVSS", "FIRST, "SDSS", "4FGL", etc."""
    identifiers = Simbad.query_objectids(source_name)
    mask = [string.startswith(survey_id) for string in identifiers["id"]]
    ids = identifiers["id"][mask]
    if len(ids) == 0:
        log.info(f"{survey_id} counterpart not available for {source_name}")
        return ""
    elif len(ids) == 1:
        log.info(f"{source_name} matched with {ids[0]} by SIMBAD")
        return ids[0]
    else:
        log.warning(
            f"{len(ids)} {survey_id} counterparts found for {source_name} by SIMBAD. Taking the first one: {ids[0]}"
        )
        log.warning(
            f"full list of counterparts: {ids.data.data} please check on SIMBAD!"
        )
        return ids[0]


def get_simbad_coordinates(source_name):
    """Get the coordinates of the source from SIMBAD"""
    simbad_query = Simbad.query_object(source_name)
    if simbad_query:
        ra = simbad_query["ra"][0]
        dec = simbad_query["dec"][0]
        return ra, dec
    else:
        log.error(f"Source {source_name} not found in SIMBAD")
        return None, None


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
            is_ul.append(True)
            uncertainties.append(float("nan"))
            if not unit.startswith("log"):
                values.append(float(uncertainty))
                units.append(unit)
            if unit.startswith("log"):
                values.append(10 ** float(uncertainty.replace("<", "")))
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
        self.find_nvss_first_sdss_counterparts()
        # let us load all the photometric measurements, but let us filter only
        # those of interest to us (e.g. optical lines and radio fluxes at 15 GHz)
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

    def find_nvss_first_sdss_counterparts(self):
        """Find the NVSS, FIRST, and SDSS identifiers.
        In principle all the sources should be in these deep surveys.
        """
        self.sdss_id_simbad = get_source_survey_identifier(self.name, "SDSS")
        self.nvss_id_simbad = get_source_survey_identifier(self.name, "NVSS")
        self.first_id_simbad = get_source_survey_identifier(self.name, "FIRST")

    def write_catalogue_row(cls, row):
        """Write the source information into a catalogue row."""
        pass

    def search_x_ray_counterparts(self):
        """Search for X-ray counterparts in various catalogues."""
        

    def __repr__(self):
        _string = f"""
            name: {self.name}\n
            ra: {self.ra:.2f}\n
            dec: {self.dec:.2f}\n
            sdss_id_simbad: {self.sdss_id_simbad}\n
            nvss_id_simbad: {self.nvss_id_simbad}\n
            first_id_simbad: {self.first_id_simbad}\n
        """
        return _string
