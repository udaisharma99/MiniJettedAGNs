# utils functions to to handle catalgoues information and interact with SIMBADs
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad


# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)


# add other fields to SIMBAD queries
Simbad.add_votable_fields("ids")
Simbad.add_votable_fields("rvz_redshift")
Simbad.add_votable_fields("mesotype")


def insert_space_source_ids(source_name):
    """In Nagar et al. (2005) sources ID are reported without a space
    e.g. 'NGC1275', in Ho et al. (1997) there is a space 'NGC1275'."""
    if source_name.startswith("IC"):
        source_name = source_name.strip("IC")
        source_name = "IC " + source_name
    if source_name.startswith("NGC"):
        source_name = source_name.strip("NGC")
        source_name = "NGC " + source_name
    if source_name.startswith("UGC"):
        source_name = source_name.strip("UGC")
        source_name = "UGC " + source_name
    if source_name.startswith("LSXPS"):
        source_name = source_name.strip("LSXPS")
        source_name = "LSXPS " + source_name
    return source_name


def get_sky_coordinates_simbad(source_name):
    """Get the coordinates of the source from SIMBAD"""
    simbad_query = Simbad.query_object(source_name)
    if simbad_query:
        ra = simbad_query["ra"][0]
        dec = simbad_query["dec"][0]
        return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    else:
        log.error(f"Source {source_name} not found in SIMBAD")
        return None


def convert_ra_dec_spaced_string(ra_string, dec_string):
    """Convert RA and DEC expressed as strings with a space between
    hours (degree) minute and second - e.g. RA = 1 03 45.34. - into
    `~astropy.SkyCoord`"""
    if isinstance(ra_string, list) and isinstance(dec_string, list):
        ra = [_.replace(" ", "h", 1).replace(" ", "m", 1) + "s" for _ in ra_string]
        dec = [_.replace(" ", "d", 1).replace(" ", "m", 1) + "s" for _ in dec_string]
    else:
        ra = ra_string.replace(" ", "h", 1).replace(" ", "m", 1) + "s"
        dec = dec_string.replace(" ", "d", 1).replace(" ", "m", 1) + "s"
    return SkyCoord(ra, dec, frame="icrs")


def get_redshift_simbad(source_name):
    """Obtain the redshift of a source from Simbad"""
    simbad_query = Simbad.query_object(source_name)
    if simbad_query:
        z = simbad_query["rvz_redshift"][0]
        return z if z > 0 else -1
    else:
        return -1


def get_source_type_simbad(source_name):
    """Obtain the source classification according to SIMBAD"""
    simbad_query = Simbad.query_object(source_name)
    if simbad_query:
        return simbad_query["mesotype.otype"][0]
    else:
        return ""


def get_source_survey_identifier(source_name, survey_id):
    """Get the source identifier from SIMBAD starting with a given string
    e.g. "NVSS", "FIRST, "SDSS", "4FGL", etc."""
    identifiers = Simbad.query_objectids(source_name)
    mask = [string.startswith(survey_id) for string in identifiers["id"]]
    ids = identifiers["id"][mask]
    if len(ids) == 0:
        # log.info(f"{survey_id} counterpart not available for {source_name}")
        return ""
    elif len(ids) == 1:
        # log.info(f"{source_name} matched with {ids[0]} by SIMBAD")
        return ids[0]
    else:
        log.warning(
            f"{len(ids)} {survey_id} counterparts found for {source_name} by SIMBAD. Taking the first one: {ids[0]}"
        )
        log.warning(
            f"full list of counterparts: {ids.data.data} please check on SIMBAD!"
        )
        return ids[0]
