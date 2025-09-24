# util functions to get source IDs and fluxes from different catalogues
import logging
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


log = logging.getLogger(__name__)


def get_source_identifier(source_name, start_id):
    identifiers = Simbad.query_objectids(source_name)
    mask = [string.startswith(start_id) for string in identifiers["id"]]
    ids = identifiers["id"][mask]
    if len(ids) == 0:
        log.info(f"{start_id} counterpart not available for {source_name}")
        return ""
    else:
        log.info(f"{source_name} matched with {ids[0]} by SIMBAD")
        return ids[0]

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


def get_source_simbad_coordinates(source_name):
    # fetch the Simbad coordinates
    simbad_query = Simbad.query_object(source_name)
    return convert_ra_dec_spaced_string(simbad_query["RA"][0], simbad_query["DEC"][0])


def get_simbad_redshift(source_name):
    """Obtain the redshift of a source from Simbad"""
    Simbad.add_votable_fields("z_value")
    simbad_query = Simbad.query_object(source_name)
    if simbad_query:
        return simbad_query["Z_VALUE"][0]
    else:
        return -1


def convert_F_nu_to_luminosity(nu, F_nu, F_nu_unit, distance):
    """Froma F_nu, returns a luminosity in erg s-1.
    F_nu is assumed to be read as a float from some catalogue"""
    surface = 4 * np.pi * distance**2
    L = nu * F_nu * F_nu_unit * surface
    return L.to_value("erg s-1")


def get_source_id_and_flux_from_catalog(
    source_name,
    catalog,
    id_colname,
    flux_colname,
    flux_err_colname,
    flux_unit,
    ra_colname,
    dec_colname,
    search_radius,
):
    """Given a catalouge, get the counterpart to a source name with its flux values.
    In case multiple sources are found as counterparts, the attributes of the
    closest one will be returned.

    Parameters
    ----------
    source_name : string
        name of the source (the coordinates associated to this name are resolved)
    catalog : string
        ID of the catalogue in which the source's counterparts have to be searched
    id_colname : string
        name of the column with the ID of the sources in the catalogue
    flux_colname : string
        name of the column with the flux values
    flux_err_colname : string
        name of the column with the flux values
    flux_unit : `~astropy.unit.Unit`
        unit used to express the flux in the catalouge
    ra_colname : string
        name of the column with the RA values
    dec_colname : string
        name of the column with the DEC values
    search_radius : `~astropy.unit.Quantity`
        radius of search around the source coordinates
    """

    columns_to_load = [
        id_colname,
        flux_colname,
        flux_err_colname,
        ra_colname,
        dec_colname,
    ]

    vizier = Vizier(columns=columns_to_load)
    table_match = vizier.query_object(
        source_name, catalog=catalog, radius=search_radius
    )

    # no match found
    if len(table_match) == 0:
        log.info(f"{source_name} not found in {catalog}")
        return "", 0 * flux_unit, 0 * flux_unit

    else:
        log.info(
            f"{len(table_match)} objects found in {catalog} within {search_radius} of {source_name}"
        )

        table_match = table_match[catalog]

        # sometimes the Column name used to fetch from Vizier does not match
        # the one in the catalogue
        try:
            table_match[id_colname]
        except KeyError:
            id_colname = "_" + id_colname

        # one possible counterpart found
        if len(table_match) == 1:
            return [
                table_match[id_colname][0],
                table_match[flux_colname][0] * flux_unit,
                table_match[flux_err_colname][0] * flux_unit,
            ]

        # more than one possible counterpart found
        else:
            # take the match closer to the source nominal position
            # use the source coordinate from SIMBAD
            source_coord = get_source_simbad_coordinates(source_name)

            if table_match[ra_colname].dtype == np.float64:
                coords = SkyCoord(table_match[ra_colname], table_match[dec_colname])
            else:
                coords = convert_ra_dec_spaced_string(
                    list(table_match[ra_colname]), list(table_match[dec_colname])
                )

            closer_idx = source_coord.separation(coords).argmin()
            row = table_match[closer_idx]

            return [
                row[id_colname],
                row[flux_colname] * flux_unit,
                row[flux_err_colname] * flux_unit,
            ]
