import re
import logging
import astropy.units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from .catalogs import morx


# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)


def make_wildcard_name(name):
    """
    Generates a search pattern for Vizier by injecting wildcards (*).
    - SDSS / NVSS / FIRST Case: 'SDSSJ123...' -> 'SDSS*J123...'
    - Standard Case: 'NGC1275' -> 'NGC*1275'
    """
    # find a pattern of letters followed by numbers
    match = re.match(r"^([A-Za-z]+)(\d+.*)$", name)

    # if the name starts with SDSS, NVSS, or FIRST, we insert a * before the J
    if name.startswith("SDSS") or name.startswith("NVSS") or name.startswith("FIRST"):
        pieces = name.split("J")
        return f"{pieces[0].replace(' ', '')}*J{pieces[1]}"

    elif match:
        # Reconstruct as: Group1 (Letters) + * + Group2 (Numbers)
        return f"{match.group(1)}*{match.group(2)}"

    else:
        return name


def check_nvss_id_match_morx_simbad(source, row, catalog="NVSS"):
    """Checks that the NVSS or FIRST IDs from Simbad and MORX match."""
    attr_name = (
        f"{catalog.lower()}_id_simbad"  # get either nvss_id_simbad or first_id_simbad
    )
    source_simbad_id = getattr(source, attr_name)
    if source_simbad_id != "":
        # we normalize strings (remove spaces) for comparison to avoid formatting errors
        cat_id = str(row[f"{catalog}-ID"]).replace(" ", "")
        src_id = str(source_simbad_id).replace(" ", "")
        if src_id == cat_id:
            log.info(
                f"Verified {catalog}-ID matches! Simbad: {source_simbad_id} | MORX: {row[f'{catalog}-ID']}"
            )
        else:
            log.warning(
                f"Name matches, but {catalog}-ID differs! Simbad: {source_simbad_id} | MORX: {row[f'{catalog}-ID']}"
            )


def search_morx_counterpart(source, radius_cone_search=3 * u.arcmin):
    """Finds the MORX X-ray catalogue entry for a given source object following a strict hierarchy:
    1. name match
    2. NVSS-ID match
    3. FIRST-ID match
    4. cone search
    """
    # STEP 1: name Match
    query_name = morx.query_constraints(Name=make_wildcard_name(source.name))

    if len(query_name) == 1:
        row = query_name[0][0]  # take the first result
        log.info(f"{source.name} matched in MORX found by name!")
        # let us check if the NVSS and FIRST IDs match with Simbad's
        check_nvss_id_match_morx_simbad(source, row, catalog="NVSS")
        check_nvss_id_match_morx_simbad(source, row, catalog="FIRST")
        return row

    # STEP 2: NVSS-ID match
    if len(query_name) == 0 and source.nvss_id_simbad:
        log.info(
            f"No MORX counterpart found with source name {source.name}, trying with the NVSS ID..."
        )
        query_nvss = morx.query_constraints(
            **{"NVSS-ID": make_wildcard_name(source.nvss_id_simbad)}
        )
        if len(query_nvss) > 0:
            print(query_nvss)
            row = query_nvss[0][0]
            log.info(f"{source.name} matched in MORX by NVSS-ID: {row['NVSS-ID']}")
            return row

    # STEP 3: FIRST-ID match
    if len(query_name) == 0 and source.first_id_simbad:
        log.info(
            f"No MORX counterpart found with source name {source.name}, trying with FIRST ID..."
        )
        query_first = morx.query_constraints(
            **{"FIRST-ID": make_wildcard_name(source.first_id_simbad)}
        )
        if len(query_first) > 0:
            row = query_first[0][0]
            log.info(f"{source.name} matched in MORX by FIRST-ID: {row['FIRST-ID']}")
            return row

    # STEP 4: cone search
    log.info(
        f"No MORX counterpart found with NVSS or FIRST IDs, performing a cone search."
    )
    cone_query = morx.query_region(source.coords, radius=radius_cone_search)
    # only one table in morx catalouge
    table = cone_query[0]
    # add a table column with angular separations
    if len(cone_query) > 0:
        separations = source.coords.separation(
            SkyCoord(ra=table["RAJ2000"], dec=table["DEJ2000"], unit=(u.deg, u.deg))
        )
        table.add_column(separations, name="Separation")
        return table
    else:
        log.info(
            f"No MORX counterpart found within {radius_cone_search} of source {source.name} at coordinates {source.coords.to_string('hmsdms')}."
        )
        return None
