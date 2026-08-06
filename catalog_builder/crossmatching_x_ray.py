import re
import logging
import numpy as np
import astropy.units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from .flux_utils import convert_x_ray_flux_to_luminosity
from .catalogs import morx, csc1, csc2, csc_acis, _4xmm


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
        if len(pieces) == 1:
            log.warning(f"couldn't split the name {name} at the J")
            return name
        else:
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
    # STEP 1: name match
    query_name = morx.query_constraints(Name=make_wildcard_name(source.name))

    if len(query_name) == 1:
        row = query_name[0][0]  # take the first result
        log.info(f"{source.name} matched in MORX by name!")
        # let us check if the NVSS and FIRST IDs match with Simbad's
        check_nvss_id_match_morx_simbad(source, row, catalog="NVSS")
        check_nvss_id_match_morx_simbad(source, row, catalog="FIRST")
        return row

    # STEP 2: NVSS-ID match
    if len(query_name) == 0 and source.nvss_id_simbad:
        log.info(
            f"No MORX counterpart found with source name {source.name}, trying with the NVSS ID {source.nvss_id_simbad}"
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
            f"No MORX counterpart found with source name {source.name}, trying with FIRST ID {source.first_id_simbad}"
        )
        query_first = morx.query_constraints(
            **{"FIRST-ID": make_wildcard_name(source.first_id_simbad)}
        )
        if len(query_first) > 0:
            row = query_first[0][0]
            log.info(f"{source.name} matched in MORX by FIRST-ID: {row['FIRST-ID']}")
            return row

    # STEP 4: cone search
    # LEAVE THE CONE SEARCH BE FOR THE TIME BEING. TODO: FIX
    log.info(
        f"No MORX counterpart found with NVSS or FIRST IDs, performing a cone search."
    )
    return None
    """
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
    """


def fetch_xmm_photometry(morx_row):
    """Parses the MORX matches and queries the XMM catalogues.
    Modified to work directly with the morx_row rather than with the source"""
    # there are four XMM catalogues in the MORX, according to the ID prefix:
    # 2XMM/2XMMi: XMM-Newton DR3, https://cdsarc.cds.unistra.fr/cat/?IX/41
    # 4XMM: XMM-Newton DR13, https://www.cosmos.esa.int/web/xmm-newton/xsa
    # XMMSL: XMM-Newton Slew Survey Release 2.0, same attribution as 4XMM
    # XMMX: XAssist XMM-Newton, https://asd.gsfc.nasa.gov/xassist/pipeline5/xmm/
    xmm_id = morx_row["XMM-ID"]

    if xmm_id and str(xmm_id) != "--":
        prefix_id = str(xmm_id).split(" ")[0]
        source_id = str(xmm_id).split(" ")[1]

        if prefix_id != "4XMM":
            return None
            # TODO: fix the following part
            # this is a source from an older catalogue (2XMM) or different survey (Slew/XAssist)
            # search for an updated counterpart in the modern 4XMM catalog
            """
            updated_id = search_updated_x_ray_catalogue(xmm_id, _4xmm, "4XMM")

            if updated_id is None:
                # fall-back strategy, the source is not in the new 4XMM catalogue
                log.info(
                    f"{xmm_id} does not have an updated counterpart in 4XMM, NOT falling back to original catalogues"
                )
            else:
                log.info(f"{xmm_id} updated to 4XMM {updated_id} in 4XMM catalog")
                res = _4xmm.query_constraints(**{"4XMM": updated_id})
                return res
            """
        else:
            # this is a source already with a 4XMM counterpart
            log.info(f"querying for {source_id} in 4XMM")
            res = _4xmm.query_constraints(**{"4XMM": source_id})
            return res
    else:
        log.info(f"No XMM ID in MORX for this source.")
        return None


def get_4xmm_luminosities(morx_row, distance):
    """For now we fetch the Flux in the 'soft' and 'hard' band.
    From a note in the catalogue
    1 =	0.2 - 0.5 keV (narrow band)
    2 =	0.5 - 1.0 keV (narrow band)
    3 =	1.0 - 2.0 keV (narrow band)
    4 =	2.0 - 4.5 keV (narrow band)
    5 =	4.5 - 12.0 keV (narrow band)
    6 =	0.2 - 2.0 keV = soft broad band, no images made
    7 =	2.0 - 12.0 keV = hard broad band, no images made
    8 =	0.2 - 12.0 keV = total band
    9 =	0.5 - 4.5 keV = XID band
    """
    _table = fetch_xmm_photometry(morx_row)
    if _table is None:
        return np.nan, np.nan, np.nan, np.nan
    else:
        Flux1 = _table[0]["Flux1"][0]
        e_Flux1 = _table[0]["e_Flux1"][0]
        Flux2 = _table[0]["Flux2"][0]
        e_Flux2 = _table[0]["e_Flux2"][0]
        Flux3 = _table[0]["Flux3"][0]
        e_Flux3 = _table[0]["e_Flux3"][0]
        Flux4 = _table[0]["Flux4"][0]
        e_Flux4 = _table[0]["e_Flux4"][0]
        Flux5 = _table[0]["Flux5"][0]
        e_Flux5 = _table[0]["e_Flux5"][0]

        flux_soft = Flux1 + Flux2 + Flux3
        e_flux_soft = np.sqrt(e_Flux1**2 + e_Flux2**2 + e_Flux3**2)
        flux_hard = Flux4 + Flux5
        e_flux_hard = np.sqrt(e_Flux4**2 + e_Flux5**2)

        # convert fluxes in nW / m^2 to erg s-1
        l_soft = convert_x_ray_flux_to_luminosity(flux_soft, distance)
        l_soft_err = convert_x_ray_flux_to_luminosity(e_flux_soft, distance)
        l_hard = convert_x_ray_flux_to_luminosity(flux_hard, distance)
        l_hard_err = convert_x_ray_flux_to_luminosity(e_flux_hard, distance)

        return l_soft, l_soft_err, l_hard, l_hard_err


#########################################################
# TODO: starting from here functions need to be improved
#########################################################


def search_updated_x_ray_catalogue(old_id, new_catalog, new_prefix_id):
    """Find the updated version of a source in a most recent X-ray catalog.
    For example the counterpart for a CSC 1.1 or Wang+ 2016 source in the CSC 2.0.
    Or the updated counterpart of a 2XMM source in the 4XMM.

    Search by string coordinates first (e.g. J123456.7+123456),
    if failing, perform a proper cone search with a small radius.

    Parameters
    ----------
    old_id : str
        the id from the outdated catalog
    new_catalog : `~astroquery.vizier.Vizier`
        the updated catalogue
    new_prefix_id : str
        the prefix of the ID in the updated catalogue

    Returns
    -------
        The updated row in the new catalogue
    """
    # strip everything before the "J" to get the coordinate part
    coord_str = old_id[old_id.find("J") :]
    res = new_catalog.query_constraints(**{new_prefix_id: coord_str})
    if len(res) == 1:
        return res[0][0][new_prefix_id]
    else:
        # the check by coordinate string failed, perform and actual cone search
        # SkyCoord can parse "J" strings (e.g., "J123456.7+123456")
        coords = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
        res = new_catalog.query_region(coords, radius=2 * u.arcsec)
        if len(res) == 0:
            log.info(f"{old_id} could not be updated to an {new_prefix_id} source")
        if len(res) == 1:
            return res[0][0]["2CXO"]
        if (len(res)) > 1:
            log.warning(
                f"more than one counterpart within 2 arcsec of {old_id}... please check this source!"
            )


def fetch_chandra_photometry(source):
    """Parses the MORX matches and queries the Chandra catalouges."""
    # get the MORX data (lazy load)
    morx_row = source.morx_xmatch_table
    if morx_row is None:
        print(f"No MORX counterpart for {source.name}. Cannot fetch photometry.")
        return None

    # there are four types of Chandra catalgoues in the MORX, according to the ID prefix:
    # CXOG: Chandra ACIS source catalog, Wang S. et al., 2016,ApJS,224,40
    # CXO:  Chandra Source Catalog v1.1, https://asc.harvard.edu/csc1/
    # 2CXO: Chandra Source Catalog v2.0, https://asc.harvard.edu/csc2/
    # CXOX: XAssist Chandra, https://asd.gsfc.nasa.gov/xassist/pipeline4/chandra/
    cx_id = morx_row["CX-ID"]

    if cx_id and str(cx_id) != "--":
        prefix_id = str(cx_id).split(" ")[0]
        source_id = str(cx_id).split(" ")[1]

        if prefix_id != "2CXO":
            # this is a source from an older catalouge CSC1 or Wang 2016
            csc2_id = search_updated_x_ray_catalogue(cx_id, csc2, "2CXO")
            if csc2_id is None:
                # fall-back strategy, the source was in CSC1 or Wang 2016
                # but it's not in the new catalogues, let us search
                log.info(
                    f"{cx_id} does not have an updated counterpart in the CSC2, falling back to original catalogues"
                )
                if prefix_id == "CXO":
                    log.info(f"querying {source_id} in CSC1")
                    res = csc_acis.query_constraints(**{"CXOGSG": source_id})
                    return res
                if prefix_id == "CXOG":
                    log.info(f"querying {source_id} in CSC ACIS")
                    res = csc_acis.query_constraints(**{"CXOGSG": source_id})
                    return res
            else:
                log.info(f"{cx_id} updated to 2CXO {csc2_id} in CSC2")
                res = csc2.query_constraints(**{"2CXO": csc2_id})
                return res
        else:
            # this is a source already with a CSC2 counterparts
            log.info(f"querying for {source_id} in CSC 2")
            res = csc2.query_constraints(**{"2CXO": source_id})
            return res
    else:
        log.info(f"No Chandra ID in MORX for {source.name}")
