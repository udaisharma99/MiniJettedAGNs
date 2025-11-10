# util functions to get source IDs and fluxes from different catalogues
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


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


# --------------------------------------------------------------
# functions interacting with SIMBAD (coordinates, redshift, IDs)
# --------------------------------------------------------------
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


def get_redshift_simbad(source_name):
    """Obtain the redshift of a source from Simbad"""
    simbad_query = Simbad.query_object(source_name)
    if simbad_query:
        return simbad_query["rvz_redshift"][0]
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


# --------------------------------------------------------------
# functions handling fluxes conversions
# --------------------------------------------------------------
def convert_flux_to_luminosity(flux, distance):

    flux = flux * u.mW/u.m**2
    D_L = distance.to(u.cm) # Convert Mpc to cm for consistent units
    flux_in_cgs = flux.to(u.erg / (u.s * u.cm**2))
    luminosity = 4 * np.pi * D_L**2 * flux_in_cgs
    
    return luminosity.to(u.erg / u.s)


def convert_F_nu_to_luminosity(nu, F_nu, F_nu_unit, distance):
    """Froma F_nu, returns a luminosity in erg s-1.
    F_nu is assumed to be read as a float from some catalogue"""
    surface = 4 * np.pi * distance**2
    L = nu * F_nu * F_nu_unit * surface
    return L.to_value("erg s-1")


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

    for band, value, uncertainty, unit in zip(
        ned_table["Observed Passband"],
        ned_table["Photometry Measurement"],
        ned_table["Uncertainty"],
        ned_table["Units"],
    ):
        # strip possible problematic characters
        unit = unit.replace("^", "").replace("(", "").replace(")", "")
        # now let us go case by case:
        # - case 0: no uncertainty provided
        if uncertainty == "" or uncertainty == "+/-":
            is_ul.append(False)
            uncertainties.append(float("nan"))
            if unit.startswith("log"):
                values.append(10 ** float(value))
                units.append(unit.replace("log", ""))
            else:
                values.append(value)
                units.append(unit)
        # - case 1.1: this is an upper limit
        elif uncertainty.startswith("<"):
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
            value = 10 ** float(value)
            values.append(value)
            # check if the uncertainty is percentual
            uncertainty = uncertainty.replace("+/-", "")
            if uncertainty.endswith("%"):
                perc = float(uncertainty.replace("%", ""))
                uncertainties.append(value * perc / 100)
            else:
                uncertainties.append(np.log(10) * value * float(uncertainty))
            units.append(unit.replace("log", ""))

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


def compile_radio_sed(ned_radio_flux_table):
    """Compile the radio SED from the NED radio flux measurements 
    obtained from the `reformat_ned_table` function.
    
    """
    nu = np.array([]) * u.GHz
    Fnu = np.array([]) * u.Jy
    Fnu_err = np.array([]) * u.Jy

    for band, value, uncertainty, is_ul, unit in zip(
        ned_radio_flux_table["Observed Passband"],
        ned_radio_flux_table["flux"],
        ned_radio_flux_table["flux_err"],
        ned_radio_flux_table["is_ul"],
        ned_radio_flux_table["unit"],
    ):
        # for this SED, we will consider only measurements with uncertainities.
        if np.isnan(uncertainty) or is_ul:
            continue
        else:
            # read the band, and eliminate anything that is not a number or a unit, e.g.
            # 347 GHz (SCUBA)
            # 1 mm (IRAM 30m)
            # in principle should be enough to remove anything following the unit
            partitioner = "Hz" if "Hz" in band else "mm"
            trailing_term = band.partition(partitioner)[-1] # this is the telescope info e.g. "(NVSS)"
            _nu = u.Quantity(band.replace(trailing_term, ""))
            # convert to frequency in Hz
            _nu.to("Hz", equivalencies=u.spectral())
            nu = np.append(nu, _nu)

            _Fnu = (value * u.Unit(unit))
            Fnu = np.append(Fnu, _Fnu)
            _Fnu_err = (uncertainty * u.Unit(unit))
            Fnu_err = np.append(Fnu_err, _Fnu_err)

    nuFnu = (nu * Fnu).to("erg cm-2 s-1")
    nuFnu_err = (nu * Fnu_err).to("erg cm-2 s-1")

    # now that we have the arrays, let us return a binned SED
    # let us consider logarithmic bins from 1 MHz to 1 THz, 5 bins per decade
    # the value in each bin is the weighted average of the nuFnu values in that bin
    bins = np.logspace(6, 12, 31) * u.Hz
    digitized = np.digitize(nu, bins)
    nu_bins_ctr = np.array([]) * u.GHz
    # store width of the bins to plot error bars on X
    nu_bins_err_neg = np.array([]) * u.GHz
    nu_bins_err_pos = np.array([]) * u.GHz
    nuFnu_binned = np.array([]) * u.Unit("erg cm-2 s-1")
    nuFnu_err_binned = np.array([]) * u.Unit("erg cm-2 s-1")

    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:
            nu_bin_center = np.sqrt(bins[i - 1] * bins[i])
            nu_bins_ctr = np.append(nu_bins_ctr, nu_bin_center)
            nu_bins_err_neg = np.append(nu_bins_err_neg, nu_bin_center - bins[i - 1])
            nu_bins_err_pos =  np.append(nu_bins_err_pos, bins[i] - nu_bin_center)

            # weighted average
            weights = 1 / (nuFnu_err[mask].value ** 2)
            nuFnu_avg = np.sum(nuFnu[mask].value * weights) / np.sum(weights)
            nuFnu_binned = np.append(nuFnu_binned, nuFnu_avg * nuFnu.unit)

            # uncertainty on the weighted average
            nuFnu_err_avg = np.sqrt(1 / np.sum(weights))
            nuFnu_err_binned = np.append(nuFnu_err_binned, nuFnu_err_avg * nuFnu_err.unit)

    return {
        "nu" : nu.to("Hz"),
        "nuFnu" : nuFnu,
        "nuFnu_err" : nuFnu_err,
        "nu_bins_ctr" : nu_bins_ctr.to("Hz"),
        "nu_bins_err_neg" : nu_bins_err_neg.to("Hz"),
        "nu_bins_err_pos" : nu_bins_err_pos.to("Hz"),
        "nuFnu_binned" : nuFnu_binned,
        "nuFnu_err_binned" : nuFnu_err_binned
    }


# --------------------------------------------------------------
# functions interacting with VIZIER (catalogue queries)
# --------------------------------------------------------------
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
