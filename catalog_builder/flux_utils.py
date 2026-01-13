# utils functions to handle fluxes conversion and SED compilation
import logging
import numpy as np
import astropy.units as u
from astropy.table import Column
import matplotlib.pyplot as plt


# set up logging, get it from the script that imports this module
log = logging.getLogger(__name__)


def convert_flux_to_luminosity(flux, distance):
    """From a flux in mW/m2, returns a luminosity in erg s-1."""
    flux = flux * u.mW / u.m**2
    D_L = distance.to(u.cm)  # Convert Mpc to cm for consistent units
    flux_in_cgs = flux.to(u.erg / (u.s * u.cm**2))
    luminosity = 4 * np.pi * D_L**2 * flux_in_cgs
    return luminosity.to(u.erg / u.s)


def convert_F_nu_to_luminosity(nu, F_nu, F_nu_unit, distance):
    """From a F_nu [erg cm-2 s-1 Hz-1], returns a luminosity in erg s-1.
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
        # for this SED, we will consider only measurements with uncertainities.
        if np.isnan(uncertainty) or is_ul:
            continue
        else:
            # read the band, and eliminate anything that is not a number or a unit, e.g.
            # 347 GHz (SCUBA)
            # 1 mm (IRAM 30m)
            # in principle should be enough to remove anything following the unit
            partitioner = "Hz" if "Hz" in band else "mm"
            trailing_term = band.partition(partitioner)[
                -1
            ]  # this is the telescope info e.g. "(NVSS)"
            _nu = u.Quantity(band.replace(trailing_term, ""))
            # convert to frequency in Hz
            _nu = _nu.to("Hz", equivalencies=u.spectral())
            nu = np.append(nu, _nu)

            _Fnu = value * u.Unit(unit)
            Fnu = np.append(Fnu, _Fnu)
            _Fnu_err = uncertainty * u.Unit(unit)
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
            nu_bins_err_pos = np.append(nu_bins_err_pos, bins[i] - nu_bin_center)

            # weighted average
            weights = 1 / (nuFnu_err[mask].value ** 2)
            nuFnu_avg = np.sum(nuFnu[mask].value * weights) / np.sum(weights)
            nuFnu_binned = np.append(nuFnu_binned, nuFnu_avg * nuFnu.unit)

            # uncertainty on the weighted average
            nuFnu_err_avg = np.sqrt(1 / np.sum(weights))
            nuFnu_err_binned = np.append(
                nuFnu_err_binned, nuFnu_err_avg * nuFnu_err.unit
            )

    return {
        "nu": nu.to("Hz"),
        "nuFnu": nuFnu,
        "nuFnu_err": nuFnu_err,
        "nu_bins_ctr": nu_bins_ctr.to("Hz"),
        "nu_bins_err_neg": nu_bins_err_neg.to("Hz"),
        "nu_bins_err_pos": nu_bins_err_pos.to("Hz"),
        "nuFnu_binned": nuFnu_binned,
        "nuFnu_err_binned": nuFnu_err_binned,
    }


def _plot_radio_sed(sed, ax=None):
    """Plot the radio SED, both original and binned.
    sed has the format returned by `compile_radio_sed` function."""
    if ax is None:
        ax = plt.gca()

    ax.errorbar(
        sed["nu"].to("Hz").value,
        sed["nuFnu"].value,
        yerr=sed["nuFnu_err"].value,
        marker=".",
        color="gray",
        alpha=0.5,
        ls="",
        label="original",
    )
    ax.errorbar(
        sed["nu_bins_ctr"].to("Hz").value,
        sed["nuFnu_binned"].value,
        xerr=[
            sed["nu_bins_err_neg"].to("Hz").value,
            sed["nu_bins_err_pos"].to("Hz").value,
        ],
        yerr=sed["nuFnu_err_binned"].value,
        marker="o",
        lw=2,
        ls="",
        label="binned",
    )
    # ax.set_xlim([1e6, 1e12])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu\,/\,{\rm Hz}$")
    ax.set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
    ax.legend()
