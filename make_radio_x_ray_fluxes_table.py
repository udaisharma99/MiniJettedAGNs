# build a table with the radio and X-ray luminosities
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from catalog_builder import (
    nagar_2005,
    fr0cat,
    Source,
    search_morx_counterpart,
    get_4xmm_luminosities,
)
import warnings

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s.%(funcName)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# run as:
# `python make_radio_x_ray_fluxes_table.py coreg`, or `python make_radio_x_ray_fluxes_table.py fr0`
source_type = sys.argv[1]

if source_type == "coreg":
    source_names = [str(_) for _ in nagar_2005[0]["Name"]]
elif source_type == "fr0":
    source_names = ["SDSS" + _ for _ in fr0cat[0]["SDSS"]]
else:
    log.error(
        f"{source_type} source type speficied not recognised try `coreg`or `fr0`, quitting..."
    )
    quit()

log.info(f"fetching radio and X-ray information for {source_type} sources")
# let us turn each of the sources into an instance of the `Source` class
sources = [Source(name) for name in source_names]

# as we focus on sources with radio properties, let's preventicely select
# only sources that in NED have at least 2 radio measurements
log.info("filter first only the sources with at least 2 radio measurements in NED")
radio_sources = []

for source in sources:
    # let's make it three points as we might want to fit them
    if source.radio_sed is not None and len(source.radio_sed["nu"]) > 2:
        radio_sources.append(source)

log.info(
    f"{len(radio_sources)} of {len(sources)} have at least 2 radio measurements and will be considered."
)

# let us start to build a table with columns
# source name, NVSS flux, L_nvss, Flux6, e_Flux6, Flux7, e_Flux7 in 4XMM
# for the definition of the X-ray bands see: https://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=IX/69
# empty lists for the columns tables
names = []
nvss_ids_simbad = []
first_ids_simbad = []
nvss_ids_morx = []
first_ids_morx = []
# store also the radio SED points for good measure
nu_radio_seds = []
nu_fnu_radio_seds = []
nu_fnu_err_radio_seds = []
L_nvss = []
L_nvss_err = []
H_alpha_fluxes = []  # TODO: store also this info, not filled now
O_III_fluxes = []  # TODO: store also this info, not filled now
L_4xmm_soft = []
L_4xmm_soft_err = []
L_4xmm_hard = []
L_4xmm_hard_err = []
hardness_ratios_xmm = []

for source in radio_sources:
    names.append(source.name)
    nvss_ids_simbad.append(source.nvss_id_simbad)
    first_ids_simbad.append(source.first_id_simbad)
    # get the NVSS luminosity from the SIMBAD name
    _L_nvss, _L_nvss_err = source.get_L_nvss()
    L_nvss.append(_L_nvss)
    L_nvss_err.append(_L_nvss_err)
    # search NVSS and FIRST counterparts in MORX
    morx_row = search_morx_counterpart(source)
    if morx_row is not None:
        nvss_ids_morx.append(morx_row["NVSS-ID"])
        first_ids_morx.append(morx_row["FIRST-ID"])
        _l_x_soft, _l_x_soft_err, _l_x_hard, _l_x_hard_err = get_4xmm_luminosities(
            morx_row, source.d_L
        )
        L_4xmm_soft.append(_l_x_soft)
        L_4xmm_soft_err.append(_l_x_soft_err)
        L_4xmm_hard.append(_l_x_hard)
        L_4xmm_hard_err.append(_l_x_hard_err)
    else:
        nvss_ids_morx.append("")
        first_ids_morx.append("")
        L_4xmm_soft.append(np.nan)
        L_4xmm_soft_err.append(np.nan)
        L_4xmm_hard.append(np.nan)
        L_4xmm_hard_err.append(np.nan)

    # now store also the radio SED
    nu_radio_seds.append(source.radio_sed["nu"])
    nu_fnu_radio_seds.append(source.radio_sed["nuFnu"])
    nu_fnu_err_radio_seds.append(source.radio_sed["nuFnu_err"])


# create and save the table
df = pd.DataFrame(
    {
        "names": names,
        "nvss_id_simbad": nvss_ids_simbad,
        "nvss_id_morx": nvss_ids_morx,
        "first_id_simbad": first_ids_simbad,
        "first_id_morx": first_ids_morx,
        "L_nvss": L_nvss,
        "L_nvss_err": L_nvss_err,
        "L_4xmm_soft": L_4xmm_soft,
        "L_4xmm_soft_err": L_4xmm_soft_err,
        "L_4xmm_hard": L_4xmm_hard,
        "L_4xmm_hard_err": L_4xmm_hard_err,
        "nu_radio_sed": nu_fnu_radio_seds,
        "nu_fnu_radio_sed": nu_fnu_radio_seds,
        "nu_fnu_err_radio_sed": nu_fnu_err_radio_seds,
    }
)

# save the dataframe
outfile = f"radio_x_table_{source_type}.csv"
outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)
df.to_csv(f"{outdir}/{outfile}")
