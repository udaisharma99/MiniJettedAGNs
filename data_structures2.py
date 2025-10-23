# structure of the table for the final catalgoue
import logging
import numpy as np
from astropy.table import Table, Column
from astroquery.simbad import Simbad


# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(level)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def get_source_survey_identifier(source_name, start_id):
    """Get the source identifier from SIMBAD starting with a given string
    e.g. "NVSS", "FIRST, "SDSS", "4FGL", etc."""
    identifiers = Simbad.query_objectids(source_name)
    mask = [string.startswith(start_id) for string in identifiers["id"]]
    ids = identifiers["id"][mask]
    if len(ids) == 0:
        log.info(f"{start_id} counterpart not available for {source_name}")
        return ""
    else:
        log.info(f"{source_name} matched with {ids[0]} by SIMBAD")
        return ids[0]

class Source():
    """
    Class to represent a single source in the catalogue.
    It will contain basic information like coordinates, name, and type.
    It will also contain basic methods to find FIRST, NVSS and SDSS names.
    """

    def __init__(self, name):
        self.name = name
        # already at initialisation, we find the NVSS, FIRST and SDSS counterparts
        # in principle all the sources should be in these deep surveys
        self.find_nvss_first_sdss_counterparts()

    def find_nvss_first_sdss_counterparts(self):
        """find the NVSS, FIRST, and SDSS identifiers. 
        In principle all the sources should be in these deep surveys.
        """
        self.sdss_id = get_source_identifier(self.name,"SDSS")
        self.nvss_id = get_source_identifier(self.name, "NVSS")
        self.first_id = get_source_identifier(self.name, "FIRST")

    def get_L_0III_luminosity(self):
        """Obtain the [O III] luminosity from SIMBAD"""
        Simbad.add_votable_fields("flux(OIII5007)")
        simbad_query = Simbad.query_object(self.name)
        if simbad_query:
            flux_0III = simbad_query["FLUX_OIII5007"][0]  # in 10^-17 erg/cm2/s
            distance = simbad_query["Distance_distance"][0]  # in pc
            L_0III = flux_0III * 1e-17 * 4 * np.pi * (distance * 3.086e18)**2  # in erg/s
            return np.log10(L_0III)
        else:
            return -1


    def from_catalogue_row(cls, row):
        """Create a Source object from a row in the catalogue table.
        We make it work with the CoreG and FR0 catalgoues."""

        return cls(
            ra_j2000=row["RA"],
            dec_j2000=row["DEC"],
            source_name=row["SOURCE_NAME"],
            source_type=row["SOURCE_TYPE"]
        )


# build the table structure
columns = [
    Column(name="RA", data=None, dtype="float64", unit="deg"),
    Column(name="DEC", data=None, dtype="float64", unit="deg"),
    Column(name="SOURCE_NAME", data=None, dtype="str", unit=""),
    Column(name="SOURCE_TYPE", data=None, dtype="str", unit=""),
    Column(name="LoTSS-MORX-ID", data=None, dtype="str", unit=""),
    Column(name="VLASS-MORX-ID", data=None, dtype="str", unit=""),
    Column(name="NVSS-ID", data=None, dtype="str", unit="", description="from SIMBAD"),
    Column(name="FIRST-ID", data=None, dtype="str", unit="", description="from SIMBAD"),
    Column(name="SDSS-ID", data=None, dtype="str", unit="", description="from SIMBAD"),
    Column(name="MORX-XMM-ID", data=None, dtype="str", unit=""),
    Column(name="4XMM-DR14-ID", data=None, dtype="str", unit="", description="updating info in MORX"),
    Column(name="MORX-CXO-ID", data=None, dtype="str", unit=""),
    Column(name="CSC2.1-ID", data=None, dtype="str", unit="", description="updating info in MORX"),
    Column(name="MORX-SWIFT-ID", data=None, dtype="str", unit=""),
    Column(name="2SXPS-ID", data=None, dtype="str", unit="", description="updating info in MORX"),
    Column(name="BAT-105Month-ID", data=None, dtype="str", unit=""),
    Column(name="FERMI-ID", data=None, dtype="str", unit=""),
    Column(name="1FLT-ID", data=None, dtype="str", unit=""),
    Column(name="TORESSI DETECTION", data=None, dtype="bool", unit=""),
    Column(name="4XMM Variability Flag", data=None, dtype="str", unit=""),
    Column(name="Chandra Variability", data=None, dtype="str", unit=""),
    Column(name="LOBE EXTENSION", data=None, dtype="float64", unit="mas"),
    Column(name="DISTANCE", data=None, dtype="float64", unit="Mpc"),
    Column(name="Log10(L_OIII)", data=None, dtype="float64", unit="erg s-1"),
    Column(name="NVSS-FLUX-XMATCH", data=None, dtype="float64", unit="erg s-1"),
    Column(name="NVSS-FLUX-ERROR-XMATCH", data=None, dtype="float64", unit="erg s-1"),
    Column(name="FIRST-FLUX-XMATCH", data=None, dtype="float64", unit="erg s-1"),
    Column(name="FIRST-FLUX-ERROR-XMATCH", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_1", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_1e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_2", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_2e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_3", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_3e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_4", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_4e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_5", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_5e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_8", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_8e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_9", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 L_9e", data=None, dtype="float64", unit="erg s-1"),
    Column(name="4XMM-DR14 HR_1", data=None, dtype="float64", unit=""),
    Column(name="4XMM-DR14 HR_2", data=None, dtype="float64", unit=""),
    Column(name="4XMM-DR14 HR_3", data=None, dtype="float64", unit=""),
    Column(name="4XMM-DR14 HR_4", data=None, dtype="float64", unit=""),
    Column(name="4XMM-DR14 Number of Detections", data=None, dtype="int64", unit=""),
    Column(name="CSC2.1 Fitted Absorbed PL Flux", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Fitted Absorbed PL Flux L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Fitted Absorbed PL Flux U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Fitted Absorbed PL PhoIndex", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 Fitted Absorbed PL PhoIndex L_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 Fitted Absorbed PL PhoIndex U_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 Flux Broad", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Broad L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Broad U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Hard", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Hard L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Hard U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Medium", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Medium L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Medium U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Soft", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Soft L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Soft U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Ultrasoft", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Ultrasoft L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Flux Ultrasoft U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="CSC2.1 Hardness Ratio HM", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 HR HM L_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 HR HM U_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 Hardness Ratio HS", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 HR HS L_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 HR HS U_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 Hardness Ratio MS", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 HR MS L_err", data=None, dtype="float64", unit=""),
    Column(name="CSC2.1 HR MS U_err", data=None, dtype="float64", unit=""),
    Column(name="SWIFT Mean Broad Flux Fitted PL", data=None, dtype="float64", unit="erg s-1"),
    Column(name="SWIFT Mean Broad Flux Fitted PL L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="SWIFT Mean Broad Flux Fitted PL U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="SWIFT Mean Broad Flux PL", data=None, dtype="float64", unit="erg s-1"),
    Column(name="SWIFT Mean Broad Flux PL L_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="SWIFT Mean Broad Flux PL U_err", data=None, dtype="float64", unit="erg s-1"),
    Column(name="SWIFT HR1", data=None, dtype="float64", unit=""),
    Column(name="SWIFT HR1 L_err", data=None, dtype="float64", unit=""),
    Column(name="SWIFT HR1 U_err", data=None, dtype="float64", unit=""),
    Column(name="SWIFT HR2", data=None, dtype="float64", unit=""),
    Column(name="SWIFT HR2 L_err", data=None, dtype="float64", unit=""),
    Column(name="SWIFT HR2 U_err", data=None, dtype="float64", unit=""),
]
table_coreG = Table(columns)
table_fr0 = Table(columns)

class CatalogBuilder():
    """
    """
    def __init__(self, table):
        self.table = table

    def add_source(self, RA, DEC, source_name, source_type, lotss_morx_id, vlass_morx_id,
                   nvss_id, first_id, sdss_id, morx_xmm_id, xmm_dr14_id,
                   morx_cxo_id, csc2_1_id, morx_swift_id, sxps_id, bat_105_id, fermi_id,
                   flt_id, toressi_detection, xmm_variability_flag,
                   chandra_variability, lobe_extension, distance, log_loiii,
                   nvss_flux_xmatch, nvss_flux_error_xmatch, first_flux_xmatch,
                   first_flux_error_xmatch, xmm_l_1, xmm_l_1e, xmm_l_2,
                   xmm_l_2e, xmm_l_3, xmm_l_3e, xmm_l_4, xmm_l_4e,
                   xmm_l_5, xmm_l_5e, xmm_l_8, xmm_l_8e, xmm_l_9,
                   xmm_l_9e, xmm_hr_1, xmm_hr_2, xmm_hr_3, xmm_hr_4,
                   xmm_number_of_detections, csc_fitted_absorbed_pl_flux,
                   csc_fitted_absorbed_pl_flux_l_err,
                   csc_fitted_absorbed_pl_flux_u_err,
                   csc_fitted_absorbed_pl_phoindex,
                   csc_fitted_absorbed_pl_phoindex_l_err,
                   csc_fitted_absorbed_pl_phoindex_u_err,
                   csc_flux_broad, csc_flux_broad_l_err,
                   csc_flux_broad_u_err, csc_flux_hard,
                   csc_flux_hard_l_err, csc_flux_hard_u_err,
                   csc_flux_medium, csc_flux_medium_l_err,
                   csc_flux_medium_u_err, csc_flux_soft,
                   csc_flux_soft_l_err, csc_flux_soft_u_err,
                   csc_flux_ultrasoft, csc_flux_ultrasoft_l_err,
                   csc_flux_ultrasoft_u_err, csc_hardness_ratio_hm,
                   csc_hr_hm_l_err, csc_hr_hm_u_err,
                   csc_hardness_ratio_hs, csc_hr_hs_l_err,
                   csc_hr_hs_u_err, csc_hardness_ratio_ms,
                   csc_hr_ms_l_err, csc_hr_ms_u_err,
                   swift_mean_broad_flux_fitted_pl,
                   swift_mean_broad_flux_fitted_pl_l_err,
                   swift_mean_broad_flux_fitted_pl_u_err,
                   swift_mean_broad_flux_pl, swift_mean_broad_flux_pl_l_err,
                   swift_mean_broad_flux_pl_u_err, swift_hr1, swift_hr1_l_err,
                   swift_hr1_u_err, swift_hr2, swift_hr2_l_err,
                   swift_hr2_u_err):
        self.table.add_row((RA, DEC, source_name, source_type, lotss_morx_id, vlass_morx_id,
                            nvss_id, first_id, sdss_id, morx_xmm_id, xmm_dr14_id,
                            morx_cxo_id, csc2_1_id, morx_swift_id, sxps_id, bat_105_id,fermi_id,
                            flt_id, toressi_detection, xmm_variability_flag,
                            chandra_variability, lobe_extension, distance, log_loiii,
                            nvss_flux_xmatch, nvss_flux_error_xmatch, first_flux_xmatch,
                            first_flux_error_xmatch, xmm_l_1, xmm_l_1e, xmm_l_2,
                            xmm_l_2e, xmm_l_3, xmm_l_3e, xmm_l_4, xmm_l_4e,
                            xmm_l_5, xmm_l_5e, xmm_l_8, xmm_l_8e, xmm_l_9,
                            xmm_l_9e, xmm_hr_1, xmm_hr_2, xmm_hr_3, xmm_hr_4,
                            xmm_number_of_detections, csc_fitted_absorbed_pl_flux,
                            csc_fitted_absorbed_pl_flux_l_err,
                            csc_fitted_absorbed_pl_flux_u_err,
                            csc_fitted_absorbed_pl_phoindex,
                            csc_fitted_absorbed_pl_phoindex_l_err,
                            csc_fitted_absorbed_pl_phoindex_u_err,
                            csc_flux_broad, csc_flux_broad_l_err,
                            csc_flux_broad_u_err, csc_flux_hard,
                            csc_flux_hard_l_err, csc_flux_hard_u_err,
                            csc_flux_medium, csc_flux_medium_l_err,
                            csc_flux_medium_u_err, csc_flux_soft,
                            csc_flux_soft_l_err, csc_flux_soft_u_err,
                            csc_flux_ultrasoft, csc_flux_ultrasoft_l_err,
                            csc_flux_ultrasoft_u_err, csc_hardness_ratio_hm,
                            csc_hr_hm_l_err, csc_hr_hm_u_err,
                            csc_hardness_ratio_hs, csc_hr_hs_l_err, csc_hr_hs_u_err,
                            csc_hardness_ratio_ms, csc_hr_ms_l_err, csc_hr_ms_u_err,
                            swift_mean_broad_flux_fitted_pl,
                            swift_mean_broad_flux_fitted_pl_l_err,
                            swift_mean_broad_flux_fitted_pl_u_err,
                            swift_mean_broad_flux_pl, swift_mean_broad_flux_pl_l_err,
                            swift_mean_broad_flux_pl_u_err, swift_hr1, swift_hr1_l_err,
                            swift_hr1_u_err, swift_hr2, swift_hr2_l_err,
                            swift_hr2_u_err))



