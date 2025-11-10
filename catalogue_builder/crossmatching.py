
def search_x_ray_counterparts(self):
    """Search for X-ray counterparts in various catalogues -- MORX, 4XMM-DR14, CSC2.1, 2SXPS, BAT 157 Month Catalog.
    Make sure x_ray_catalogs = [morx[0], fourxmm, cxotwo[0], twosxps_swift[0], bat157]
    """
    # Getting XMM, CXO and Swift counterparts from MORX
    coords_morx = SkyCoord(
        ra=self.x_ray_catalogs[0]["RAJ2000"],
        dec=self.x_ray_catalogs[0]["DEJ2000"],
        unit=(u.deg, u.deg),
    )
    crossmatch_morx = match_coordinates_sky(
        self.source_coords, coords_morx, nthneighbor=1
    )
    self.morx_xmm = self.x_ray_catalogs[0][crossmatch_morx[0].item()]["XMM-ID"]
    self.morx_cxo = self.x_ray_catalogs[0][crossmatch_morx[0].item()]["CX-ID"]
    self.morx_swift = self.x_ray_catalogs[0][crossmatch_morx[0].item()]["Swift-ID"]
    self.morx_lotss = self.x_ray_catalogs[0][crossmatch_morx[0].item()]["LoTSS-ID"]
    self.morx_vlass = self.x_ray_catalogs[0][crossmatch_morx[0].item()]["VLASS-ID"]
    self.morx_seperation = crossmatch_morx[1].item()
    # Sky extent of Longer Lobe in arcseconds
    self.lobe_extension = (
        self.x_ray_catalogs[0][crossmatch_morx[0].item()]["Lobedist"] * u.mas
    )
    # 4XMM-DR14 counterpart
    coords_4xmm = SkyCoord(
        ra=self.x_ray_catalogs[1]["ra"],
        dec=self.x_ray_catalogs[1]["dec"],
        unit=(u.deg, u.deg),
    )
    crossmatch_4xmm = match_coordinates_sky(
        self.source_coords, coords_4xmm, nthneighbor=1
    )
    self.xmm_id = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]["iauname"]
    self.xmm_flux4 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()][
        "sc_ep_4_flux"
    ]
    self.xmm_flux4_err = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()][
        "sc_ep_4_flux_err"
    ]
    self.xmm_flux5 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()][
        "sc_ep_5_flux"
    ]
    self.xmm_flux5_err = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()][
        "sc_ep_5_flux_err"
    ]
    self.xmm_hr3 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]["sc_hr3"]
    self.xmm_hr3e = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]["sc_hr3_err"]
    self.xmm_hr4 = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]["sc_hr4"]
    self.xmm_hr4e = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()]["sc_hr4_err"]
    self.xmm_variability_flag = self.x_ray_catalogs[1][crossmatch_4xmm[0].item()][
        "sc_var_flag"
    ]
    self.xmm_seperation = crossmatch_4xmm[1].item()
    # CSC2.1 counterpart
    coords_cxo = SkyCoord(
        ra=self.x_ray_catalogs[2]["RAICRS"],
        dec=self.x_ray_catalogs[2]["DEICRS"],
        unit=(u.deg, u.deg),
    )
    crossmatch_cxo = match_coordinates_sky(
        self.source_coords, coords_cxo, nthneighbor=1
    )
    self.cxo_id = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["2CXO"]
    self.cxo_flux = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["FPL0.5-7"]
    self.cxo_flux_lerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()][
        "b_FPL0.5-7"
    ]
    self.cxo_flux_uerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()][
        "B_FPL0.5-7"
    ]
    self.cxo_phoindex = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["GamPL"]
    self.cxo_phoindex_lerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()][
        "b_GamPL"
    ]
    self.cxo_phoindex_uerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()][
        "B_GamPL"
    ]
    self.cxo_hr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["HRhm"]
    self.cxo_hr_lerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["b_HRhm"]
    self.cxo_hr_uerr = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["B_HRhm"]
    self.cxo_var_flag = self.x_ray_catalogs[2][crossmatch_cxo[0].item()]["fv"]
    self.cxo_seperation = crossmatch_cxo[1].item()
    # 2SXPS counterpart
    coords_swift = SkyCoord(
        ra=self.x_ray_catalogs[3]["RAJ2000"],
        dec=self.x_ray_catalogs[3]["DEJ2000"],
        unit=(u.deg, u.deg),
    )
    crossmatch_swift = match_coordinates_sky(
        self.source_coords, coords_swift, nthneighbor=1
    )
    self.swift_id = self.x_ray_catalogs[3][crossmatch_swift[0].item()]["IAUName"]
    self.swift_flux = self.x_ray_catalogs[3][crossmatch_swift[0].item()]["FPCO0"]
    self.swift_flux_lerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()][
        "e_FPCO0"
    ]
    self.swift_flux_uerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()][
        "E_FPCO0"
    ]
    self.swift_phoindex = self.x_ray_catalogs[3][crossmatch_swift[0].item()][
        "Gamma"
    ]
    self.swift_phoindex_lerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()][
        "e_Gamma"
    ]
    self.swift_phoindex_uerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()][
        "E_Gamma"
    ]
    self.swift_hr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]["HR2"]
    self.swift_hr_lerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]["e_HR2"]
    self.swift_hr_uerr = self.x_ray_catalogs[3][crossmatch_swift[0].item()]["E_HR2"]
    self.swift_seperation = crossmatch_swift[1].item()
    # BAT 157 Month Survey Catalog counterpart
    coords_bat = SkyCoord(
        ra=self.x_ray_catalogs[4]["col3"],
        dec=self.x_ray_catalogs[4]["col4"],
        unit=(u.deg, u.deg),
    )
    crossmatch_bat = match_coordinates_sky(
        self.source_coords, coords_bat, nthneighbor=1
    )
    self.bat_id = self.x_ray_catalogs[4][crossmatch_bat[0].item()]["col2"]
    self.bat_seperation = crossmatch_bat[1].item()

def search_gamma_ray_counterparts(self):
    """Search for gamma-ray counterparts in various catalogues -- Fermi 4FGL-DR4 and Fermi Transient 1FLT Catalog.
    Make sure gamma_ray_catalogs = [fermi_4fgl, fermi_transient]"""

    # Fermi 4FGL-DR4 counterpart
    coords_4fgl = SkyCoord(
        ra=self.gamma_ray_catalogs[0]["RAJ2000"],
        dec=self.gamma_ray_catalogs[0]["DEJ2000"],
        unit=(u.deg, u.deg),
    )
    crossmatch_4fgl = match_coordinates_sky(
        self.source_coords, coords_4fgl, nthneighbor=1
    )
    self.fgl_id = self.gamma_ray_catalogs[0][crossmatch_4fgl[0].item()][
        "Source_Name"
    ]
    self.fgl_seperation = crossmatch_4fgl[1].item()
    # Fermi Transient 1FLT counterpart
    coords_1flt = SkyCoord(
        ra=self.gamma_ray_catalogs[1]["RAJ2000"],
        dec=self.gamma_ray_catalogs[1]["DEJ2000"],
        unit=(u.deg, u.deg),
    )
    crossmatch_1flt = match_coordinates_sky(
        self.source_coords, coords_1flt, nthneighbor=1
    )
    self.flt_id = self.gamma_ray_catalogs[1][crossmatch_1flt[0].item()][
        "Source_Name"
    ]
    self.flt_seperation = crossmatch_1flt[1].item()

        def flux_information(self):
        _string = f"""
            name: {self.name}\n
            4XMM DR14 (2-4.5 keV) flux: {self.xmm_flux4} +/- {self.xmm_flux4_err}\n
            4XMM DR14 (4.5-12 keV) flux: {self.xmm_flux5} +/- {self.xmm_flux5_err}\n
            4XMM DR14 HR (1-2 keV /2-4.5 keV): {self.xmm_hr3} +/- {self.xmm_hr3e}\n
            4XMM DR14 HR (2-4.5 keV /4.5-12 keV): {self.xmm_hr4} +/- {self.xmm_hr4e}\n
            2CXO FAPL (0.5-7 keV) Flux: {self.cxo_flux} +/- {0.5*(self.cxo_flux_uerr+self.cxo_flux_lerr)}
            2CXO Best Fit Photon Index: {self.cxo_phoindex} +/- {0.5*(self.cxo_phoindex_lerr+self.cxo_flux_uerr)}
            2CXO Hardness Ratio (Hard-Medium): {self.cxo_hr} +/- {0.5*(self.cxo_hr_lerr+self.cxo_hr_uerr)}
            2SXPS Flux (0.3-10 keV): {self.swift_flux} +/- {0.5*(self.swift_flux_uerr+self.swift_flux_lerr)}
            2SXPS Best Fit Photon Index: {self.swift_phoindex} +/- {0.5*(self.swift_phoindex_lerr+self.swift_phoindex_uerr)}
            2SXPS Hardness Ratio: {self.swift_hr} +/- {0.5*(self.swift_hr_lerr+self.swift_hr_uerr)}
        """
        return _string


    def write_catalogue_row(self, table):
        """Write the source information into a catalogue row."""
        table.add_row(
            [
                self.name,
                self.ra,
                self.dec,
                self.source_type,
                self.morx_lotss,
                self.morx_vlass,
                self.nvss_id_simbad,
                self.first_id_simbad,
                self.sdss_id_simbad,
                self.morx_xmm,
                self.xmm_id,
                self.morx_cxo,
                self.cxo_id,
                self.morx_swift,
                self.swift_id,
                self.bat_id,
                self.fgl_id,
                self.flt_id,
                self.torresi_detection,
                self.xmm_variability_flag,
                self.cxo_var_flag,
                self.lobe_extension,
                self.distance,
                self.LogL_OIII,
                convert_F_nu_to_luminosity(
                    1.4 * u.GHz, self.nvss_flux, u.mJy, self.distance
                ),
                convert_F_nu_to_luminosity(
                    1.4 * u.GHz, self.nvss_flux_error, u.mJy, self.distance
                ),
                convert_F_nu_to_luminosity(
                    1.4 * u.GHz, self.first_flux, u.mJy, self.distance
                ),
                convert_F_nu_to_luminosity(
                    1.4 * u.GHz, self.first_flux_error, u.mJy, self.distance
                ),
                convert_flux_to_luminosity(self.xmm_flux4, self.distance),
                convert_flux_to_luminosity(self.xmm_flux4_err, self.distance),
                convert_flux_to_luminosity(self.xmm_flux5, self.distance),
                convert_flux_to_luminosity(self.xmm_flux5_err, self.distance),
                self.xmm_hr3,
                self.xmm_hr3e,
                self.xmm_hr4,
                self.xmm_hr4e,
                convert_flux_to_luminosity(self.cxo_flux, self.distance),
                convert_flux_to_luminosity(self.cxo_flux_lerr, self.distance),
                convert_flux_to_luminosity(self.cxo_flux_uerr, self.distance),
                self.cxo_phoindex,
                self.cxo_phoindex_lerr,
                self.cxo_phoindex_uerr,
                self.cxo_hr,
                self.cxo_hr_lerr,
                self.cxo_hr_uerr,
                convert_flux_to_luminosity(self.swift_flux, self.distance),
                convert_flux_to_luminosity(self.swift_flux_lerr, self.distance),
                convert_flux_to_luminosity(self.swift_flux_uerr, self.distance),
                self.swift_phoindex,
                self.swift_phoindex_lerr,
                self.swift_phoindex_uerr,
                self.swift_hr,
                self.swift_hr_lerr,
                self.swift_hr_uerr,
            ]
        )


    def get_OIII_luminosity(self):
        """Calculate the [OIII] luminosity from Ho et al. (1997) or FR0CAT."""
        match_ho = ho_1997[1]["Name"] == insert_space_source_ids(self.name)
        if np.any(match_ho):  # Check if there is at least one True
            _log_L_alpha = ho_1997[1]["logL(Ha)"][match_ho][0]
            _OIII = ho_1997[1]["[OIII]"][match_ho][0]
            L_OIII = np.power(10, _log_L_alpha) * _OIII * u.Unit("erg s-1")
            self.LogL_OIII = np.log10(L_OIII.to_value("erg s-1"))
        elif np.any(self.sdss_id_simbad == fr0cat[0]["SDSS"]):
            sdss_match = self.sdss_id_simbad == fr0cat[0]["SDSS"]
            self.LogL_OIII = fr0cat[0]["logL[OIII]"][sdss_match][0]
        else:
            self.LogL_OIII = 0

