# make a plot of the radio and X-ray luminosities (indexes etc...)
import pandas as pd
import matplotlib.pyplot as plt

# load the table and drop all the rows with NaNs
df_coreg = pd.read_csv("results/radio_x_table_coreg.csv")
df_coreg = df_coreg.dropna()

df_fr0 = pd.read_csv("results/radio_x_table_fr0.csv")
df_fr0 = df_fr0.dropna()

# make a correlation plot of L_nvss vs L_soft / hard in X rays
fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
ax[0].errorbar(
    df_coreg["L_nvss"],
    df_coreg["L_4xmm_soft"],
    xerr=df_coreg["L_nvss_err"],
    yerr=df_coreg["L_4xmm_soft_err"],
    ls="",
    marker=".",
    label="Core G",
    color="C0",
)
ax[0].errorbar(
    df_fr0["L_nvss"],
    df_fr0["L_4xmm_soft"],
    xerr=df_fr0["L_nvss_err"],
    yerr=df_fr0["L_4xmm_soft_err"],
    ls="",
    marker=".",
    label="FR0",
    color="C1",
)
ax[1].errorbar(
    df_coreg["L_nvss"],
    df_coreg["L_4xmm_hard"],
    xerr=df_coreg["L_nvss_err"],
    yerr=df_coreg["L_4xmm_hard_err"],
    ls="",
    marker=".",
    label="Core G",
    color="C0",
)
ax[1].errorbar(
    df_fr0["L_nvss"],
    df_fr0["L_4xmm_hard"],
    xerr=df_fr0["L_nvss_err"],
    yerr=df_fr0["L_4xmm_hard_err"],
    ls="",
    marker=".",
    label="FR0",
    color="C1",
)
ax[0].set_ylabel(r"$L_{\rm 4XMM,\, soft}\,/\,({\rm erg}\,{\rm s}^{-1})$")
ax[1].set_ylabel(r"$L_{\rm 4XMM,\, hard}\,/\,({\rm erg}\,{\rm s}^{-1})$")
ax[1].set_xlabel(r"$L_{\rm NVSS}\,/\,({\rm erg}\,{\rm s}^{-1})$")

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set_yscale("log")

ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
fig.savefig("radio_x_ray_correlation.png")
