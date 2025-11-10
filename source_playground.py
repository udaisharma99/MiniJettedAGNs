import logging
from catalogue_builder import Source, search_morx_counterpart
import matplotlib.pyplot as plt
import IPython

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(name)s|%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

source = Source("NGC 1275")
print(source)

a, b, c = search_morx_counterpart(source)
import IPython; IPython.embed()


"""
IPython.embed()

fig, ax = plt.subplots()
ax.errorbar(
    sed["nu"].to("Hz").value,
    sed["nuFnu"].value,
    yerr=sed["nuFnu_err"].value,
    marker=".",
    color="gray",
    alpha=0.5,
    ls="",
    label="original"
)
ax.errorbar(
    sed["nu_bins_ctr"].to("Hz").value,
    sed["nuFnu_binned"].value,
    xerr=[
        sed["nu_bins_err_neg"].to("Hz").value,
        sed["nu_bins_err_pos"].to("Hz").value
    ],
    yerr=sed["nuFnu_err_binned"].value,
    marker="o",
    lw=2,
    ls="",
    label="binned"
)
ax.set_xlim([1e6, 1e12])
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
"""


