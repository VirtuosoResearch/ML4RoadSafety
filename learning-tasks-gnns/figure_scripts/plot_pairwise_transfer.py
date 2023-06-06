# %%
from inspect import trace
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


results = [[0.0, 0.17, 0.56, 0.38, 0.18, -0.01, -0.01, 0.29],
[0.88, 0.0, 1.2, 1.01, 0.57, 0.77, 0.83, 1.98],
[0.56, -0.08, 0.0, 0.37, 0.29, -0.13, 0.14, 0.14],
[0.89, 0.28, 0.68, 0.0, -0.54, -0.32, -0.05, 0.26],
[0.33, 0.11, 0.58, 0.66, 0.0, 0.06, 0.27, 0.27],
[3.04, 3.03, 3.49, 2.94, 3.44, 0.0, 2.95, 3.14],
[0.08, 0.15, -0.23, -0.01, 0.06, 0.29, 0.0, 0.19],
[0.95, -4.37, -3.89, 1.17, 1.14, 0.52, 0.2, 0.0]]
results = [np.array(trace) for trace in results]
results = np.array(results)

fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(results,cmap="RdYlGn")

cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel(rotation=-90, va="bottom", fontsize=24)
cbar.set_ticks([-1, 0, 1, 2, 3])
im.set_clim(-1.5, 1.5)  
cbar.set_ticklabels([r"$-1$", r"$0$", r"$1$", r"$2$", r"$3$"])
cbar.ax.tick_params(labelsize=28)

ax.set_xticks([])
# for minor ticks
ax.set_yticks([])
ax.set_xlabel(r"$\mathrm{Source~state}$", fontsize = 32)
ax.set_ylabel(r"$\mathrm{Target~state}$", fontsize = 32)
fig.tight_layout()
plt.savefig(f"./figures/pairwise_transfer_results.pdf", format="pdf", dpi=1200)

plt.show()