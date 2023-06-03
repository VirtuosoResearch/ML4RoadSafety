# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


acc_count = np.array([3811.0,
 5735.0,
 7152.0,
 9653.0,
 10537.0,
 10444.0,
 13665.0,
 11980.0,
 13170.0,
 16183.0,
 13618.0,
 17802.0,
 19750.0,
 22203.0,
 19860.0,
 26430.0,
 28893.0,
 37418.0,
 47767.0,
 74679.0])

acc_count = np.reshape(acc_count, (10,2))
acc_count = np.sum(acc_count, axis=1)

aadt_bins = np.array([6891.0,
 17705.5,
 26609.5,
 35409.0,
 44192.0,
 53918.5,
 64554.5,
 75497.5,
 87593.0,
 100725.0,
 115052.5,
 132087.0,
 151931.5,
 174521.5,
 200117.0,
 229674.5,
 271385.0,
 345547.0,
 570706.5,
 3745554.5])

x_axis = np.arange(0,len(acc_count))

# Plot the histogram
fig, ax = plt.subplots(figsize=(6.5, 5))

ax.bar(x_axis, acc_count, color='royalblue')
plt.xlabel(r'$\mathrm{Traffic~volume~/~day}$', fontsize=32)
plt.ylabel('Accidents', fontsize=32)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

plt.xticks(x_axis+0.5,[25, "", 50, "", 100,
"", 200, "", 400, ""],rotation=0)

plt.yticks(np.arange(0, 160000, 50000))
plt.ylim(0, 150000)

ax.tick_params(axis='both', which='major', labelsize=32)
ax.tick_params(axis='both', which='minor', labelsize=32)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
ax.xaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/traffic_volume_vs_accidents.pdf', format='pdf', dpi=100)
plt.show()































