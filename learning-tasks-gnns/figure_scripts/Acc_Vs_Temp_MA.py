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


acc_count = np.array([
 17096.0,
 11144.0,
 9860.0,
 8644.0,
 6343.0,
 4443.0,
 4000,
 4859.0])*8


bins = np.array([-1.15, 3.92, 7.3, 13.68, 17.38, 20.06, 22.76])

x_axis = np.arange(len(acc_count))

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the histogram
ax.bar(x_axis, acc_count, color='royalblue', width=0.7)
# plt.xlabel('Temp Bin')
# plt.ylabel('acc_count')
# plt.title('Histogram of acc_count by Temp')
# plt.xticks(list(range(len(acc_count))),bins)

plt.xlabel(r'$\mathrm{Temperature~}(\mathrm{C}^\circ)$', fontsize=32)
plt.ylabel('Accidents', fontsize=32)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

plt.xticks(x_axis+0.5,[0, "", 8, "", 16, "", 24, ""],rotation=0)

plt.yticks(np.arange(0, 160000, 50000))
# plt.ylim(0, 150000)

ax.tick_params(axis='both', which='major', labelsize=32)
ax.tick_params(axis='both', which='minor', labelsize=32)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
ax.xaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/temperature_vs_accidents.pdf', format='pdf', dpi=100)
plt.show()
