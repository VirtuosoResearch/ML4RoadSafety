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

acc_count = np.array([199.5,
 0.0,
 369.65238095238095,
 8201.82731990232,
 47769.45073398823,
 124988.29943251194,
 131764.15826257077,
 125823.56228077479,
 106660.4640900766,
 100401.27929015429,
 94802.55981657232,
 63682.35494227994,
 23983.32778887779,
 6343.988095238095,
 1443.625,
 594.3833333333333,
 10.066666666666666,
 147.52619047619046,
 40.833333333333336,
 138.36904761904762])

# acc_count = np.reshape(acc_count, (10,2))
# acc_count = np.sum(acc_count, axis=1)
# acc_count = acc_count[1:-1]
acc_count = acc_count[4:-6]
bins = np.array([0.85,
 2.6,
 4.32,
 6.06,
 7.78,
 9.52,
 11.24,
 12.98,
 14.7,
 16.44,
 18.16,
 19.9,
 21.62,
 23.35,
 25.08,
 26.82,
 28.54,
 30.28,
 32.0,
 33.74])


x_axis = np.arange(len(acc_count))

fig, ax = plt.subplots(figsize=(6.5, 5))

# Plot the histogram
ax.bar(x_axis, acc_count, color='royalblue')
# plt.xlabel('Temp Bin')
# plt.ylabel('acc_count')
# plt.title('Histogram of acc_count by Temp')
# plt.xticks(list(range(len(acc_count))),bins)

plt.xlabel(r'$\mathrm{Wind~Speed~}(\mathrm{mph})$', fontsize=32)
plt.ylabel('Accidents', fontsize=32)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],[
 8,
 "",
 12,
 "",
 16,
 "",
 20,
 "",
 24,
 "",],rotation=0)

plt.yticks(np.arange(0, 160000, 50000))
plt.ylim(0, 150000)

ax.tick_params(axis='both', which='major', labelsize=32)
ax.tick_params(axis='both', which='minor', labelsize=32)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
ax.xaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/wind_speed_vs_accidents.pdf', format='pdf', dpi=100)
plt.show()































