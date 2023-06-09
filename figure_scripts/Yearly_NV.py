
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



yearly_acc_NV = np.array([50.224, 51.751, 48.854, 49.813, 36.696])*1000



years_NV = [2016, 2017, 2018, 2019, 2020]
x_axis = np.arange(len(yearly_acc_NV))


# Plotting with seaborn
fig, ax = plt.subplots(figsize=(5.5, 5))
# sns.lineplot(data=df, x='year', y='acc_count', marker='o', linewidth=1)
ax.plot(x_axis, yearly_acc_NV, lw=3, color='royalblue')
ax.scatter(x_axis, yearly_acc_NV, marker='o', s=150, color='royalblue')


# # Customize the x-ticks and labels
plt.xticks(x_axis, [2016, "", 2018, "", 2020], rotation=30, ha='center')
# plt.yticks(np.arange(50000, 260000, 50000))
# plt.ylim(44000, 61000)

# plt.xlabel("Year", )
plt.ylabel('Accidents', fontsize=36)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Nevada}$', fontsize=36, x=0.55, y=1.0)

ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
ax.xaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/Yearly_Accidents_MT.pdf', format='pdf', dpi=100)
plt.show()