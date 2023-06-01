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


yearly_acc_IL = np.array([
 274.395,
 285.468,
 296.194,
 313.319,
 324.496,
 311.93,
 319.213,
 313.185,
 246.868,
 295.634
])*1000



years_IL = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
x_axis = np.arange(len(yearly_acc_IL))


# Plotting with seaborn
fig, ax = plt.subplots(figsize=(5.5, 5))
# sns.lineplot(data=df, x='year', y='acc_count', marker='o', linewidth=1)
ax.plot(x_axis, yearly_acc_IL, lw=3, color='royalblue')
ax.scatter(x_axis, yearly_acc_IL, marker='o', s=150, color='royalblue')


# # Customize the x-ticks and labels
plt.xticks(x_axis, [2012, "", 2014, "", 2016, "", 2018, "", 2020, ""], rotation=30, ha='center')

# plt.xlabel("Year", )
plt.ylabel('Accidents', fontsize=36)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Illinois}$', fontsize=36)

ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
ax.xaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/Yearly_Accidents_IL.pdf', format='pdf', dpi=100)
plt.show()




