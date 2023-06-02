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


monthly_acc_MT = np.array([1.8032000000000001,
 1.7712,
 1.3316,
 1.154,
 1.3164,
 1.5582,
 1.7004000000000001,
 1.6172,
 1.6056,
 1.8434000000000001,
 2.0558,
 2.2308000000000003])*1000

seasonal = np.array([
    monthly_acc_MT[[11, 0, 1]].sum(), # Winter
    monthly_acc_MT[[2, 3, 4]].sum(), # Spring
    monthly_acc_MT[[5, 6, 7]].sum(), # Summer
    monthly_acc_MT[[8, 9, 10]].sum(), # Fall
])


x_axis = np.arange(len(seasonal))


# Plotting with seaborn
fig, ax = plt.subplots(figsize=(6.2, 5))
# sns.lineplot(data=df, x='year', y='acc_count', marker='o', linewidth=1)
ax.bar(x_axis, seasonal, width = 0.6, color='royalblue',)
# ax.scatter(x_axis, seasonal, marker='o', s=150, color='royalblue')


# # Customize the x-ticks and labels
plt.xticks([0, 1, 2.1, 3], [ "Winter", "Spring", "Summer", "Fall"], rotation=0, ha='center')
plt.yticks(np.arange(3000, 6100, 1000))
plt.ylim(3000, 6000)

# plt.xlabel("Year", )
plt.ylabel('Accidents', fontsize=36)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Montana}$', fontsize=36)

ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/Monthly_Accidents_MT.pdf', format='pdf', dpi=100)
plt.show()