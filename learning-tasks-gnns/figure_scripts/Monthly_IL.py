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


monthly_acc_IL = np.array([25.818900000000003,
 23.6395,
 22.3858,
 21.547400000000003,
 24.8629,
 25.290200000000002,
 24.5054,
 24.6045,
 24.493299999999998,
 27.2167,
 27.047900000000002,
 26.657700000000002])*1000

seasonal = np.array([
    monthly_acc_IL[[11, 0, 1]].sum(), # Winter
    monthly_acc_IL[[2, 3, 4]].sum(), # Spring
    monthly_acc_IL[[5, 6, 7]].sum(), # Summer
    monthly_acc_IL[[8, 9, 10]].sum(), # Fall
])


x_axis = np.arange(len(seasonal))


# Plotting with seaborn
fig, ax = plt.subplots(figsize=(6.5, 5))
# sns.lineplot(data=df, x='year', y='acc_count', marker='o', linewidth=1)
ax.bar(x_axis, seasonal, width = 0.6, color='royalblue',)
# ax.scatter(x_axis, seasonal, marker='o', s=150, color='royalblue')


# # Customize the x-ticks and labels
plt.xticks([0, 1, 2.1, 3], [ "Winter", "Spring", "Summer", "Fall"], rotation=0, ha='center')
plt.yticks(np.arange(65000, 81000, 5000))
plt.ylim(65000, 80000)

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
plt.tight_layout()
plt.savefig('./figures/Monthly_Accidents_IL.pdf', format='pdf', dpi=100)
plt.show()
