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

l2 = np.array([70.60, 83.24, 87.34, 91.88])
l3 = np.array([75.63, 84.64, 88.65, 92.96])

N = 4
ind = np.arange(N) * 4  # the x locations for the groups
width = 0.8      # the width of the bars
shift = 0.2


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(20,6))

rects3 = ax.bar(ind + width * 1 + shift, l2, width, color='royalblue', ecolor='white')
rects2 = ax.bar(ind + width * 2 + shift*2, l3, width, color='orange', ecolor='white')

ax.legend(
    (rects3[0], rects2[0]), 
    (r'$\mathrm{STL~on~accident~prediction}$', r'$\mathrm{Joint~training~w/~traffic~volume~prediction}$'), 
    loc=2, fontsize=34)
plt.xticks([])

name_list = ['Massachusetts', 'Delaware', 'Maryland', 'Nevada']
ax.set_xticks(ind + width  + shift + 0.5)
ax.set_xticklabels(name_list)

# plt.xlim([0.3, 2.7])
plt.ylim([67, 100])
plt.yticks(np.arange(70, 105, 5), [70, "", 80, "", 90, "", 100])
plt.ylabel(r'$\mathrm{AUC{-}ROC}~(\%)$', fontsize=40)

# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(28)

ax.yaxis.grid(True, lw=0.4)
# ax.set_title(r'$\mathrm{Massachusetts}$', fontsize=32)

ax.tick_params(axis='both', which='major', labelsize=40)
ax.tick_params(axis='both', which='minor', labelsize=40)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/joint_w_traffic_volume.pdf', format='pdf', dpi=100)
plt.show()
# %%
