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



yearly_acc_MA = np.array([0.05867772890539441,
 0.06391409313746751,
 0.06651935589788936,
 0.07248351934396986,
 0.07029243395530886,
 0.06323418797069433,
 0.05785299095984999,
 0.07429305833604866,
 0.06375925398745863,
 0.06381033435314233,
 0.0653276563430348,
 0.06822202462279495,
 0.02445039418556393,
 0.013244004605238018,
 0.0561064968001589,
 0.05845490002914162])



years_MA = [2002,
 2003,
 2004,
 2005,
 2007,
 2008,
 2009,
 2010,
 2014,
 2015,
 2016,
 2017,
 2018,
 2019,
 2021,
 2022]
x_axis = np.arange(len(yearly_acc_MA))


# Plotting with seaborn
fig, ax = plt.subplots(figsize=(5.5, 5))
# sns.lineplot(data=df, x='year', y='acc_count', marker='o', linewidth=1)
ax.plot(x_axis, yearly_acc_MA, lw=3, color='royalblue')
ax.scatter(x_axis, yearly_acc_MA, marker='o', s=150, color='royalblue')


# # Customize the x-ticks and labels
plt.xticks([ 1,  3,  5,   7,  9, 11, 13,  15], [2004,  "",  2008, "", 2016,  "",  2020, ""], rotation=30, ha='center')
# plt.yticks(np.arange(50000, 260000, 50000))
# plt.ylim(44000, 61000)

# plt.xlabel("Year", )
plt.ylabel('Accidents', fontsize=36)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(28)

ax.yaxis.grid(True, lw=0.4)
ax.set_title(r'$\mathrm{Massachusetts}$', fontsize=36, x=0.55, y=1.0)

ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)


# Display the plot
ax.yaxis.grid(True, lw=0.4)
ax.xaxis.grid(True, lw=0.4)
plt.tight_layout()
plt.savefig('./figures/Yearly_Accidents_MA.pdf', format='pdf', dpi=100)
plt.show()