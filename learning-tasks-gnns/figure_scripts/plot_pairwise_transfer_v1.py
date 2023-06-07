# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import rc
import matplotlib as mpl


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


results = np.array([[0.0, 0.17, 0.56, 0.38, 0.18, -0.01, -0.01, 0.29],
                    [0.88, 0.0, 1.2, 1.01, 0.57, 0.77, 0.83, 1.98],
                    [0.56, -0.08, 0.0, 0.37, 0.29, -0.13, 0.14, 0.14],
                    [0.89, 0.28, 0.68, 0.0, -0.54, -0.32, -0.05, 0.26],
                    [0.33, 0.11, 0.58, 0.66, 0.0, 0.06, 0.27, 0.27],
                    [3.04, 3.03, 3.49, 2.94, 3.44, 0.0, 2.95, 3.14],
                    [0.08, 0.15, -0.23, -0.01, 0.06, 0.29, 0.0, 0.19],
                    [0.95, -4.37, -3.89, 1.17, 1.14, 0.52, 0.2, 0.0]])

# Create a custom colormap
# cmap = mcolors.LinearSegmentedColormap.from_list("GreyMap", ['#808080','#000000']) # Black
cmap = mcolors.LinearSegmentedColormap.from_list("GreyMap", ['#D9E8E5', 'mediumseagreen']) # Green

# Create a correlation matrix heatmap
plt.figure(figsize=(4.5, 4.5))
sns.heatmap(results[::-1], annot=False, fmt=".1f", cmap=cmap, vmin=-5, vmax=5, square=True, cbar=False)
# plt.title("Correlation Matrix")
plt.xlabel(r"$\mathrm{Source}$", fontsize=15)
plt.ylabel(r"$\mathrm{Target}$", fontsize=15)

# Manually annotate the heatmap values and highlight the boxes
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        if i != j:  # Exclude diagonal elements
            if round(results[i, j], 1) >= 0:
                color = cmap(results[i, j] / 5)  # Scale the value between 0 and 1
                text_color = "black"  # Modify the text color to ensure better visibility
                if round(results[i, j], 1) == 0.0:
                    plt.text(j + 0.5, i + 0.5, "0", ha="center", va="center", color=text_color, fontsize=14)  # Increase font size
                else:
                    plt.text(j + 0.5, i + 0.5, f"{round(results[i, j], 1)}", ha="center", va="center", color=text_color, fontsize=14)  # Increase font size
            else:
                color = 'tomato'  # Dull red color
                text_color = "white"  # Modify the text color to ensure better visibility
                if round(results[i, j], 1) == 0.0:
                    plt.text(j + 0.5, i + 0.5, "0", ha="center", va="center", color=cmap(results[i, j] / 5), fontsize=14)  # Increase font size
                else:
                    plt.text(j + 0.5, i + 0.5, f"{round(results[i, j], 1)}", ha="center", va="center", color=text_color, fontsize=14)  # Increase font size
        else:
            color = '#F5F5F5'  # Very light color for diagonal elements
            text_color = "black"  # Modify the text color to ensure better visibility
            plt.text(j + 0.5, i + 0.5, f"{round(results[i, j], 1)}", ha="center", va="center", color=text_color, fontsize=14)  # Increase font size

        rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color, edgecolor=color)  # Specify color for the rectangle patch
        plt.gca().add_patch(rect)


# Set custom x-axis tick labels
tick_labels = [r"$\mathrm{DE}$", r"$\mathrm{IA}$", r"$\mathrm{IL}$", r"$\mathrm{MA}$", r"$\mathrm{MD}$", r"$\mathrm{MN}$", r"$\mathrm{MT}$", r"$\mathrm{NV}$"]
plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, fontsize=15)
plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, fontsize=15)
# plt.xlim(0, len(tick_labels))
# plt.ylim(0, len(tick_labels))
plt.tight_layout()

# Save the figure
plt.savefig("pairwise_transfer_results.pdf", dpi=300)

# Show the figure
#plt.show()
