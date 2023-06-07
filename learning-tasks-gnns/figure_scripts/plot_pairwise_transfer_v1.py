# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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
cmap = mcolors.LinearSegmentedColormap.from_list("GreyMap", ['#D9E8E5', '#004030']) # Green

# Create a correlation matrix heatmap
plt.figure(figsize=(20, 4))
sns.heatmap(results[::-1], annot=False, fmt=".1f", cmap=cmap, vmin=-5, vmax=5, square=True, cbar=False)
# plt.title("Correlation Matrix")
plt.xlabel("Source", fontsize=12)
plt.ylabel("Target", fontsize=12)

# Manually annotate the heatmap values and highlight the boxes
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        if i != j:  # Exclude diagonal elements
            if round(results[i, j], 1) >= 0:
                color = cmap(results[i, j] / 5)  # Scale the value between 0 and 1
                text_color = "black"  # Modify the text color to ensure better visibility
                if round(results[i, j], 1) == 0.0:
                    plt.text(j + 0.5, i + 0.5, "0", ha="center", va="center", color=text_color, fontsize=12)  # Increase font size
                else:
                    plt.text(j + 0.5, i + 0.5, f"{round(results[i, j], 1)}", ha="center", va="center", color=text_color, fontsize=12)  # Increase font size
            else:
                color = '#7A0000'  # Dull red color
                text_color = "white"  # Modify the text color to ensure better visibility
                if round(results[i, j], 1) == 0.0:
                    plt.text(j + 0.5, i + 0.5, "0", ha="center", va="center", color=cmap(results[i, j] / 5), fontsize=12)  # Increase font size
                else:
                    plt.text(j + 0.5, i + 0.5, f"{round(results[i, j], 1)}", ha="center", va="center", color=text_color, fontsize=12)  # Increase font size
        else:
            color = '#F5F5F5'  # Very light color for diagonal elements
            text_color = "black"  # Modify the text color to ensure better visibility
            plt.text(j + 0.5, i + 0.5, f"{round(results[i, j], 1)}", ha="center", va="center", color=text_color, fontsize=12)  # Increase font size

        rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color, edgecolor=color)  # Specify color for the rectangle patch
        plt.gca().add_patch(rect)


# Set custom x-axis tick labels
tick_labels = ["DE", "IA", "IL", "MA", "MD", "MN", "MT", "NV"]
plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels)
plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels)
# plt.xlim(0, len(tick_labels))
# plt.ylim(0, len(tick_labels))

# Save the figure
plt.savefig("pairwise_transfer_results.png", dpi=300)

# Show the figure
plt.show()
