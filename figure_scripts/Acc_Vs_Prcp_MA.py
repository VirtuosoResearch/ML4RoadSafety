








import seaborn as sns
import matplotlib.pyplot as plt






acc_count = [9786.261227661227,
 52474.43492895993,
 129412.53854756354,
 174383.94507298258,
 150765.3577214452,
 78725.79841131091,
 48399.34787573537,
 20638.32617660118,
 11395.598076923077,
 7082.733261183261,
 4615.460714285714,
 1864.9914141414142,
 2160.6579143079143,
 60.833333333333336,
 532.225,
 956.0,
 1327.9743145743146,
 20.0,
 0.0,
 159.0]


bins = [11.71,
 35.85,
 59.75,
 83.65,
 107.55,
 131.45,
 155.35,
 179.25,
 203.15,
 227.05,
 250.95,
 274.85,
 298.75,
 322.65,
 346.55,
 370.45,
 394.35,
 418.25,
 442.15,
 466.05]





# Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(height=acc_count,x = list(range(len(acc_count))))
plt.xlabel('Prcp Bin')
plt.ylabel('acc_count')
plt.title('Histogram of acc_count by Prcp')
plt.xticks(list(range(len(acc_count))),bins)
plt.show()































