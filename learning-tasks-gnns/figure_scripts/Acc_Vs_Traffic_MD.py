






import seaborn as sns
import matplotlib.pyplot as plt






acc_count = [868.0,
 943.0,
 911.0,
 1540.0,
 1569.0,
 1281.0,
 1832.0,
 1764.0,
 1660.0,
 1815.0,
 1384.0,
 1660.0,
 1529.0,
 2355.0,
 2132.0,
 2492.0,
 2416.0,
 2066.0,
 2990.0,
 4152.0]


aadt_bins = [2814.92,
 7399.78,
 11507.08,
 15795.22,
 20510.62,
 26061.72,
 32493.2,
 39355.7,
 46543.18,
 55547.58,
 65849.32,
 77575.73,
 91496.75,
 109173.75,
 131713.78,
 158698.08,
 196641.38,
 270098.47,
 457393.32,
 85031678.03]





# Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(height=acc_count,x = list(range(len(acc_count))))
plt.xlabel('AADT Bin')
plt.ylabel('acc_count')
plt.title('Histogram of acc_count by AADT (Quantile-based Bins)')
plt.xticks(list(range(len(acc_count))),aadt_bins,rotation=70)
plt.show()































