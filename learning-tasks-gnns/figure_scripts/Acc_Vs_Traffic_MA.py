





import seaborn as sns
import matplotlib.pyplot as plt






acc_count = [3811.0,
 5735.0,
 7152.0,
 9653.0,
 10537.0,
 10444.0,
 13665.0,
 11980.0,
 13170.0,
 16183.0,
 13618.0,
 17802.0,
 19750.0,
 22203.0,
 19860.0,
 26430.0,
 28893.0,
 37418.0,
 47767.0,
 74679.0]


aadt_bins = [6891.0,
 17705.5,
 26609.5,
 35409.0,
 44192.0,
 53918.5,
 64554.5,
 75497.5,
 87593.0,
 100725.0,
 115052.5,
 132087.0,
 151931.5,
 174521.5,
 200117.0,
 229674.5,
 271385.0,
 345547.0,
 570706.5,
 3745554.5]





# Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(height=acc_count,x = list(range(len(acc_count))))
plt.xlabel('AADT Bin')
plt.ylabel('acc_count')
plt.title('Histogram of acc_count by AADT (Quantile-based Bins)')
plt.xticks(list(range(len(acc_count))),aadt_bins,rotation=70)
plt.show()































