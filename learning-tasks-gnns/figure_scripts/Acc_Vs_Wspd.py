








import seaborn as sns
import matplotlib.pyplot as plt






acc_count = [199.5,
 0.0,
 369.65238095238095,
 8201.82731990232,
 47769.45073398823,
 124988.29943251194,
 131764.15826257077,
 125823.56228077479,
 96660.4640900766,
 100401.27929015429,
 94802.55981657232,
 63682.35494227994,
 23983.32778887779,
 6343.988095238095,
 1443.625,
 594.3833333333333,
 10.066666666666666,
 147.52619047619046,
 40.833333333333336,
 138.36904761904762]


bins = [0.85,
 2.6,
 4.32,
 6.06,
 7.78,
 9.52,
 11.24,
 12.98,
 14.7,
 16.44,
 18.16,
 19.9,
 21.62,
 23.35,
 25.08,
 26.82,
 28.54,
 30.28,
 32.0,
 33.74]





# Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(height=acc_count,x = list(range(len(acc_count))))
plt.xlabel('Wspd Bin')
plt.ylabel('acc_count')
plt.title('Histogram of acc_count by Wspd')
plt.xticks(list(range(len(acc_count))),bins)
plt.show()































