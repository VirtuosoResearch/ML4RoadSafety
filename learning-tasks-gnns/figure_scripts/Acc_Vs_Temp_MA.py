







import seaborn as sns
import matplotlib.pyplot as plt






acc_count = [17096.0,
 11144.0,
 9860.0,
 8644.0,
 6343.0,
 4443.0,
 8859.0]


bins = [-1.15, 3.92, 7.3, 13.68, 17.38, 20.06, 22.76]





# Plot the histogram
plt.figure(figsize=(8, 6))
plt.bar(height=acc_count,x = list(range(len(acc_count))))
plt.xlabel('Temp Bin')
plt.ylabel('acc_count')
plt.title('Histogram of acc_count by Temp')
plt.xticks(list(range(len(acc_count))),bins)
plt.show()































