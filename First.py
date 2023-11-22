import statistics
import numpy as np
l1=[115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4,195.5]
a=len(l1)
s=sum(l1)
mean=(s/a)
print("MEAN :",mean)

print("Using Inbuilt function of Mean ")
x=statistics.mean(l1)
print(x)
print("****************************")
#median
values_sorted = sorted(l1)
n = len(values_sorted)
if n % 2 == 0:
    median = (values_sorted[n//2 - 1] + values_sorted[n//2]) / 2
else:
    median = values_sorted[n//2]
print( "MEDIAN :",median)

print("Using Inbuilt function of Median")
y=statistics.median(l1)
print(y)
print("****************************")
#mode
mode_dict = {}
for value in l1:
    if value in mode_dict:
        mode_dict[value] += 1
    else:
        mode_dict[value] = 1
mode = max(mode_dict, key=mode_dict.get)
print("Mode:", mode)
print("Using Inbuilt function of Mode")
z=statistics.mode(l1)
print(z)
print("****************************")
#standard deviation
mean = sum(l1) / len(l1)
variance = sum((value - mean) ** 2 for value in l1) / len(l1)
std_dev = variance ** 0.5
print("Standard Deviation:", std_dev)
print("Using Inbuilt function of Standard Deviation ")
d=statistics.stdev(l1)
print(d)
print("****************************")
#Variance
variance = sum((value - mean) ** 2 for value in l1) / len(l1)
print("Variance:", variance)
print("Using Inbuilt function of Variance")
v=np.var(l1)
print(v)
print("****************************")
#min-max normalization 
min_value = min(l1)
max_value = max(l1)
min_max_norm = [(value - min_value) / (max_value - min_value) for value in l1]
print("Min-Max Normalization:", min_max_norm)
print("Using Inbuilt function of Min-Max Normalization ")
min_max_norm = (l1 - np.min(l1)) / ( np.max(l1) - np.min(l1))
print("Min-Max Normalization:", min_max_norm)
print("****************************")
#standardization
mean = sum(l1) / len(l1)
variance = sum((value - mean) ** 2 for value in l1) / len(l1)
standardization = [(value - mean) / variance ** 0.5 for value in l1]
print("Standardization:", standardization)
print("Using Inbuilt function of standardization ")
standardization = (l1 - np.mean(l1)) / np.std(l1)
print("Standardization:", standardization)