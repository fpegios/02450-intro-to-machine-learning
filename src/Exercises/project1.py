import numpy as np
filename = "../marketingdata.txt"
marketingdata  = np.genfromtxt(filename, delimiter=" ")
N = len(marketingdata)

# Remove rows with na values
count = 0
for i in range(N):
    if True in np.isnan(marketingdata[i,:]):
        count += 1

sortedData = np.zeros((N-count,14))
Count = 0

for i in range(N):
    if True not in np.isnan(marketingdata[i,:]):
        sortedData[Count,:] = marketingdata[i,:]
        Count += 1

for i in range(13):
    a = sortedData[:, i+1];
    print(np.mean(a))
    # print(np.var(a))
    # print(np.std(a))
