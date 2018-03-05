import numpy as np
filename = "marketingdata.txt"
marketingdata  = np.genfromtxt(filename, delimiter=" ")
N = len(marketingdata)

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

# Nan i 3. marital status - er der en other?
# 5 education - other?
# 6 occupation - other?
# 7 lived in the area - not living there?
# 9 persons in house - homeless?
# 12 type of home - why when there is other?
# 13 etnic - vil ikke svare?
# 14 language - vil ikke svare?

#A = sorted(set(marketingdata[:,13]))