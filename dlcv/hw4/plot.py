import numpy as np   
import matplotlib.pyplot as plt 
import sys
data = np.load(sys.argv[1]).reshape(-1,1)
length = data.shape[0]

x = [ i for i in range(1,length+1)]
x = np.array(x).reshape(-1,1)

plt.figure()
plt.plot(x,data)
plt.xlabel(sys.argv[2])
plt.ylabel(sys.argv[3])
plt.title(sys.argv[4])
plt.xlim(0,length)
#plt.ylim(min(data),max(data))
plt.savefig(sys.argv[5]+'.png')
plt.show()

