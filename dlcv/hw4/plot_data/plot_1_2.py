import numpy as np   
import matplotlib.pyplot as plt 
import sys

data1 = []
data2 = []
with open(sys.argv[1],'r') as f :
    f.readline()
    for i in range(int(sys.argv[3])):
        a = f.readline().replace('\n','').split(',')[2]
        data1.append(float(a))

with open(sys.argv[2],'r') as f :
    f.readline()
    for i in range(int(sys.argv[3])):
        a = f.readline().replace('\n','').split(',')[2]
        data2.append(float(a))

data1 = np.array(data1).flatten()
data2 = np.array(data2).flatten()


length = data1.shape[0]
#length = len(data1)
x = [ i for i in range(1,length+1)]
x = np.array(x).reshape(-1,1)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x,data1)
#plt.xlabel(sys.argv[4])
plt.ylabel(sys.argv[5])
plt.title(sys.argv[6])
plt.xlim(0,length)
#plt.ylim(min(data),max(data))

plt.subplot(2,1,2)
plt.plot(x,data2)
#plt.xlabel(sys.argv[4])
plt.ylabel(sys.argv[7])
plt.title(sys.argv[8])
plt.xlim(0,length)
plt.tight_layout()
plt.savefig('fig1_2'+'.jpg')
#plt.show()
