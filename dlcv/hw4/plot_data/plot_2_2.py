import numpy as np   
import sys
import matplotlib.pyplot as plt 
data1 = []
data2 = []
data3 = []
data4 = []
with open(sys.argv[1],'r') as f :
    f.readline()
    for i in range(1000):
        a = f.readline().replace('\n','').split(',')[2]
        data1.append(float(a))

with open(sys.argv[2],'r') as f :
    f.readline()
    for i in range(1000):
        a = f.readline().replace('\n','').split(',')[2]
        data2.append(float(a))

with open(sys.argv[3],'r') as f :
    f.readline()
    for i in range(1000):
        a = f.readline().replace('\n','').split(',')[2]
        data3.append(float(a))


with open(sys.argv[4],'r') as f :
    f.readline()
    for i in range(1000):
        a = f.readline().replace('\n','').split(',')[2]
        data4.append(float(a))

data1 = np.array(data1).reshape(-1,1)
data2 = np.array(data2).reshape(-1,1)
data3 = np.array(data3).reshape(-1,1)
data4 = np.array(data4).reshape(-1,1)

length = data1.shape[0]

x = [ i for i in range(1,length+1)]
x = np.array(x).reshape(-1,1)

#plt.figure(figsize=(60,60))
plt.figure()
plt.subplot(2,1,1)
plt.plot(x,data1,label='real')
plt.plot(x,data2,label='fake')
#plt.xlabel(sys.argv[4])
#plt.ylabel(sys.argv[5])
plt.title(sys.argv[5])
plt.xlim(0,length)
plt.legend(loc='upper right')
#plt.ylim(min(data),max(data))

plt.subplot(2,1,2)
plt.plot(x,data3,label='real')
plt.plot(x,data4,label='fake')
#plt.xlabel(sys.argv[4])
#plt.ylabel(sys.argv[7])
plt.title(sys.argv[6])
plt.xlim(0,length)
plt.legend(loc='lower right')
plt.tight_layout()
'''
plt.subplot(2,2,3)
plt.plot(x,data3)
plt.xlabel(sys.argv[9])
#plt.ylabel(sys.argv[9])
plt.title(sys.argv[7])
plt.xlim(0,length)

plt.subplot(2,2,4)
plt.plot(x,data4)
plt.xlabel(sys.argv[9])
#plt.ylabel(sys.argv[11])
#plt.title(sys.argv[8])
plt.xlim(0,length)
'''
plt.savefig('fig2_2'+'.jpg')
#plt.show()

