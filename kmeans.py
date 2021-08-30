import numpy as np
import math
import matplotlib.pyplot as plt
import time
from sklearn.base import BaseEstimator

xset=np.random.randint(0,100,(1,100))
#xset=np.hstack( (xset,np.random.randint(50,100,(1,50))) )
yset=np.random.randint(0,100,(1,100))
#yset=np.hstack( (yset,np.random.randint(50,100,(1,50))) )
plt.axis([0, 101, 0, 101])
plt.scatter(xset, yset)

print(xset[0][0])

rx=np.random.randint(0,100,(1,2))
ry=np.random.randint(0,100,(1,2))

lrx=rx
lry=ry


#plt.scatter(rx, ry,marker="o",c=['darkblue','darkred'])

for itr in range(1,1000):
    plt.clf()
    plt.title("No.%d interation" %itr)
    plt.scatter(rx, ry,marker="o",c=['darkblue','darkred'])# 
    plt.ion()

    redx=[]
    redy=[]
    bluex=[]
    bluey=[]
    c=[]

    for i in range(0,100):
        t1=math.pow((xset[0][i]-rx[0][0]),2)+math.pow((yset[0][i]-ry[0][0]),2)
        t2=math.pow((xset[0][i]-rx[0][1]),2)+math.pow((yset[0][i]-ry[0][1]),2)
        #print("point %d t1=%d, t2=%d" %(i,t1,t2))
        if(t1>t2):
            c.append(-1)
            plt.scatter(xset[0][i], yset[0][i],marker="*",c=['red'])
            #print("point %d belong to red")
            redx.append(xset[0][i])
            redy.append(yset[0][i])
        else:
            c.append(1)
            plt.scatter(xset[0][i], yset[0][i],marker="*",c=['blue'])
            #print("point %d belong to blue")
            bluex.append(xset[0][i])
            bluey.append(yset[0][i])
    lrx=np.array(rx,copy=True)
    lry=np.array(ry,copy=True)

    print("last blue=(%d,%d)" %(lrx[0][0],lry[0][0]))
    print("last red=(%d,%d)" %(lrx[0][1],lry[0][1]))

    rx[0][0]=sum(bluex)/len(bluex)
    ry[0][0]=sum(bluey)/len(bluey)
    
    rx[0][1]=sum(redx)/len(redx)
    ry[0][1]=sum(redy)/len(redy)

    print("new blue=(%d,%d)" %(rx[0][0],ry[0][0]))
    print("new red=(%d,%d)" %(rx[0][1], ry[0][1]))

    plt.show()
    plt.pause(1)

    print((lrx==rx).all())

    if ((lrx==rx).all()):
        if (lry==ry).all(): 
            break

plt.title("Trainning Complete")
plt.ioff()
plt.show()


    