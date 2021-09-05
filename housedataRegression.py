from math import sqrt
import random as r
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

def bestW(XX,YY):
    AA=np.matmul(XX.T, XX)
    AAI=np.linalg.inv(AA)

    BB=np.matmul(XX.T, YY)
    W=np.matmul(AAI,BB)
    return W

def h(x, p1,p2):
    return p1 * x +p2




#trainning_data_x=np.array([1,2,3,5,6,7,8,9,10,12]).reshape(10,1)
#print(trainning_data_x)
#trainning_data_y=np.array([5,18,23,34,45,56,76,88,95,102]).reshape(10,1)

boston_housing=tf.keras.datasets.boston_housing
(train_x, train_y),(test_x, test_y) = boston_housing.load_data()

trainlen=len(train_y)

trainning_data_x=np.array(train_x[:,5]).reshape( trainlen,1 )
trainning_data_y=np.array(train_y).reshape( trainlen,1 )

#print(trainning_data_x)
print(trainning_data_x.shape)
#print(trainning_data_y)
print(trainning_data_y.shape)

###

trainning_data_x_b=np.insert( trainning_data_x, 0, 1, axis=1 )
#print(trainning_data_x_b)
print(trainning_data_x_b.shape)

w=bestW(trainning_data_x_b,trainning_data_y)
print(w)

plt1=plt.subplot(221)
plt1.scatter(trainning_data_x,trainning_data_y,marker='x', color='yellow')
yhats=np.matmul(trainning_data_x_b, w)
plt1.plot(trainning_data_x, yhats)
t=str.format("w=%f, b=%f" %(w[1],w[0]))
plt1.set_title(t)
#plt.show()

"""

alpha=0.1

t0=10
t1=10

print(bestW(trainning_data_x,trainning_data_y))

print("t0, t1=%f,%f" %(t0,t1))

m=len(trainning_data_x)
print("m=%d" %m)


for i in range(10):
    print("round: %d" %i)
    sum1=0
    sum2=0

    for j in range(0,m):
        sum1 += h(trainning_data_x[j],t0,t1) - trainning_data_y[j]
        sum2 += (h(trainning_data_x[j],t0,t1)-trainning_data_y[j])*trainning_data_x[j]
        #print("sum1=%d, sum2=%d" %(sum1, sum2))
    d1=sum1/m*alpha/2
    d2=sum2/m*alpha/2
    print("d1=%f" %d1)
    print("d2=%f" %d2)
    t0=t0-d1
    t1=t1-d2
    print("new t0=%f, new t1=%f" %(t0,t1))
    print("Loss: %f, t0:%f, t1:%f" %(trainning_data_y[j]-h(trainning_data_x[j],t0,t1), t0, t1))
    plt.scatter(trainning_data_x,trainning_data_y,marker='x', color='yellow')
    yhats=[]
    for j in range(0,m):
        yhat= h(trainning_data_x[j],t0,t1)
        yhats.append(yhat)
    #plt.plot(trainning_data_x, yhats )
    loss=h(trainning_data_x[j],t0,t1) - trainning_data_y[j]
    tx="(%s):Loss: %f, t0:%f, t1:%f"%(i, loss , t0, t1)
    #plt.ion
    #plt.text(5,5,tx)
    #plt.show()
    #plt.pause(0.01)

"""

mse=[]
alpha=0.005
np.random.seed(612)
W=tf.Variable(np.random.randn())
B=tf.Variable(np.random.randn())
print("w:%f b:%f" %(W, B))

itr=10000
for i in range(itr):

    with tf.GradientTape() as tape:
        pred =W*trainning_data_x+B
        #print("pred: %s" %pred)
        #print("y: %s" %y)
        Loss = 0.5 * tf.reduce_mean(tf.square(trainning_data_y - pred))
        #print("<%d>: Loss:%f  w:%f b:%f" %(i, Loss, W, B))
        mse.append(Loss)

        dL_dw, dL_db=tape.gradient(Loss, [W,B])
        W.assign_sub(alpha*dL_dw)
        B.assign_sub(alpha*dL_db)

plt3=plt.subplot(223)
plt3.scatter(trainning_data_x,trainning_data_y,marker='x', color='yellow')
yhats2=[]
for k in range(trainlen): 
    yh=trainning_data_x[k]*W+B
    yhats2.append(yh)
plt3.plot(trainning_data_x, yhats2)
t=str.format("w:%f b:%f" %(W, B))
plt3.set_title(t)

plt4=plt.subplot(224)
plt4.plot(range(1,itr+1), mse)
plt.show()