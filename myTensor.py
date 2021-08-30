import numpy as np
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from PIL import Image
#tf.disable_v2_behavior()


#t = tarfile.open("C:/Users/mic/.keras/datasets/iris_training.csv.tar.gz")
#t.extractall("./")

cach_dir="C:/Users/mic/.keras/datasets"
train_url="http://download.tensorflow.org/data/iris_training.csv"
train_path=tf.keras.utils.get_file(train_url.split('/')[-1], train_url, cach_dir)
iris=pd.read_csv(train_path)
#print(iris.head())

irisNp = np.array(iris)
print(irisNp[:,2])
plt.scatter(irisNp[:,2],irisNp[:,3],c=irisNp[:,4], cmap='brg')

boston_housing=tf.keras.datasets.boston_housing
(train_x, train_y),(test_x, test_y) = boston_housing.load_data()
print(np.shape(train_x))
#plt.scatter(train_x[:,5], train_y)
#plt.show()



"""
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z
"""

"""
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")
"""

'''
coefficient = np.array([[1.],[-20.],[25.]])
print(coefficient)

w=tf.Variable(0, dtype=tf.float32)
x=tf.placeholder(tf.float32,[3,1])
#cost = tf.add(tf.add(w**2, tf.multiply(-10.,w)), 25)
#cost = w**2 - 10*w +25
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]


train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init=tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print(session.run(w))
    session.run(train, feed_dict={x:coefficient})
    print(session.run(w))
    for i in range(1,1000):
        session.run(train, feed_dict={x:coefficient})
        print(session.run(w))
'''
