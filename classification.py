from io import DEFAULT_BUFFER_SIZE
import numpy as np
import struct
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt

IMAGE_TR_FILE_NAME='C:/Mic/projects/AD/machinelearning/dataset/train-images.idx3-ubyte'
LABEL_TR_FILE_NAME='C:/Mic/projects/AD/machinelearning/dataset/train-labels.idx1-ubyte'

IMAGE_TS_FILE_NAME='C:/Mic/projects/AD/machinelearning/dataset/t10k-images.idx3-ubyte'
LABEL_TS_FILE_NAME='C:/Mic/projects/AD/machinelearning/dataset/t10k-labels.idx1-ubyte'

def load_images(file_name):
    binfile=open(file_name,'rb')
    buffers=binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    #print("%d,%d,%d,%d" %(magic,num,rows,cols))
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    return images

def load_labels(file_name):
    ##   打开文件
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据    
    buffers = binfile.read()
    ##   读取label文件前2个整形数字，label的长度为num
    magic,num = struct.unpack_from('>II', buffers, 0) 
    ##   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    ##   关闭文件
    binfile.close()
    ##   转换为一维数组
    labels = np.reshape(labels, [num])
    return labels  

def showImage(buffer,index=1):
    p=plt.subplot(10,10,index)
    p.imshow(buffer, cmap = matplotlib.cm.binary, interpolation="nearest")
    p.axis("off")
    plt.ion()
    plt.show()
    plt.pause(1)
    

ims=load_images(IMAGE_TR_FILE_NAME)
labs=load_labels(LABEL_TR_FILE_NAME)

print(ims)

y_test_5 = (labs == 5)
#print(y_test_5)

"""
for i in range(1,26):
    some_digit = ims[i]
    some_digit_image = some_digit.reshape(28, 28)
    p=plt.subplot(5,5,i)
    p.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    p.axis("off")
plt.show()
"""

#sgd_clf = SGDClassifier(random_state=42)
#sgd_clf.fit(ims, y_test_5)

ims_ts=load_images(IMAGE_TS_FILE_NAME)
labs_ts=load_labels(LABEL_TS_FILE_NAME)

j=1
for i in range(1,1000):
    some_digit = ims_ts[i]
    some_digit_image = some_digit.reshape(28, 28)
    res=sgd_clf.predict([some_digit])
    #print(res)
    #plt.title(str(res))
    if res[0]==True:
        showImage(some_digit_image, j)
        j+=1
