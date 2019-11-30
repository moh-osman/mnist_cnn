#!/usr/bin/env python
# coding: utf-8

# In[5]:


# CV/Train/Test

import tensorflow.keras as tfk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout,Activation, Dense, Activation, Input, MaxPooling2D

# Model 1
def m1():
    model = Sequential([
        Conv2D(32,(3,3),input_shape=(28,28,1)),
        Flatten(),
        Dense(10)
    ])
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

# Model 2
def m2():
    model = Sequential([
        Conv2D(32,(3,3),input_shape=(28,28,1),activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(10,activation="relu")
    ])
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

# Model 3
def m3():
    model = Sequential([
        Conv2D(32,(3,3),input_shape=(28,28,1),activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64,(3,3),input_shape=(28,28,1),activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(10,activation="relu")
    ])
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

# K-fold algo
def kfold(xt,yt,m,k):
    div = int(np.floor(xt.shape[0]/k))
    loss = 0
    for i in range(0,k):
        print((i+1),"out of",k)
        md = m()
        xt_valid = xt[div*i:div*(i+1)]
        xt_train = np.concatenate((xt[0:div*i], xt[div*(i+1):]))

        yt_valid = yt[div*i:div*(i+1)]
        yt_train = np.concatenate((yt[0:div*i], yt[div*(i+1):]))
        
        md.fit(xt_train,yt_train,epochs=2)
        loss += md.evaluate(xt_valid,yt_valid)[0]
        
    return loss/k

# load data
mnist = tfk.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Reshape data to 4D for model
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# Standardize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Recatagorize labels
y_train = tfk.utils.to_categorical(y_train, 10)
y_test = tfk.utils.to_categorical(y_test, 10)

# Perform 3-fold cross-validation for each model
# and return average loss for each
print("3-Fold Cross Validation:")
kl = [kfold(x_train,y_train,m1,3), kfold(x_train,y_train,m2,3), kfold(x_train,y_train,m3,3)]

# Instantiate new models
md1 = m1()
md2 = m2()
md3 = m3()

# Train them on training data
print("Training model:")
md1.fit(x_train,y_train,epochs=2)
md2.fit(x_train,y_train,epochs=2)
md3.fit(x_train,y_train,epochs=2)

# Test on hold-out data and store losses
print("Testing model:")
l1 = md1.evaluate(x_test,y_test)
l2 = md2.evaluate(x_test,y_test)
l3 = md3.evaluate(x_test,y_test)


# In[16]:


# Graphing and plotting losses

import matplotlib.pyplot as plt
import numpy as np

# Confidence values
cf1 = 0.01 # 97% Simultaneous
cf2 = 0.05 # 95% One-at-a-time

# Size of hold-out set
mv = 10000

x = ["CNN1","CNN2","CNN3"]
y = [l1[0],l2[0],l3[0]]

e1 = np.sqrt(np.log(2.0/cf1)/(mv*2.0))
err1 = [[e1,e1,e1],[e1,e1,e1]]

e2 = np.sqrt(np.log(2.0/cf2)/(mv*2.0))
err2 = [[e2,e2,e2],[e2,e2,e2]]

# Confidence bounds for true risk
plt.figure()
plt.errorbar(x, y, yerr=err1, fmt='o')
plt.suptitle("Simultaneous 97% confidence intervals:",fontsize=14)
plt.ylim(0)
plt.show()

print("Simultaneous 97% Conf. Interval values for each model:")
print("CNN1:",([l1[0] - e1,l1[0] + e1]))
print("CNN2:",([l2[0] - e1,l2[0] + e1]))
print("CNN3:",([l3[0] - e1,l3[0] + e1]))

plt.figure()
plt.errorbar(x, y, yerr=err2, fmt='o')
plt.suptitle("One-at-a-time 95% confidence intervals:",fontsize=14)
plt.ylim(0)
plt.show()

print("One-at-a-time 95% Conf. Interval values for each model:")
print("CNN1:",([l1[0] - e2,l1[0] + e2]))
print("CNN2:",([l2[0] - e2,l2[0] + e2]))
print("CNN3:",([l3[0] - e2,l3[0] + e2]))

# Empirical loss from 3F CV
plt.figure()
plt.suptitle("Empirical loss from 3-fold cross-validation", fontsize=14)
plt.bar(x=x,height=kl)
plt.show()

print("Cross-validation losses for each model:")
print("CNN1:",kl[0])
print("CNN2:",kl[1])
print("CNN3:",kl[2])

# Empirical loss from hold-out
plt.figure()
plt.suptitle("Empirical loss from hold-out set", fontsize=14)
plt.bar(x=x,height=[l1[0], l2[0], l3[0]])
plt.show()

print("Hold-out set losses for each model:")
print("CNN1:",l1[0])
print("CNN2:",l2[0])
print("CNN3:",l3[0])


# In[ ]:




