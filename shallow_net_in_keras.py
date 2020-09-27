#!/usr/bin/env python
# coding: utf-8

# # Shallow Neural Network in Keras

# Build a shallow neural network to classify MNIST digits

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/shallow_net_in_keras.ipynb)

# #### Load dependencies

# In[99]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import time


# #### Load data

# In[100]:


(X_train, y_train), (X_valid, y_valid) = mnist.load_data()


# In[101]:


X_train.shape


# In[102]:


y_train.shape


# In[103]:


y_train[0:12]


# In[104]:


plt.figure(figsize=(5,5))
for k in range(12):
    plt.subplot(3, 4, k+1)
    plt.imshow(X_train[k], cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[105]:


X_valid.shape


# In[106]:


y_valid.shape


# In[107]:


plt.imshow(X_valid[0], cmap='Greys')


# In[108]:


X_valid[0]


# In[109]:


y_valid[0]


# #### Preprocess data

# In[110]:


X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')


# In[111]:


X_train /= 255
X_valid /= 255


# In[112]:


X_valid[0]


# In[113]:


n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)


# In[114]:


y_valid[0]


# #### Design neural network architecture

# In[115]:


model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))


# In[116]:


model.summary()


# In[117]:


(64*784)


# In[118]:


(64*784)+64


# In[119]:


(10*64)+10


# #### Configure model

# In[120]:


model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])


# 
# #### Train!

# In[121]:


inicio = time.time()
model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_valid, y_valid))
final = time.time()


# In[122]:


model.evaluate(X_valid, y_valid)


# In[123]:


tiempo = final-inicio


# In[124]:


print (inicio)


# In[125]:


print(final)


# In[126]:


print(tiempo)


# In[ ]:





# In[ ]:





# In[ ]:




