#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf


# In[18]:


npz = np.load('AudioBook_train_data.npz')

train_input = npz['inputs'].astype(np.float)
train_target= npz['targets'].astype(np.float)

npz = np.load('AudioBook_validation_data.npz')

validation_input = npz['inputs'].astype(np.float)
validation_target= npz['targets'].astype(np.float)

npz = np.load('AudioBook_test_data.npz')

test_input = npz['inputs'].astype(np.float)
test_target= npz['targets'].astype(np.float)


# In[30]:


# buildin the model
input_size = 10
output_size = 2
hidden_layer_size = 100
activation_function = 'elu'
model = tf.keras.Sequential(layers=[
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),        
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),        
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),
        tf.keras.layers.Dense(hidden_layer_size,activation=activation_function),

        tf.keras.layers.Dense(output_size,activation='softmax'),
])

batch_size = 50
max_epochs = 100
callbacks = tf.keras.callbacks.EarlyStopping(patience=4)
model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_input,
          train_target,
          batch_size = batch_size,
          epochs = max_epochs,
          callbacks =[callbacks],
          validation_data=(validation_input,validation_target),
          verbose=0)


# In[29]:


## testing the data
total_loss,total_accuracy = model.evaluate(test_input,test_target,verbose=0)
print("Total Loss = ",total_loss," Total Accuracy : ",total_accuracy)


# In[ ]:




