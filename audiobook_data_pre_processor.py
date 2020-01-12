#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
from sklearn import preprocessing


# In[42]:


raw_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')

unscaled_input_data = raw_data[:,1:11]
unscaled_target_data = raw_data[:,-1]


# In[43]:


##balancing the data
total_number_of_samples = len(unscaled_input_data)
number_of_target_one = float(np.sum(unscaled_target_data))
number_of_target_zero = 0
indices_to_remove = []

for i in range(total_number_of_samples):
    if(unscaled_target_data[i] == 0):
        number_of_target_zero +=1
        if(number_of_target_zero > number_of_target_one):
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_input_data,indices_to_remove,axis=0)
unscaled_targets_equal_priors = np.delete(unscaled_target_data,indices_to_remove,axis=0)


# In[44]:


# Scalling the inputs and targets
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)


# In[45]:


shuffeled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffeled_indices)

shuffeled_inputs = scaled_inputs[shuffeled_indices]
shuffeled_targets = unscaled_targets_equal_priors[shuffeled_indices]


# In[48]:


# deviding the dataset in the as split train and test
TRAIN_SIZE = int(0.8 * len(shuffeled_inputs))
VALIDATION_SIZE = int(0.1 * len(shuffeled_inputs))
TEST_SIZE= len(shuffeled_inputs) - TRAIN_SIZE - VALIDATION_SIZE

train_input = shuffeled_inputs[:TRAIN_SIZE]
train_targets = shuffeled_targets[:TRAIN_SIZE]

validation_inputs= shuffeled_inputs[TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE]
validation_targets= shuffeled_targets[TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE]

test_inputs = shuffeled_inputs[TRAIN_SIZE+VALIDATION_SIZE:]
test_targets = shuffeled_targets[TRAIN_SIZE+VALIDATION_SIZE:]


# In[49]:


## saving the file in npz format
np.savez("AudioBook_train_data",inputs = train_input,targets=train_targets)
np.savez("AudioBook_validation_data",inputs = validation_inputs,targets=validation_targets)
np.savez("AudioBook_test_data",inputs = test_inputs,targets=test_targets)


# In[ ]:




