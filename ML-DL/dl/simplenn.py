from keras.models import Sequential
from tensorflow.keras.layers import Dense
#dense means every single noruons connected to everysingle nerouns in the next layer
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])


model = Sequential()
model.add(Dense(4,input_dim =2, activation = 'sigmoid'))
#the first one is the numbers of hidden layer and the second one is the input dimention and the last one 
#as its obvious is activation func, so that was for input 
model.add(Dense(1,input_dim =4,activation = 'sigmoid'))
#this is like this because first we have one output noroun and second before the output layer we 
#have 4 hidden layer
print(model.weights)

model.compile( loss ='mean_squared_error', optimizer ='adam', metrics =['binary_accuracy'])
model.fit(x,y,epochs= 100, verbose = 2)
