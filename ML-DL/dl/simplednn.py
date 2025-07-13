import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
y = np.array([[0],[1],[1],[0]],"float32")

model = Sequential()
model.add(Dense(16,input_dim =2,activation = 'relu'))
model.add(Dense(16,input_dim =16,activation = 'relu'))
model.add(Dense(16,input_dim =16,activation = 'relu'))
model.add(Dense(16,input_dim =16,activation = 'relu'))
model.add(Dense(16,input_dim =16,activation = 'relu'))
model.add(Dense(16,input_dim =16,activation = 'relu'))
model.add(Dense(16,input_dim =16,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss ='mean_squared_error',
             optimizer ='adam',
             metrics =['binary_accuracy'])

model.fit(x,y,epochs =500, verbose =2)
print(model.predict(x).round())
