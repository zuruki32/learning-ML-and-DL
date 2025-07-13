from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers  import Adam

path =r'E:/Udemy/learning-ML-and-DL/dataset/smiles_dataset/training_set/'
path_test =r'E:/Udemy/learning-ML-and-DL/dataset/smiles_dataset/test_set/'
pixel_intensities =[]
test_pixels =[]
label =[]
label_test =[]

for filename in os.listdir(path):
    #print(filename)
    image = Image.open(path+filename).convert('1')
    #with converting to 1 its going to be gray scale(intensity between 0-255)  
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] =='happy':
        label.append([1,0])
    elif filename[0:3] == 'sad':
        label.append([1,0])

for filename in os.listdir(path_test):
    image_test = Image.open(path_test+filename).convert('1')
    #with converting to 1 its going to be gray scale(intensity between 0-255)  
    test_pixels.append(list(image_test.getdata()))
    if filename[0:5] =='happy':
        label_test.append([1,0])
    elif filename[0:3] == 'sad':
        label_test.append([1,0])
"""
for p in pixel_intensities:
    print(p)
"""
pixel_intensities = np.array(pixel_intensities)
test_pixels = np.array(test_pixels)
pixel_intensities = pixel_intensities/255
test_pixels = test_pixels/255
#print(pixel_intensities.shape)
label_test = np.array(label_test)
label = np.array(label)

model= Sequential()
model.add(Dense(1024,input_dim =1024, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

optimizer = Adam(learning_rate = 0.001)
model.compile(loss ='categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])

model.fit(pixel_intensities,label,epochs= 1000, batch_size=32, verbose = 2)
print(model.predict(test_pixels).round())