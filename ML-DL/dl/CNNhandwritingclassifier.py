import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train, Y_train), (X_test,Y_test) = mnist.load_data()
"""
plt.imshow(X_train[0], cmap = 'gray')
plt.title('Class' + str(Y_train[0]) )
plt.show()
#tensorflow handle format (batch,height,width,channel)
"""
features_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
features_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

features_train /= 255
features_test /= 255
#when we have more than 2 callas we need to use one hot encoder  to make them to represent of binary

target_train = to_categorical(Y_train, 10)
target_test = to_categorical(Y_test, 10)


model = Sequential()
#32 is the numbers of filters and (3x3) is the size of filters and the one is for channel numbers
model.add(Conv2D(32,(3,3), input_shape = (28,28,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#so we make it flatten so we can use feed forward neural network
model.add(Flatten())
model.add(BatchNormalization())
#this is fully connected nerual netwrok with 12 hidden neruns
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10, activation = "softmax"))

model.summary()

model.compile( loss ='categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'] )
#cause its a classification problem we used cross entropy
"""
model.fit(features_train,target_train, batch_size= 128, epochs=2,
          validation_batch_size=(features_test,target_test), verbose=1)
"""
train_generator = ImageDataGenerator( rotation_range = 7, width_shift_range = 0.05, shear_range = 0.06,
                                      height_shift_range = 0.07, zoom_range = 0.05)

test_generator = ImageDataGenerator()

train_generator = train_generator.flow(features_train, target_train, batch_size = 64)
test_generator = test_generator.flow(features_test,target_test, batch_size = 64)

model.fit(train_generator,steps_per_epoch= 60000//64, epochs=5,
          validation_data=test_generator, validation_steps=10000//64)