from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers  import Adam
 
iris_data =load_iris()
features = iris_data.data
label = iris_data.target.reshape(-1,1)
#-1: "Infer this dimension from the data"
#1: Force the second dimension to be 1

#so we use one hot encoded cause  the classes cant be 0,1,2 instead we use something like (1,0,0) and etc...

encoder = OneHotEncoder()
target = encoder.fit_transform(label).toarray()

#The .toarray() method is used when converting a sparse matrix into a dense NumPy array, typically after one-hot encoding categorical labels

train_features, test_features, train_target, test_target = train_test_split(features,target,test_size=0.2 )
model = Sequential()
model.add(Dense(10,input_dim =4, activation = 'sigmoid'))
model.add(Dense(3, activation = 'softmax'))

optimizer = Adam(learning_rate = 0.001)
model.compile(loss ='categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])

model.fit(train_features, train_target, epochs =1000, batch_size = 20,verbose =2)
results =model.evaluate(test_features,test_target)
#enables parallel processing
print("training is finished")
print(model.predict(x).round())