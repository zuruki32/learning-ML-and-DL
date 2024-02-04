import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score

x_blue = np.array([0.3,0.5,1,1.4,1.7,2])
y_blue = np.array([1,4.5,2.3,1.9,8.9,4.1])

x_red = np.array([3.3,3.5,4,4.4,5.7,6])
y_red = np.array([7,1.5,6.3,1.9,2.9,7.1])

X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
Y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
# 0  = blue 1 = red
plt.plot(x_blue,y_blue, 'ro',color = 'blue')
plt.plot(x_red,y_red, 'ro',color = 'red')
plt.plot(3,5, 'ro',color = 'green',markersize = 15)
plt.axis([-0.5,10,0.5,10])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,Y)
predict = classifier.predict(np.array([[5,5]]))
print(predict)
plt.show()

credit = r'C:\Users\apply-system\Documents\code\credit_data.csv'
creditdata = pd.read_csv(credit)
feature = creditdata[["income","age","loan"]]
target = creditdata.default
X = np.array(feature).reshape(-1,3)
Y =  np.array(target)
X = preprocessing.MinMaxScaler().fit_transform(X)
cross_valid_Score = []
for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    cross_valid_Score.append(scores.mean())
print("optimal k :",np.argmax(cross_valid_Score))    

feature_train,feature_test,target_train,target_test = train_test_split(X,Y,test_size= 0.3)
model = KNeighborsClassifier(n_neighbors=32)
fitmodel = model.fit(feature_train,target_train)
prediction = model.predict(feature_test)
print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))

