import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
"""
xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])

xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])

X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class, 1: red class

plt.plot(xBlue, yBlue, 'ro', color = 'blue')
plt.plot(xRed, yRed, 'ro', color='red')
plt.plot(3,5,'ro',color='green', markersize=15)
plt.axis([-0.5,10,-0.5,10])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,y)
predict = classifier.predict(np.array([[1,5]]))
print(predict)
plt.show()
"""
filepath= r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\credit_data.csv"
credits = pd.read_csv(filepath)
features = credits[["income","age","loan"]]
Target = credits.default
x = np.array(features).reshape(-1,3)
y = np.array(Target)

x = preprocessing.MinMaxScaler().fit_transform(x)

features_train, features_test, Target_train, Target_test = train_test_split(x,y,test_size=0.2)
model= KNeighborsClassifier(n_neighbors=20)
fited_model = model.fit(features_train,Target_train)
prediction = fited_model.predict(features_test)
print(confusion_matrix(Target_test,prediction))
print(accuracy_score(Target_test,prediction))

cross_valid_score = []
for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,x,y, cv = 10,scoring = 'accuracy')
    cross_valid_score.append(score.mean())
print("optimal k with cross_validation: ", np.argmax(cross_valid_score))    

