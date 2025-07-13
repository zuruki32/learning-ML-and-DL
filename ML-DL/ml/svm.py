import  numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
"""
xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])

xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])

X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class, 1: red class

plt.plot(xBlue, yBlue, 'ro', color='blue')
plt.plot(xRed, yRed, 'ro', color='red')
plt.plot(2.5,4.5,'ro',color='green')

classifier= svm.SVC(C=1)
classifier.fit(X,y)
print(classifier.predict([[2.5,4.5]]))
plt.axis([-0.5,10,-0.5,10])
plot_decision_regions(X,y,clf=classifier,legend=2)
plt.show()
"""
iris_data = datasets.load_iris()
features = iris_data.data
target=  iris_data.target

features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=0.2)

model= svm.SVC()

param_grid = {'C':[0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200],
              'gamma':[1, 0.1, 0.01, 0.001],
              'kernel':['rbf','sigmoid','poly']}

grid= GridSearchCV(model,param_grid,refit=True)
grid.fit(features_train,target_train)
print(grid.best_estimator_)
grid_prediction = grid.predict(features_test)
#fitted_model= model.fit(features_train,target_train)
#prediction= fitted_model.predict(features_test)

print(confusion_matrix(target_test,grid_prediction))
print(accuracy_score(target_test,grid_prediction))