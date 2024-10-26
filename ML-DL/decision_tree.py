import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
"""
iris_data = datasets.load_iris()

features = iris_data.data
target= iris_data.target
param_grid = {'max_depth':np.arange(1,10)}
 
features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=0.2)
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(features_train, target_train)
print("best parameter:", tree.best_params_)

grid_prediction = tree.predict(features_test)
print(confusion_matrix(target_test,grid_prediction))
print(accuracy_score(target_test,grid_prediction))
"""
cancer_Data = datasets.load_breast_cancer()
features = cancer_Data.data
labels = cancer_Data.target 
features_train , features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.2)

model = DecisionTreeClassifier(criterion='entropy',max_depth=5)
prediction = cross_validate(model,features,labels,cv=10)
print(np.mean(prediction['test_score']))