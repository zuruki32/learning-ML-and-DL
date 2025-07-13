from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
iris_data = datasets.load_iris()
from sklearn import preprocessing
import  pandas as pd
import numpy as np
"""
features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features,targets, test_size=0.2)
model = AdaBoostClassifier(n_estimators=100,learning_rate=1, random_state=123)
fitted_model = model.fit(feature_train,target_train)
model_prediction = fitted_model.predict(feature_test)
print(confusion_matrix(target_test,model_prediction))
print(accuracy_score(target_test,model_prediction))
"""


filepath= r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\wine.csv"

def is_testy(quality):
    if quality>7:
        return 1 
    else:
        return 0
data = pd.read_csv(filepath,sep =";")
print(data)