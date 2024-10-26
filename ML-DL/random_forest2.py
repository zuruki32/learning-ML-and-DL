from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
"""
filepath= r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\credit_data.csv"
credits = pd.read_csv(filepath)

features = credits[["income","age","loan"]]
targets = credits.default

x= np.array(features).reshape(-1,3)
y = np.array(targets)

model = RandomForestClassifier()
prediction = cross_validate(model,x,y,cv=10)

print(np.mean(prediction['test_score'])) 

"""

digit_Data= datasets.load_digits()

features = digit_Data.images.reshape((len(digit_Data.images),-1))
target = digit_Data.target

random_forest_model = RandomForestClassifier(n_jobs=-1,max_features='sqrt')
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2 )

param_grid = {
    "n_estimators":[10,100,500,1000],
    "max_depth":[1,5,10,15],
    "min_samples_leaf":[1,2,4,10,15,30,50]
}

grid_search = GridSearchCV(estimator=random_forest_model,param_grid=param_grid,cv=10)
grid_search.fit(features_train,target_train)
print(grid_search.best_params_)

grid_predictiom = grid_search.predict(features_test)
print(confusion_matrix(target_test,grid_predictiom))
print(accuracy_score(target_test,grid_predictiom))