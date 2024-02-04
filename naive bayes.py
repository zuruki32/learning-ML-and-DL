import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB


credit = r'C:\Users\apply-system\Documents\code\credit_data.csv'
creditdata = pd.read_csv(credit)
feature = creditdata[["income","age","loan"]]
target = creditdata.default

feature_train, feature_test, target_train, target_test = train_test_split(feature,target,test_size= 0.3)
model = GaussianNB()
model_fitted = model.fit(feature_train,target_train)
prediction = model.predict(feature_test)
print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))


