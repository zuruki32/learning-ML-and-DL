import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import  GaussianNB


filepath= r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\credit_data.csv"
credits = pd.read_csv(filepath)
features = credits[["income","age","loan"]]
Target = credits.default
print(features.corr())
x = np.array(features).reshape(-1,3)
y = np.array(Target)

features_train , features_test, Target_train, Target_test = train_test_split(x,y,test_size=0.2)
model = GaussianNB()
fitedmodel = model.fit(features_train,Target_train)
predict = fitedmodel.predict(features_test)
print(confusion_matrix(Target_test,predict))
print(accuracy_score(Target_test,predict))