import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
"""

x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

plt.plot(x1,y1,'ro',color='blue')
plt.plot(x2,y2,'ro',color='red')

model= LogisticRegression()
model.fit(X,y)

print("b0 is",model.intercept_)
print("b1 is",model.coef_)

def logistic(classifier, x):
	return 1/(1+np.exp(-(model.intercept_ + model.coef_ * x)))
	
for i in range(1,120):
	plt.plot(i/10.0-2,logistic(model,i/10.0),'ro',color='green')

plt.axis([-2,10,-0.5,2])
plt.show()

pred = model.predict_proba([[10]])
print("Prediction: ", pred)
"""
filepath= r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\credit_data.csv"
credits = pd.read_csv(filepath)

print(credits.head())
print(credits.describe())
print(credits.corr())

feature= credits[["income", "loan","age"]]
target =  credits.default
feature_train, feature_test, target_train, target_test = train_test_split(feature,target,test_size=0.3)
model = LogisticRegression()
model.fit= model.fit(feature_train,target_train)
prediction= model.predict(feature_test)
print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))