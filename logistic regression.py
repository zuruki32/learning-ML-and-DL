import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
Y = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
plt.plot(x1,y1,'ro',color ='blue')
plt.plot(x2,y2,'ro',color ='red')
plt.show()

model = LogisticRegression()
model.fit(X,Y)

print("b0 is:", model.intercept_)
print("b1 is:",model.coef_)

PRED = model.predict([[1]])
PRED1 = model.predict_proba([[3.5]])
print("prediction:", PRED1)

################################

credit = r'C:\Users\apply-system\Documents\code\credit_data.csv'
creditdata = pd.read_csv(credit)

print(creditdata.head())
print(creditdata.describe())
print(creditdata.corr())

feature = creditdata[["income","age","loan"]]
target = creditdata.default
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.3)
model = LogisticRegression()
model.fit = model.fit(feature_train,target_train)
prediction = model.fit.predict(feature_test)

print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))
#b1-b2-b3
print(model.fit.coef_)
#b0
print(model.fit.intercept_)