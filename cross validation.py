import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


credit = r'C:\Users\apply-system\Documents\code\credit_data.csv'
creditdata = pd.read_csv(credit)
feature = creditdata[["income","age","loan"]]
Y = creditdata.default
X = np.array(feature).reshape(-1,3)

model = LogisticRegression()
predict = cross_validate(model,X,Y, cv = 5)
print(predict['test_score'])
print(np.mean(predict['test_score']))