import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

filepath= r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\credit_data.csv"
credits = pd.read_csv(filepath)
features = credits[["loan","income","age"]]
target= credits.default

x = np.array(features).reshape(-1,3)
y = np.array(target)
model = LogisticRegression()
predicted = cross_validate(model,x,y,cv=5)
print(predicted['test_score'])