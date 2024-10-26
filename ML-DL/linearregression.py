import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

file_path = r"D:\Udemy_2023_Machine_Learning_and_Deep_Learning_Bootcamp_in_Python_2022-8_Downloadly.ir\Udemy\deeplearning_code\Datasets\Datasets\house_prices.csv"

# Read the CSV file using the specified path
house_data = pd.read_csv(file_path)
#print(house_data)
size = house_data['sqft_living']
price= house_data['price']
#print(size)
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)
#print(x)


model = LinearRegression()
model.fit(x,y)

regression_model_mse= mean_squared_error(x,y)
print("mse",math.sqrt(regression_model_mse))
print("r squeared vlaue",model.score(x,y))
print(model.coef_[0])
print(model.intercept_[0])

plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x),color='black')
plt.title("LinearRegression")
plt.xlabel("size")
plt.ylabel("price")
plt.show()

print("prediction",model.predict([[2000]]))