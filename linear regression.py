import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
file_path = r'C:\Users\apply-system\Documents\code\house_prices.csv'
house_data = pd.read_csv(file_path)
size = house_data['sqft_living']
price = house_data['price']
 

x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

model = LinearRegression()
model.fit(x,y)

regression_model_mse = mean_squared_error(x,y)
print("mse:",math.sqrt(regression_model_mse))
print("r squard value:" , model.score(x,y))

print(model.coef_[0])
print(model.intercept_[0])

plt.scatter(x,y, color ='green')
plt.plot(x,model.predict(x), color='black')
plt.xlabel("size")
plt.ylabel("price")
plt.title("linear regression")
plt.show()
print("predict by model:", model.predict([[2000]]))