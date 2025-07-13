
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
"""
digits = load_digits()
print(digits.data.shape)

x_digits = digits.data
y_digits = digits.target
 
estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_digits)

print(x_pca.shape)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    px = x_pca[:,0][y_digits == i]
    py = x_pca[:,1][y_digits == i]
    plt.scatter(px,py, c = colors[i])
    plt.legend(digits.target_names)

plt.xlabel('first component')
plt.ylabel('second component')
plt.show()

print("explained variance: %s" % estimator.explained_variance_ratio_)"
"""
mnist_Data = fetch_openml('mnist_784')

features = mnist_Data.data
targets = mnist_Data.target

print(features.shape)

train_img, test_img, train_lb, test_lb = train_test_split(features,targets, test_size=0.15 ,  random_state=0)

scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#keep 95% of information
pca = PCA(.95)
pca.fit(train_img)

train_img = pca.transform(train_img.shape)
test_img = pca.transform(test_img.shape)

print(train_img.shape)