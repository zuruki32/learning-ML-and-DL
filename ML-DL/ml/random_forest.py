from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()
features = iris_data.data
target = iris_data.target

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
model = RandomForestClassifier(n_estimators= 1000, max_features= 'sqrt')
fitted_model = model.fit(features_train,target_train)
prediction = fitted_model.predict(features_test)

print(confusion_matrix(target_test,prediction))
print(accuracy_score(target_test,prediction))