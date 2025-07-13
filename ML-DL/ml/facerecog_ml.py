from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut 
from sklearn.model_selection import  KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti= fetch_olivetti_faces()
features = olivetti.data
target = olivetti.target

"""
fig, sub_plots = plt.subplots(nrows=5 , ncols= 8 , figsize = (14,8))
print( sub_plots)
sub_plots = sub_plots.flatten()
for unique_user_id in np.unique(target):
    image_index = unique_user_id * 8
    sub_plots[unique_user_id].imshow(features[image_index].reshape(64,64), cmap = 'gray')
    sub_plots[unique_user_id].set_xticks([])
    sub_plots[unique_user_id].set_yticks([])
    sub_plots[unique_user_id].set_title("face id:%s" %unique_user_id)
plt.suptitle("dataset")
plt.show()
"""

x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.3,stratify=target, random_state=0)
"""
pca = PCA()
pca.fit(features)
plt.figure(1,figsize=(12,8))
plt.plot(pca.explained_variance_,linewidth =2)
plt.xlabel('component')
plt.ylabel('variance')
plt.show()
"""

pca = PCA(n_components= 100, whiten= True)
pca.fit(x_train)
x_pca = pca.fit_transform(features)
x_train_pca= pca.transform(x_train)
x_test_pca = pca.transform(x_test)
"""
number_of_eighenfaces = len(pca.components_)
eighenfaces = pca.components_.reshape((number_of_eighenfaces,64,64))
fig, sub_plots = plt.subplots(nrows=10 , ncols= 10 , figsize = (15,15))
sub_plots = sub_plots.flatten()
for i in range(number_of_eighenfaces):
    sub_plots[i].imshow(eighenfaces[i], cmap = 'gray')
    sub_plots[i].set_xticks([])
    sub_plots[i].set_yticks([])
   
plt.suptitle("eighen")
plt.show()
"""

models = [("logistice regression", LogisticRegression()),("support vector machine", SVC()), ("naive", GaussianNB())]
"""
for name,model in models:
    classifier_model = model
    classifier_model.fit(x_train_pca,y_train)
    y_pred = classifier_model.predict(x_test_pca)
    print("result with %s" % name)
    print("accuracy %s " % (metrics.accuracy_score(y_test,y_pred)))
"""
for name,model in models:
    kFold = KFold(n_splits= 5, shuffle= True, random_state=0)
    cv_score = cross_val_score(model,x_pca,target,cv= kFold)
    print("mean: %s" % cv_score.mean())