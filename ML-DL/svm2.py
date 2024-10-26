from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
digits = datasets.load_digits()
images_and_labels= list(zip(digits.images,digits.target))

for index, (image,label) in enumerate(images_and_labels[:6]):
    plt.subplot(2,3, index +1)
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation = 'nearest')
    plt.title('target:%i' % label)

plt.show()

data = digits.images.reshape((len(digits.images),-1))
classifier = svm.SVC(gamma=0.001)
train_test_split = int(len(digits.images)*0.75)
classifier.fit(data[:train_test_split],digits.target[:train_test_split])

expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])
print(confusion_matrix(expected,predicted))
print(accuracy_score(expected,predicted))

