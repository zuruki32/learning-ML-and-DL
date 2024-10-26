from sklearn.feature_extraction.text import TfidfVectorizer , TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
"""
vec = TfidfVectorizer()

tfidf = vec.fit_transform(['i like machine learning and clustring algorithms',
                           'apples, oranges and any kind of fruits are healthy',
                            'is it feasible with machine learning algorithms?',
                             'my family is happy because of the healthy fruits' ])
#print(tfidf)
#print(tfidf.A)
#SIMILARITY MATRX
print((tfidf*tfidf.T).A)
"""
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
trainig_data = fetch_20newsgroups(subset='train',categories=categories,shuffle=True, random_state=42)
#print("\n".join(trainig_data.data[0].split("\n")[:10]))
#print("target is", trainig_data.target_names[trainig_data.target[0]])

count_vector = CountVectorizer()
x_train_count= count_vector.fit_transform(trainig_data.data)
#print(count_vector.vocabulary_)

Tfidf_transformer = TfidfTransformer()
x_train_tfidf = Tfidf_transformer.fit_transform(x_train_count)
model = MultinomialNB().fit(x_train_tfidf,trainig_data.target)
new = ['this has nothing to do with church or religion ','software engineering is getting hotter and hotter nowadays' ]
x_new_count = count_vector.transform(new)
x_new_tfidf = Tfidf_transformer.transform(x_new_count)
prediction = model.predict(x_new_tfidf)

for doc, categories in zip(new,prediction):
    print(f'{doc!r} -----> {trainig_data.target_names[categories]}')