import sys
import os
from os import path
import pandas as pd
import numpy as np
sys.path.append(path.abspath('./util'))

from clearMail import Mail2txt

directory="./CSDMC2010_SPAM/CSDMC2010_SPAM/TRAINING/"

holdtext = Mail2txt(directory)



from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=3000)
vect.fit(holdtext)
simple_train_dtm = vect.transform(holdtext)
std   =  simple_train_dtm.toarray()


# load labels
path="./CSDMC2010_SPAM/CSDMC2010_SPAM/SPAMTrain.label"
td = pd.read_csv(path)
label=[]
with open(path) as m:
    te = m.readlines()
for i in range(len(te)):
    lb = te[i].split(" ")
    label.append(int(lb[0]))

y = np.asarray(label)
X = std


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import train_test_split

np.random.seed(7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)

result1 = model1.predict(X_test)
result2 = model2.predict(X_test)


from sklearn.metrics import confusion_matrix 

print("Confusion_matrix: MultinomialNB() ",confusion_matrix(y_test,result1))
print("Confusion_matrix: LinearSVC()",confusion_matrix(y_test,result2))

from sklearn.metrics import accuracy_score
print("acc of using model: MultinomialNB() is : ",accuracy_score(y_test,result1))
print("acc of using model: LinearSVC() is : ",accuracy_score(y_test,result2))