import sys
import os
from os import path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append(path.abspath('./util'))
from clearMail import Mail2txt
import logging
                                

directory="./CSDMC2010_SPAM/CSDMC2010_SPAM/TRAINING/"

holdtext = Mail2txt(directory)
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



np.random.seed(7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)


feature_columns = [tf.contrib.layers.real_valued_column("", dimension=3000)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[100,120,170],n_classes=2,
                                              optimizer=tf.train.AdamOptimizer(0.0001))


# Define the training inputs
def get_train_inputs():
    x = tf.constant(X_train)
    y = tf.constant(y_train)

    return x, y

  # Fit model.

logging.getLogger().setLevel(logging.INFO)  
classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
def get_test_inputs():
    x = tf.constant(X_test)
    y = tf.constant(y_test)

    return x, y


accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))