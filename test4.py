# -*- coding: utf-8 -*-

# Required Python Machine learning Packages
import csv
import pandas as pd
import numpy
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

filename = 'heart.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')

	
print "Dataset Length:: ", len(data)
print "Dataset Shape:: ", data.shape

X = data[:,0:12]
Y = data[:,13]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.05, random_state = 100)

clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print Y_pred

print "Accuracy is",accuracy_score(Y_test, Y_pred, normalize = True)*100
