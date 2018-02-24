# -*- coding: utf-8 -*-

import csv
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

filename = 'heart.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')

	
print "Dataset Length:: ", len(data)
print "Dataset Shape:: ", data.shape

X = data[:,0:12]
Y = data[:,13]

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = None)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=7, min_samples_leaf=5)
clf_gini.fit(X_train, Y_train)




clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=7, min_samples_leaf=5)
clf_entropy.fit(X_train, Y_train)



print "Gini Index"                                                    

y_pred = clf_gini.predict(X_test)
print y_pred

print "Entropy"
y_pred_en = clf_entropy.predict(X_test)
print y_pred_en

print "Gini Index"
print "Accuracy is ", accuracy_score(Y_test,y_pred)*100

print "Entropy"
print "Accuracy is ", accuracy_score(Y_test,y_pred_en)*100


# Gini : Gini(E)=1−∑(j=1 to c)(pj)^2
# Entropy:H(E) = −∑(j=1 to c)(pj)log(pj)

