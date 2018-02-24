import csv
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

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

scaler = StandardScaler()

scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,Y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(Y_test,predictions))

print(classification_report(Y_test,predictions))

print predictions

print "Accuracy is ", accuracy_score(Y_test,predictions)*100
