# NAVIE - BAYES BERNOULLI USING NORMALIZER

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\30th, 31st\Social_Network_Ads.csv') 
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.preprocessing import Normalizer
norm = Normalizer()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)'''

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
cr

bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance


dataset1  = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\29th\Future prediction1.csv')
d2 = dataset1.copy()
d2

dataset1 = dataset1.iloc[:,[2,3]].values
dataset1
'''
from sklearn.preprocessing import Normalizer
norma = Normalizer()
M = norma.fit_transform(dataset1)
M
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)
M
y_pred1 = pd.DataFrame(M)
y_pred1

d2['y_pred1'] = classifier.predict(M)
d2['y_pred1']
