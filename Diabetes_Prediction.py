from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
dataset = pd.read_csv('diabetes.csv')
print (dataset.info())
print (dataset.isnull().sum())
X = dataset.iloc[:,:-1]
y = dataset.iloc [:, [-1]]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 0)
classifier = RandomForestClassifier (n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit (X_train, y_train)
y_pred= classifier.predict (X_test)
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(classification_report (y_test, y_pred))
print ("The accuracy is :", classifier.score (X_test, y_test))
import scikitplot as skplt
import matplotlib.pyplot as plt
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()
print ("The new predection is:", classifier.predict (np.array([[6,142,72,45,0, 38.6,0.627,50]])))
print ("The new predection is:", classifier.predict (np.array([[1,109, 30, 38, 83, 53.3,0.193,33]])))