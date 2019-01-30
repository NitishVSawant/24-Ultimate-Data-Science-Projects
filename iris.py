   #importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlretrieve
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

   #Loading the dataset
   
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(iris, names=attributes, sep=',')
print(df.head())

   #Data Exploration
   
print(df.info()) #information about data types
print(df.shape) #dimensions of dataframe
print(df.describe()) #descriptive stats
df.columns
print(df.groupby('class').size()) #class distribution

   #Data Visualization

# box and whisker plots
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
df.hist()
plt.show()

# scatter plot matrix
scatter_matrix(df)
plt.show()

    #Splitting the data
    
#array = df.values
#x = array[:,0:4]
#y = array[:,4]
X = df.iloc[:,:-1].values #or X = df.iloc[:,0:4].values
Y = df.iloc[:,-1].values  #or Y = df.iloc[:,4].values

test_size = 0.20
seed = 5
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    #Scaling the data
    
sc_X = StandardScaler()
sc_X_train = sc_X.fit_transform(X_train)
sc_X_test = sc_X.fit_transform(X_test)

    #Model Building

#1. Logistic Regression   

LR_Classifier = LogisticRegression(random_state= seed)   
LR_Classifier.fit(sc_X_train, Y_train)
LR_Y_pred = pd.DataFrame(LR_Classifier.predict(X_test))
LR_CM = confusion_matrix(Y_test, LR_Y_pred)

#2. KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train,)





# Provide data whose class labels are to be predicted
X = [
    [5.9, 1.0, 5.1, 1.8],
    [3.4, 2.0, 1.1, 4.8],
]

# Prints the data provided
print(X)

# Store predicted class labels of X
prediction = knn.predict(X)

# Prints the predicted class labels of X
print(prediction)