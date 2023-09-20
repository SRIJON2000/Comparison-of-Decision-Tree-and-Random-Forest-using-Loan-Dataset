# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.datasets import load_iris

li = load_iris()

# Reading the data By converting Normal String DataSet to Raw String DataSet

irisdata = pd.read_csv(r"C:\Users\satyaki\Downloads\IRIS.csv")

# Printing First 5 Rows

irisdata.head()

data = irisdata.values
X=data[:, 0:4]
Y=data[:,4]
X[0:4]

Y[0:4]

# Splitting Data

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # 80% training and 20% test

# Building Decision Tree Model

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = clf.predict(X_test)

#Evaluating Model

# Model Accuracy, how often is the classifier correct?

print("Accuracy Of Decision Tree:",metrics.accuracy_score(Y_test, Y_pred))

# Visualizing Decision Trees

3 ways to do visualization:-


1.
'''from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True, feature_names = li.feature_names,class_names=li.target_names)

graphviz.Source(dot_data)'''


2.
'''from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


dot_data= StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = li.feature_names,class_names=li.target_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('species.png')
Image(graph.create_png())'''


3.
'''from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

#Printing the Decision Tree
dot_data = StringIO()
filename = "speciestree.png"
featureNames = irisdata.columns[0:4]
targetNames = irisdata["species"].unique().tolist()
out=tree.export_graphviz(clf,feature_names=featureNames, out_file=dot_data, class_names= np.unique(Y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')'''

# Optimizing Decision Tree Performance

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

'''from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


dot_data= StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = li.feature_names,class_names=li.target_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('species.png')
Image(graph.create_png())'''