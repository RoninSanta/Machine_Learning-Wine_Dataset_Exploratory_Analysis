# Deep Learning-Wine Dataset Exploratory Analysis
I will be working with the Wine dataset. This is a 178 sample dataset that categories 3 different types of Italian wine using 13 different features. Then I will use machine learning to analyze the data and find the `common features` among the types of wines.


### [Data Visualization]
The first part of tackling any ML problem is visualising the data in order to understand some of the properties of the problem at hand, here are some methods is used to achieve that:
- Create a colored grid for the Wine dataset, with each off-diagonal subplot showing the interaction between two features, and each of the classes represented as a different colour.
The function is invoked something like this `myplotGrid(X,y,...)`
```
#get Number of column in grid
 num_Col = X.shape[1]
 #Create axis and plot Grid
 fig,ax = plt.subplots(num_Col,num_Col,figsize=(12,12))
 fig.suptitle("Interaction between selected features")
```
### Exploratory Data Analysis under noise
When data are collected under real-world settings they usually contain some amount of noise that makes classification more challenging. The model below is used to test how it performs unders such environments.

### Implementing kNN
Define a function that performs k-NN given a set of data. Your function should be invoked similary to:
` y_ = mykNN(X,y,X_,options)`

where X is your training data, y is your training outputs, X_ are your testing data and y_ are your predicted outputs for X_. The options argument (can be a list or a set of separate arguments depending on how you choose to implement the function) should at least contain the number of neighbours to consider as well as the distance function employed.

### Classifier evaluation
A way to evaluate the model created and test for metrics such as accuracy.
```
#Confusion matrix
def confusion_matrix(y_test, y_):

#Evaluate accuracy of the matrix
def Accuracy_score(y_test,y_):

#Get the precision of the model
def Precision(y_test,y_):

#Get Recall/Sensitive value of the model per class
def Recall(y_test,y_):


# test evaluation code
# PLOT the confusion matrix using Matplotlib
matrix = confusion_matrix(y_test, y_)

```
