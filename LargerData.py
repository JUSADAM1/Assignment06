#2/12/2019
#importing the numpy library and using
#NP as my alis to reference

# used this so one is able to place data as a document
print(__doc__)

import numpy as np
#i imported matplotlib as plt to be able to plot data points
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
#imported this to be able to use the Classifier to classify data and plot on a graph
from sklearn.tree import DecisionTreeClassifier


#Importing the Iris dataset to be able to use it
iris_data_set = load_iris()

#printing out all the iris dataset
print("_" * 40)
#prints features
print("Features")
#prints and calling Iris data
print(iris_data_set.feature_names)

#Prints out the labels of the flowers
print("_" * 40)
#prints labels
print("Labels")
#labels
print(iris_data_set.target_names)

#Lets take a look at our data
#lets have a look at the first row
print("-" * 40)
print("")
#Notice the data below will line up perfectly
#The feature above. this is feature data
print("Notice the data below will line up perfectly with")
print("the features above. this is feature data.")
print(iris_data_set.data[0])

#Lets look at the label for this line of data
print("_" * 40)
print("show the label for row 0. We are just taking a peek at the data")
print("labels table")
print("0 = setosa 1 = versicolor 2 = virginica")
print("labels= ", iris_data_set.target[0])
#beginning of a new
print ("_" * 40)
print("the first iris dataset will be used to classifiy 3 types of flowers")
print("The data set is 150 rows. 50 rows for each flower")
print("these rows are in order.")
print("Rows 0 - 49")
print("rows 50 - 99")
print("rows 100 -149")
#print out the data set to have a refernce
print("_" * 40)
print("the full data set to  reference  ")
# the data data
#TO BE HONEST I though this was kinda cool
#Only because one used a for loop to use the iris dataset to where it counts
# down with the label number and the features on it in one row and has it going  down
for i in range(len(iris_data_set.target)):
    #how one wants the dataset to print out:
    #THE COOL PART.
    print("Example %d: label %s Features %s:" % (i, iris_data_set.target[i], iris_data_set.data[i]))

#Remove one type of each flower type
#Because we are going
#This test data will be data never seen before by our classifier
# What we put in a test
test_index = [60, 50, 100, 0, 149]

# now lets make a set of training data
#this is the bulk of our data
#we will have to 147 rows of data to use for training
train_target = np.delete(iris_data_set.target, test_index)
train_data = np.delete(iris_data_set.data, test_index, axis=0)

# this right here!! is unseen data by the classifier which contains the three test flowers that one is classifying.
test_target = iris_data_set.target[test_index]
# Iris dataset to test because we are going back.
test_data = iris_data_set.data[test_index]

#Here is the coolest part!!!!!!!!!!
#create our  classifiers
# using a decision tree to classify
dt_clf = tree.DecisionTreeClassifier()
#finding the pattern within the data
dt_clf.fit(train_data, train_target)

#Here is where the magic happens
print("_" * 40)
#test print
print("test data")
#targets the necessary to compute
#our data that we input
print(test_target)
#line break
print("_" * 40)
#here is where the machine makes its prediction on the data or unknowns
#given by we the human beings
print("Machine's prediction data, check against test")
print("is this a match human")
#output of what the machine got
#machines prediction
print(dt_clf.predict(test_data))
print("_" * 40)
print("_" * 40)
print("As you look above you will see the data of a Iris flower"
      "and you will see many dataset")
print("_" * 40)
#below one will see how the data from the I Iris dataset plotted on a graph
print("below one will see how the data from the I Iris dataset plotted on a graph")
print("One will also see the difference between all three flowers")
print("_" * 65)
print("Based on data")
print("_" * 65)
print("SETOSA ANALYSIS")
print("_________________")
print("One will see that the setosa SEPAL has the largest width rather than the versicolor and the virginica flowers")
print("As one my also look at the rest of our data, the SETOSA flower is that it does not have a such a ")
print(" good record when it comes down to petal lenght to sepal length, petal width to sepal length and the others")
print("_" * 70)
print("VERSICOLOR ANALYSIS")
print("_________________________")
print("As one may take a look at the Versicolor flower, it just about has the same similarities as the Virginica flower.")
print("One would swear they where brother and sister plants")
print("Well I guess they are brother and sister plans lol")
print("__" * 75)
print("VIRGINICA Plant")
print("__________________________")
print(" AT last, Finally the final plant")
print("When taking a look at the data based on the Virginica flower one will see that it is the best one out of the all")
print("Meaning if you are going to get a Iris plant I would suggestion this one, only because its data is better than all the others.")
print("ALL-IN-ALL GET THE VIRGINICA PLANT")

#this would be the parameters used to make everything happen
# n_classes is the number of  classification problem
n_classes = 5
#this is the colors it goes by
#r = red
#y = yellow
#b = blue
# colors for plot points
plot_colors = "ryb"
#plot_step
plot_step = 0.02

#Again loading in the data to make graph
iris = load_iris()

#points to plot
#data pairs
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):

    # We only take the two corresponding features when plotting points
    #SET for X AXIS on the Graph
    X = iris.data[:, pair]
    #SET FOR Y AXIS on graph
    y = iris.target

    # Training data
    clf = DecisionTreeClassifier().fit(X, y)

    # Plotting data points
    plt.subplot(2, 3, pairidx + 1)
    # setting up the max and min of the graph
    #data can only be 0 or bigger than 1
    #max and min of the x axis
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #max and min of y axis
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #grid type
    #and how to arrange them on the plot map
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    #Layout of Graphs(h_pad=height)
    #w_pad = width
    #pad = how big the board is overall
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # data prediction based off the data given
    # The couple of lines below this line is where the graph may change depending upon the data
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # X AXIS LABEL Feature name for the data
    plt.xlabel(iris.feature_names[pair[0]])
    #Y AXIS of the data points
    plt.ylabel(iris.feature_names[pair[1]])

    # Plotting trianing points
    for i, color in zip(range(n_classes), plot_colors):
        #the where
        idx = np.where(y == i)
        #Graph type
        #setting up how the graph colors
        #like edge colors,plot colors, name color
        #as one may look (
        #edge color changes the outer outline of the plot points
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
#Placing the name at the top of the Graph
plt.suptitle("Decision surface of a decision tree using paired features")
#How the graph may fit on you blank canvas
#this is for the little box in the corner
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
#how you want the setup to be.
# in this case its "tight"(think of a Cheerio and then look at the graph)
# then youll know whatI mean
plt.axis("tight")
#shows it all
plt.show()

