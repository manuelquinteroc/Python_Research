# Logistic Regression
    
# Generating Example Classification Data ---------------------------------------------------------------------
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
#  matplotlib notebook

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

h = 1
sd = 1
n = 50

def gen_data(n, h, sd1, sd2):
    x1 = ss.norm.rvs(-h, sd, n) # centered at -h
    y1 = ss.norm.rvs(0, sd, n)

    x2 = ss.norm.rvs(h, sd, n) # centered at h
    y2 = ss.norm.rvs(0, sd, n)
    
    return (x1, y1, x2, y2)

(x1, y1, x2, y2) = gen_data(1000, 1.5, 1, 1.5)

# visualize data
def plot_data(x1, y1, x2, y2):
    plt.figure()
    plt.plot(x1,y1, "o", ms = 2)
    plt.plot(x2,y2, "o", ms = 2)
    plt.xlabel("$X_1$")
    plt.xlabel("$x_2$")

plot_data(x1, y1, x2, y2)

# Logistic Regression ----------------------------------------------------------------------------------------
# Notice that we want p(x) = beta_0 + beta_1 X
# Instead of modelling p(x), we model log (p(x)/(1 - p(x))), where p(x) / (1 - p(x)) is called the odds ratio
# We usually estimate using MLE

# The following code creates a function that converts probability to odds:
def prob_to_odds(p):
    if p <= 0 or p >= 1:
        print("Probabilities must be between 0 and 1.")
    return p / (1-p) 

# Logistic Regression in Code --------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression()
# we need to built our x matrix for logistic regression appending all columns of X into a single column vector, and duplicating the values of y_i n times in a column vector
# or information matrix contains these two matrices
n = 1000
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T)) # .T stands for Transpose
X.shape
y = np.hstack((np.repeat(1,n), np.repeat(2,n))) 
y.shape

# generate test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, random_state = 1) #random_state is the seed

# fit the classifier
clf.fit(X_train, y_train)
clf.score(X_test, y_test) # accuracy

# compute estimates of class probs
clf.predict_proba(np.array([-2,0]).reshape(1, -1)) # probabilities of belonging to either group

# make a prediction
clf.predict(np.array([-2,0]).reshape(1, -1)) # probabilities of belonging to either group
# for the values of -2 and 0, the prediction is class 1

# Computing Predictive Probabilities Across the Grid ---------------------------------------------------------
# how to use meshgrid and ravel to compute predictive probabilities
# how to plot the predictive Probabilities

def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1)) # convert into vectors with ravel()
    Z = probs[:,class_no] # prob. from all rows for a particular class number
    Z = Z.reshape(xx1.shape) # turns into xx1 shape
    CS = ax.contourf(xx1, xx2, Z) # plot the value of Z at xx1, xx2
    cbar = plt.colorbar(CS) 
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

# create figures
plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0) # provide axes
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1) 
plt.title("Pred. prob for class 2");

