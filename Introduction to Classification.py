# Introduction to kNN Classification -------------------------------------------------------------------------------
# Categorical variables are called classification problems
# Continuous variables are called regression problems


# Finding the Distance Between Two Points --------------------------------------------------------------------------
import numpy as np
p1 = np.array([1,1])
p2 = np.array([4,4])

# euclidean distance
np.sqrt(sum(np.power((p2 - p1), 2)))


def distance(p1 , p2):
    """
    Find the distance between points p1 and p2
    """
    return np.sqrt(sum(np.power((p2 - p1), 2)))

distance(p1,p2)

# Majority Vote ----------------------------------------------------------------------------------------------------
# The most common element

def majority_vote(votes):
    """
    create a dictionary with keys and counts
    """
    vote_counts = {}
    for vote in votes: # if already exist, add one
        if vote in vote_counts:
            vote_counts[vote] += 1
        else: # if does not exist, add a key
            vote_counts[vote] = 1
    return vote_counts

votes = [1,2,3,1,2,3,1,2,3,3,3,3]
vote_counts = majority_vote(votes)

max(vote_counts.keys()) # get key with max value
max_counts = max(vote_counts.values()) # max value

# Loop over all values in the dictionary
for vote, count in vote_counts.items():
    print(vote, count)

# 
winners = []
max_counts = max(vote_counts.values()) # max value
for vote, count in vote_counts.items(): # tuple vote and 
    if count == max_counts:
        winners.append(vote)
        
# in case of a tie, pick one winner at random and make a function
import random
def majority_vote(votes):
    """
    Return the most element in votes.
    """
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    
    # once the dictionary is created, get max count and then the key with such value, if repeated, append the keys
    winners = []
    max_counts = max(vote_counts.values()) # max value
    for vote, count in vote_counts.items(): # tuple vote and (A dict_items object with elements consisting of tuples of key, value pairs)
        if count == max_counts:
            winners.append(vote)
    
    return random.choice(winners)

votes = [1,2,3,1,2,3,1,2,3,3,3,3,2,2,2]
majority_vote(votes)

# SHORCUT: FIND THE MODE OF A NUMPY ARRAY
import scipy.stats as ss
def majority_vote_short(votes):
    """
    Return the most element in votes.
    """
    mode, count = ss.mstats.mode(votes) # the function returns a tuple of two values
    return mode
    
votes = [1,2,3,1,2,3,1,2,3,3,3,3,2,2,2]
list(majority_vote_short(votes))

# Finding Nearest Neighbors ----------------------------------------------------------------------------------------
# pseudocode
    # loop over all points
        # compute the distance between points p and every other point
    # sort distances and return those k points that are nearest to point p

    # create dataset
points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3]]) #embeded list in another list for numpy to work
p = np.array([2.5, 2])    

import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], "ro")
plt.plot(p[0], p[1], "bo")
plt.axis([0.5, 3.5, 0.5, 3.5])

# calcualte distances from p to all points
distances = np.zeros(points.shape[0])
for i in range(len(distances)):
    distances[i] = distance(p, points[i])
    
ind = np.argsort(distances) # array of indexes
distances[ind] # sorted distances

distances[ind[0:2]] # two nearest points (k = 2)

# create a function
def find_nearest_neighbors(p, points, k = 5):
    """
    Find the k-nearest neighbors of point p and return their indices.
    """
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

ind = find_nearest_neighbors(p, points, k = 2); print(points[ind]) # return coordinates of the 2 points nearest to p
ind = find_nearest_neighbors(p, points, k = 3); print(points[ind]) # return coordinates of the 3 points nearest to p

# create a predict function
def knn_predict(p, points, outcomes, k = 5): # note that points is our training data
    ind = find_nearest_neighbors(p, points, k) # returns an index
    return majority_vote(outcomes[ind]) # use class with mox votes, here is the class where the p point belongs to
     
    
# define outcomes, 9 points, 9 classes initially
outcomes = np.array([0,0,0,0,1,1,1,1,1]) # assume 2 clases
len(outcomes)

knn_predict(np.array([2.5, 2.7]), points, outcomes, 2) # classified as 1
knn_predict(np.array([1, 2.7]), points, outcomes, 2) # classified as 0


# Generating Synthetic Data ----------------------------------------------------------------------------------------
# We're going to write a function that generates two end data points, where
# the first end points are from class 0, and the second end points are from class 1.
# we are going to use the ipstats module
ss.norm(0,1).rvs((5,2)) # generate a 5x2 of normal 0,1 
ss.norm(1,1).rvs((5,2)) # generate a 5x2 of normal 1,1 

# generate points and outcome vector
def generate_synth_data(n = 50):
    """
    Create two sets of points for bivariate normal distributions.
    """
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis = 0) # concatenate along the rows
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)

n = 20
points, outcomes = generate_synth_data(n) # save a tuple

plt.figure()
plt.plot(points[:n,0], points[:n,1], "ro")
plt.plot(points[n:,0], points[n:,1], "bo")
plt.savefig("bivardata.pdf")



# Making a Prediction Grid ----------------------------------------------------------------------------------------
def make_prediction_grid(predictors, outcomes, limits, h, k):
    """
    Classify each point on the prediction grid.
    """
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h) # h is the step size. Creates regularly spaced values between the first and second argument, with spacing given in the third argument 
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys) # inputs two vectors and returns matrices, first matrix contains x values and second the y values,
    
    # make a prediciton for each point on the grid
    prediction_grid = np.zeros(xx.shape, dtype = int)
    
    for i,x in enumerate(xs): # enumearte is useful when dealing with sequences and we want to access simultaneously 2 things: elements and index values
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k) # j corresponds to y values, rows and x are columns
    
    return (xx, yy, prediction_grid)

# example of enumerate
seasons = ["spring", "summer", "fall", "winter"]
list(enumerate(seasons)) # returns a sequence of tuples. Inside every tuple there is an index and the secon the specific object
for ind, season in enumerate(seasons):
    print(ind, season)
    
# Plotting the Prediction Grid ------------------------------------------------------------------------------------
def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)


(predictors, outcomes) = generate_synth_data()
k = 5; filename = "knn_synth_5.pdf"; limits = (-3,4,-3,4); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

(predictors, outcomes) = generate_synth_data()
k = 50; filename = "knn_synth_50.pdf"; limits = (-3,4,-3,4); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

# Applying the kNN Method -----------------------------------------------------------------------------------------
from sklearn import datasets
iris = datasets.load_iris()

predictors = iris.data[:, 0:2] # column 1 and 2
outcomes = iris.target
plt.plot(predictors[outcomes == 0][:,0], predictors[outcomes == 0][:,1], "ro")
plt.plot(predictors[outcomes == 1][:,0], predictors[outcomes == 1][:,1], "go")
plt.plot(predictors[outcomes == 2][:,0], predictors[outcomes == 2][:,1], "bo")
plt.savefig("iris.pdf")

# Iris prediction plot
k = 5; filename = "iris_grid.pdf"; limits = (4,8,1.5,4.5); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

# Import Knn from sklearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)
# chekcs
sk_predictions.shape
sk_predictions[0:10]

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors]) # apply knn for each predictor of the 150
my_predictions.shape

# compare our predictiso to the one from sklearn
np.mean(sk_predictions == my_predictions)*100 # 97.33% of the time

np.mean(sk_predictions == outcomes)*100 # how frequent sklearn predictios agree with the actual observed outcomes
np.mean(my_predictions == outcomes)*100 # how frequent my predictions agree with the actual observed outcomes

# conclusion: using sklearn the accuracy is lower than our predictions




