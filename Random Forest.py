# Random Forest 


# Tree-Based Methods for Regression and Classification ---------------------------------------------------------------
# Tree based methods can be used for both regression and classification

# To make predictions:
# In a regression setting we return the mean of the outcomes of the trainingo observations in that particular regiont
# classification we return the mode of the region
# whenever we make a split, we consider all predictors from x1 to xp and for each predictor, we consider all possible cut points.
# We choose the predictor - cut point combination such that the resulting division of the predictor space has the lowest value of some criterion,
# usually a loss function that we are trying to minimize 

# Regression the loss function is the RSS
# Classificion we use the Gini index or the cross-Entropy index


# Random Forest Predictions ------------------------------------------------------------------------------------------
# how to aggregate the predictions of several trees to do random forest classification and regression
# two types of randomness introduced by the random forest method
# classification we return the mode of the region

# Randomness: 
# 1) Bagging or Bootstrap in decision trees means that we draw a number of bootstrap datasets and fit each one a tree.
# (Each tree gets a bootstrapped random sample of training data.)
# 2) How we split the predicor space. Each time we make a split, we take a new sample of predictors. That is to make a cut we might select
# predictors x1,x2,x3 but for the second tree x2,x4,x7, and so on.... (Each split only uses a subset of predictors.)


# Make a prediction for each tree and then take the mean or mode 

from sklearn.ensemble import RandomForestRegressor # To do random forest regression, you use the following import:
from sklearn.ensemble import RandomForestClassifier # To do random forest classification, you use the following import:


# Random forests get their name by introducing randomness to decision trees in two ways, once at the data level and once at the predictor level. 
