# Statistical Learning: Linear Regression


# Introduction to Statistical Learning ---------------------------------------------------------------------------
# Quantitative variables are called regressions
# Qualitative variabels are called classifications
# This distinction is based on the nature of the outputs
# Loss function to asses how far our predictions are from the actual Y
# most common loss function is Squared error loss, the best value to predict is the conditional expectation

# Introduction to Statistical Learning, Question 1  ---
# What is the difference between supervised and unsupervised learning?
# Answer: Supervised learning matches inputs and outputs, whereas unsupervised learning discovers structure for inputs only. 

# Q2: 
# Regression results in continuous outputs, whereas classification results in categorical outputs. 

# Q3
# Least squares loss is used to estimate the expected value of outputs, whereas 0 - 1 loss is used to estimate the probability of outputs. 

# Generating Example Regression Data ---------------------------------------------------------------------------
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10*ss.uniform.rvs(size = n) # generate 0,1 uniform values
y = beta_0 + beta_1*x + ss.norm.rvs(loc = 0, scale = 1, size = n)

plt.figure()
plt.plot(x, y, "o", ms = 5) # realization of the model, sample
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1 * xx) # function we use to generate data
plt.xlabel("x")
plt.ylabel("y")

# Simple Linear Regression --------------------------------------------------------------------------------------
# RSS = sum((Y_hat - Y)**2)

# Simple Linear Regression, Question 2  ---
# he following code implements the residual sum of squares for this regression problem:
def compute_rss(y_estimate, y):
    return sum(np.power(y-y_estimate, 2))

def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x

rss = compute_rss(estimate_y(x, beta_0, beta_1), y) 
print(rss)

# Least Squares Estimation in Code ------------------------------------------------------------------------------
# estimate the correct slope
rss = []
slopes = np.arange(-10, 15, 0.01)
for slope in slopes:
    rss.append(np.sum((y -  beta_0 - slope * x)**2))

ind_min = np.argmin(rss)
print("Estimate for the slope: ", slopes[ind_min])

# plot figure of RSS
plt.figure()
plt.plot(slopes,rss)
plt.xlabel("Slope")
plt.ylabel("RSS")
# Note: the plot is a parabola with a abs


# Simple Linear Regression in Code  -----------------------------------------------------------------------------
# VERY SIMILAR TO R???
# the smaller the sd the more precisely is being estimated
# 95% CI: coef +- 1.96*sd
import statsmodels.api as sm
mod = sm.OLS(y, x) # no intercept
est = mod.fit()
print(est.summary())

# with intercept
X = sm.add_constant(x)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())

# Multiple Linear Regression ------------------------------------------------------------------------------------
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size = n) # rvs stands for realizations of this dist
x_2 = 10 * ss.uniform.rvs(size = n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc = 0, scale = 1, size = n)

X = np.stack([x_1, x_2], axis = 1) # append as cols

# 3d plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], y, c = y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$");

# scikit-learn for Linear Regression ----------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X,y)
lm.intercept_
lm.coef_

X_0 = np.array([2, 4])
lm.predict(X_0.reshape(1,-1)) # reshape comes from the warning trying to run it without it

lm.score(X,y) # R-squared

# Assessing Model Accuracy --------------------------------------------------------------------------------------
# mean squared error (MSE) to evaluate the performance of a regression model
# MSE = 1/n sum((y_i - f(x_i))^2)
# Training error rate: proportion of errors the classifier makes on the training data
# Test error rate: analogous for test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, random_state = 1) #random_state is the seed
lm = LinearRegression(fit_intercept = True)

lm.fit(X_train, y_train) # fit the model in the training data
lm.score(X_test, y_test) # Accuracy of the model when testing 

# Assessing Model Accuracy, Question 1  ---
# When evaluating the performance of a model in a regression setting on test data, which measure is most appropriate?
# Answer: Test MSE 
# When evaluating the performance of a model in a classification setting on test data, which measure is most appropriate?
# Test error rate 






