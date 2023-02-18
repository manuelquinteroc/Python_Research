# Topic 4: Rnadomness and Time 
# Simulating Randomness --------------------------------------------------------------------------------------
import random
random.choice(["H", "T"]) # heads or tails 
random.choice([0,1]) 
random.choice(range(1,7)) # dice example

# example of 3 dices with different faces 
random.choice([range(1,7), range(1,9), range(1,11)]) # first select a dice, one with 6, 8, or 10 faces
random.choice(random.choice([range(1,7), range(1,9), range(1,11)])) # this works to chose a face from one of the dices chosen uniformly

# Simulating Randomness: Question 3  ---
# Which of the following lines of code takes the sum of 10 random integers between 0 and 9?
sum(random.choice(range(10)) for i in range(10))

# Examples Involving Randomness -------------------------------------------------------------------------------
# Ex1: Roll a dice 100 times and plot a histogram of the outcomes.
import matplotlib.pyplot as plt
import numpy as np

rolls = []
for x in range(100000):
    rolls.append(random.choice(range(1,7)))

plt.hist(rolls, bins = np.linspace(0.5,6.5,7), density= True) # Almost uniformly at 16.66% because of 100000 random numbers

# Ex2: consider rolling 10 independent dies, from x1,...,x10. Let y = x1 + ... + x10, 
# we want to understand the distribution of y (we know it would be centered around 30)
ys = []
for rep in range(100):
    y = 0 
    for k in range(10):
        x = random.choice(range(1,7))
        y = y + x
    ys.append(y)
    
min(ys)
max(ys)

plt.hist(ys);

ys = []
for rep in range(100000):
    y = 0 
    for k in range(10):
        x = random.choice(range(1,7))
        y = y + x
    ys.append(y)
    
min(ys)
max(ys)

plt.hist(ys); # like a normal distribution. Distribution of sum of uniforms is asintotically normal because of the central limit theorem


#  Using the NumPy Random Module ------------------------------------------------------------------------------------------
# we use numpy for 2 reasons:
# 1) very fast (10 times faster than standard python code)
# 2) many distirbutions

np.random.random() # 0,1 uniform distribution
np.random.random(5) # array of 5 obs
np.random.random((5,3)) # matrix of 5x3

np.random.normal(0,1,5) # mu, sigma, N
np.random.normal(0,1, (2,5)) # matrix

# Ex2 using numpy
# create a matrix of 10 x's where each y is the sum of rows
X = np.random.randint(1,7, (1000000,10)) # generate integers
# np.sum?
Y = np.sum(X, axis=1) # axis = 0 sums cols and axis = 1 sums rows
plt.hist(Y, density=True);

# Using the NumPy Random Module: Question 1  ---
# What does numpy.random.random((5,2,3)) do?
# Answer: Generates a 5 x 2 x 3 NumPy array with random uniform values. 

# Using the NumPy Random Module: Question 4 --- 
# What is the dimension of numpy.sum(numpy.random.randint(1,7,(100,10)), axis=0)?
x = np.sum(np.random.randint(1,7,(100,10)), axis=0)
x.shape

# Measuring Time --------------------------------------------------------------------------------------------------------------
import time
start_time = time.process_time() # .clock was deprecated
end_time = time.process_time()
end_time - start_time


start_time = time.process_time()
ys = []
for rep in range(1000000):
    y = 0 
    for k in range(10):
        x = random.choice(range(1,7))
        y = y + x
    ys.append(y)
end_time = time.process_time()


start_time2 = time.process_time()
X = np.random.randint(1,7, (1000000,10))
Y = np.sum(X, axis=1)
end_time2 = time.process_time()

# Compare the time of standard python and numpy to see performance
[end_time - start_time, end_time2 - start_time2]
(end_time - start_time) / (end_time2 - start_time2) # numpy is 71 times faster

# Random Walks ---------------------------------------------------------------------------------------------------------

# Ex: 100 steps from normal 0,1. We assume x and y displacements are independent
delta_X = np.random.normal(0,1,(2,5))

plt.plot(delta_X[0], delta_X[1], "go")

# cumlative sum in numpy: numpy.cumsum
# sum incrments
X = np.cumsum(delta_X, axis = 1) # over columns
plt.plot(X[0], X[1], "ro-");
plt.savefig("RandomWalk.pdf")

# concatenate numpy arrays: numpy.concatenate(a1,a2, axis)
X_0 = np.array([[0], [0]])
delta_X = np.random.normal(0,1,(2,100))
X = np.concatenate((X_0, np.cumsum(delta_X, axis = 1)), axis = 1) # starting point + deviations
plt.plot(X[0], X[1], "ro-");
plt.savefig("RandomWalk2.pdf")