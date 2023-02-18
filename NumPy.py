# NumPy notes
# NumPy arrays have a size that is fixed when they are constructed
# Elements of arrays are also all of the same data
# By default, the elements are floating point numbers

import numpy as np

zero_vector = np.zeros(5)
zero_matrix = np.zeros((5,3))
zero_matrix

# empty array
empty_vector = np.empty(3)
empty_vector

x = np.array([1,2,3])
y = np.array([2,4,6])

[[1,3], [5,9]] # nested list
np.array([[1,3], [5,9]]) # convert into an array

# Tranpose
A = np.array([[1,3], [5,9]])
A.transpose()

# Introduction to NumPy Arrays: Question 2  ---
np.array([0., 0., 0., 0., 0.])
# What code will produce that object
np.zeros(5)


# Slicing NumPy Arrays ------------------------------------------------------------------------------------------------
x = np.array([1,2,3])
y = np.array([2,4,6])
X = np.array([[1,2,3], [4,5,6]]) # Two dimensional ararys
Y = np.array([[2,4,6], [8,10,12]])

x[2]
x[0:2]

z = x + y # element-wise addition
z

X[:,1] + Y[:,1] # add the first columns of both arrays
X[1,:] + Y[1,:] # add the first row of both arrays

X[1,:] == X[1] # equivalente

# Slicing NumPy Arrays: Question 2  ---
a = np.array([1,2])
b = np.array([3,4,5])
a + b # returns an error, different sizes

# Indexing NumPy Arrays ---------------------------------------------------------------------------------------------

z1 = np.array([1,3,5,7,9])
z2 = z1 + 1

ind = [0,2,3]

z1[ind]

ind = np.array([0,2,3]) # equivalent with np arrays
z1[ind]

# Boolean elements
z1 > 6
z1[z1 > 6] 
z2[z1 > 6]

# defining index with booleans 
ind = z1 > 6
z1[ind]

# with slicing, original vectors are modified, preferable to use indexes!!
z1
w = z1[0:3]
w[0] = 3
z1
# with indexes:
ind = [0,1,2]
w = z1[ind]
w[0] = 5
w
z1 # was not modified

# Building and Examining NumPy Arrays ----------------------------------------------------------------------------------------
np.linspace(0,100,10) # 100 is included
np.logspace(1, 2, 10) # logarithmically spaced elements

np.logspace(np.log10(250), np.log10(500), 10) # taking log base 10 of numbers

# dimensions (shape) and number of elements (size)
X = np.array([[1,2,3], [4,5,6]])
X.shape # dimensions
X.size # number of elements

# np has its own random module
np.random.random(10) # standard uniform distribution

np.any(x > 0.9) # if any element
np.all(x > 0.1) # if all elements


# Building and Examining NumPy Arrays: Question 1  ---
x = 20
not np.any([x%i == 0 for i in range(2, x)])
# Answers: finds whether x is prime