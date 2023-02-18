# Scope Rules -----------------------------------------------------------------------------------------------------------
# if I have more than one variable, function called with the same way, python searches for the object layer by layer. 
# Moving from inner layers to outer layers
# LEGB 
# Local
# Enclosing Function
# Global
# Built-in
def update(n,x):
    n = 2
    x.append(4)
    print("update: ", n, x)

def main():
    n = 1
    x = [0,1,2,3]
    print("main: ", n, x)
    update(n,x)
    print("main: ", n, x)

main()


# Scope Rules: Question 1 ---
def increment(n):
    n += 1
    print(n)

# What will this print? 
n = 1
increment(n)
print(n)

# Scope Rules: Question 2 ---
# Fill in the #blank# to ensure this prints 10. 
def increment(n):
    n += 1
    return n#blank#

n = 1
while n < 10:
    n = increment(n)
print(n)

# Classes and Object-Oriented Programming ------------------------------------------------------------------------------
ml = [5,9,3,6,8,11,4,3]
ml.sort()
ml

ml.remove(3) # only first occurances are remove

# Define a new class
# the functions defined inside a class are known as "instance methods"
# by convention the name of the class instance is called "self", is always passed as the first argument
class Mylist(list): 
    def remove_min(self): 
        self.remove(min(self))
    def remove_max(self):
        self.remove(max(self))

x = [10,3,5,1,2,7,6,4,8]
y = Mylist(x)
dir(y) # we can find the methods created available
y.remove_min()
y
y.remove_max()
y

# Classes and Object-Oriented Programming: Question 1  ---

class NewList(list):
    def remove_max(self):
        self.remove(max(self))
    def append_sum(self):
        self.append(sum(self))

x = NewList([1,2,3])
while max(x) < 10:
    x.remove_max()
    x.append_sum()

print(x) # DO NOT RUN, THE WHILE NEVER STOPS

