# Matplotlib and Pyplot
# Matplotlib is a big library. Pyplot is a collection of functions that make matplotlib work like Matlab
# Pyplot is useful for interactive work.
# Pyplot also knows LaTeX!!

import matplotlib.pyplot as plt
import numpy as np

plt.plot([0,1,4,9,16])
plt.plot([0,1,4,9,16]); # supress the printing of the object 

x = np.linspace(0,10,20)
y = x**2
plt.plot(x,y);

y1 = x**2
y2 = x**1.5

plt.plot(x,y1,"bo-"); #use blue, circles and solid lines AS IN MATLAB
plt.plot(x,y1,"bo-", linewidth = 2, markersize = 4);
plt.plot(x,y2,"gs-", linewidth = 2, markersize = 4); # gree, squares

# Introduction to Matplotlib and Pyplot: Question 2  ---
plt.plot([0,1,2],[0,1,4],"rd-")

# Customizing Your Plots ----------------------------------------------------------------------------------------------------
plt.plot(x,y1,"bo-", linewidth = 2, markersize = 12, label = "First")
plt.plot(x,y2,"gs-", linewidth = 2, markersize = 12, label = "Second")
plt.xlabel("$X$") # math mode to latex
plt.ylabel("$Y$")
plt.axis([-0.5, 10.5, -5, 105]) # [xmin, xman, ymin, ymax]
plt.legend(loc = "upper left")
plt.savefig("myplot.pdf")

# Plotting Using Logarithmic Axes --------------------------------------------------------------------------------------------
# semilogx() - plots the x-axis on a log scale ant the y in the original scale
# semilogy() - similar to X
# loglog() - both x and y on logarithmic scales
plt.loglog(x,y1,"bo-", linewidth = 2, markersize = 12, label = "First")
plt.loglog(x,y2,"gs-", linewidth = 2, markersize = 12, label = "Second")
plt.xlabel("$X$") # math mode to latex
plt.ylabel("$Y$")
plt.axis([1, 10.5, 1, 105]) # [xmin, xman, ymin, ymax]
plt.legend(loc = "upper left")
plt.savefig("myplot_log.pdf")

# Even spaces in the log axis we use np.logspace 
x = np.logspace(-1, 1, 40) # first point is 0.1 and last is 10
y1 = x**2
y2 = x**1.5
plt.loglog(x,y1,"bo-", linewidth = 2, markersize = 12, label = "First")
plt.loglog(x,y2,"gs-", linewidth = 2, markersize = 12, label = "Second")
plt.xlabel("$X$") # math mode to latex
plt.ylabel("$Y$")
plt.axis([1, 10.5, 1, 105]) # [xmin, xman, ymin, ymax]
plt.legend(loc = "upper left")
plt.savefig("myplot_log2.pdf")


# Generating Histograms -----------------------------------------------------------------------------------------------
# by default hist uses 10 evenly spaced bins and tries to optimize both bin width and bin locaions
x = np.random.normal(size = 1000)
plt.hist(x);
plt.hist(x, density = True); # density = True: proportion of obs or frequency instead of the total numbers (density formely called normed)
plt.hist(x, density = True, bins = np.linspace(-5,5,21)); # 20 bins

# subplots: 3 arguments (rows, col, plot number)
x = np.random.gamma(2,3,100000)
plt.hist(x, bins = 30)
plt.hist(x, bins = 30, density = True) # normalized
plt.hist(x, bins = 30, cumulative = True) # cumulative distribution
plt.hist(x, bins = 30, density = True, cumulative = True, histtype = "step") # both

# Create figure with 4 panels
plt.figure()
plt.subplot(221)
plt.hist(x, bins = 30)
plt.subplot(222)
plt.hist(x, bins = 30, density = True) # normalized
plt.subplot(223)
plt.hist(x, bins = 30, cumulative = True) # cumulative distribution
plt.subplot(224)
plt.hist(x, bins = 30, density = True, cumulative = True, histtype = "step") # both

