#CSCI_440-10-m+c_eg_py_matplotlib

#https://matplotlib.org/stable/gallery/index.html
#https://matplotlib.org/stable/users/explain/quick_start.html


import matplotlib.pyplot as plt
import numpy as np

#1. Simple Line Plot: A basic line plot connects a series of (x, y) coordinates. If the x-points are not specified, they default to 0, 1, 2, ... based on the length of the y-points array. 
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints)
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.title("Simple Line Plot Example")
plt.show()

#2. Scatter Plot: A scatter plot uses 'o' (rings) as a shortcut marker notation to display only the points and not the connecting lines. 
# Data for two sets of points
x1 = [2, 4, 6, 8]
y1 = [1, 3, 7, 5]
x2 = [1, 3, 5, 7]
y2 = [2, 4, 6, 8]

plt.scatter(x1, y1, label="Data Set 1", color="blue")
plt.scatter(x2, y2, label="Data Set 2", color="red")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.title("Scatter Plot Example")
plt.legend() # Shows the legend with labels
plt.show()

#3. Bar Chart: Bar charts are effective for visualizing categorical data. 
names = ['Group A', 'Group B', 'Group C']
values = [1, 10, 100]

plt.bar(names, values)
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Bar Chart Example")
plt.show()

#4. Creating Subplots: The pyplot.subplots() function is the simplest way to create a Figure and a set of Axes (the actual plotting areas). 
# Sample data
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
z = np.cos(x)

# Create a figure and a set of subplots (2 rows, 1 column)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle('Vertically Stacked Subplots')

# Plot data on the first subplot (top)
ax1.plot(x, y, 'tab:blue')
ax1.set_title('Sine Squared') # Set title for individual subplot
ax1.set_ylabel('Y values')

# Plot data on the second subplot (bottom)
ax2.plot(x, z, 'tab:orange')
ax2.set_title('Cosine')
ax2.set_xlabel('X values')
ax2.set_ylabel('Z values')

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap
plt.show()


#5. Pretty Plots
np.random.seed(19680801)  # seed the random number generator.
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')