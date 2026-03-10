
#VS Code, menu bar, New Terminal
#pip install seaborn

#Why Seaborn
#https://seaborn.pydata.org/archive/0.11/tutorial/function_overview.html
#less code than matplotlib

#The seaborn namespace is flat; all of the functionality is accessible at the top level. 
#But the code itself is hierarchically structured, with modules of functions that achieve similar visualization goals through different means. Most of the docs are structured around these modules: you’ll encounter names like “relational”, “distributional”, and “categorical”.

#sample tutorial dataset
#https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv

#kde: kernel density plot
#A kernel density plot (KDE) is a smoothed, continuous visualization of a data distribution, 
# representing a smoothed version of a histogram. 
# It estimates the probability density function of a continuous variable by 
# placing a small "kernel" (often a Gaussian bell curve) over each data point and summing them. 

#inside VS Code
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#load example dataset
tips = sns.load_dataset("tips")

#create RelPlot: relative plot comparison
sns.relplot(
    data=tips,
    x="total_bill",
    y="tip",
    col="time",
    hue="smoker",
    style="smoker",
    size="size",
)
plt.show()

#create line graph overlay 
fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri,
    kind="line",
    x="timepoint",
    y="signal",
    col="region",
    hue="event",
    style="event",   
)
plt.show()

#create categorical swarm plot
sns.catplot(
    data=tips,
    kind="swarm",
    x="day",
    y="total_bill",
    hue="smoker",
    height=5,
    aspect=2.3
)
plt.show

#create categorical violin plot
#compare distributions intergroup (between groups)
#compare distributions intragroup (within groups) 
sns.catplot(
    data=tips,
    kind="violin",
    x="day",
    y="total_bill",
    hue="smoker",
    split=True,
    height=5,
    aspect=2.3
)
plt.show

#create categorical bar plot
#compare distributions intergroup (between groups)
sns.catplot(
    data=tips,
    kind="bar",
    x="day",
    y="total_bill",
    hue="smoker", #ADA 508 compliant: orange and blue
    height=5,
    aspect=2.3
)
plt.show

#Jointplot
penguins = sns.load_dataset("penguins")
sns.jointplot(
    data=penguins,
    x="flipper_length_mm",
    y="bill_length_mm",
    #y="bill_length_mm ", whitespace after variable name causes an error
    hue="species",
    height=10
)
plt.show

#Pairplot
#Why: Combines joint and marginal views — but rather than focusing on a single relationship, it visualizes every pairwise combination of variables simultaneously
sns.pairplot(
    data=penguins,
    hue="species"
)

#Kind parameter
#Allws you to quickly swap in a different representation:
sns.jointplot(
    data=penguins,
    x="flipper_length_mm",
    y="bill_length_mm",
    hue="species",
    kind="hist"
)