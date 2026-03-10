import numpy as np
import matplotlib.pyplot as plt

data1, data2, data3, data4, data5 = np.random.randn(5, 20)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data1, "o", label='data1')
ax.plot(data2, "d", label='data2')
ax.plot(data3, "v", label='data3')
ax.plot(data4, "s", label='data4')
ax.plot(data5, "p", label='data5')

ax.legend()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics as stats
from statistics import mean
from statistics import stdev
import matplotlib.patches as patches

X, Y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
Z = (1 - X / 2 + X ** 5 + Y ** 3) * np.exp(-(X ** 2) - Y ** 2)

fig, axs = plt.subplots(2, 2, figsize=(15, 15), layout="constrained")
pc = axs[0, 0].pcolormesh( X, Y, Z, vmin=-1, vmax=1, cmap="RdBu_r")
fig.colorbar(pc, ax=axs[0, 0])
axs[0, 0].set_title("pcolormesh()")

co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
fig.colorbar(co, ax=axs[0, 1])
axs[0, 1].set_title("contourf()")

pc = axs[1, 0].imshow(
    Z ** 2 * 100, cmap="plasma", norm=mpl.colors.LogNorm(vmin=0.01, vmax=100)
)

fig.colorbar(pc, ax=axs[1, 0], extend="both")
axs[1,0].set_title("imshow() with LogNorm()")

pc = axs[1, 1].scatter(data1, data2, c=data3, cmap="RdBu_r")
fig.colorbar(pc, ax=axs[1,1], extend='both')
axs[1, 1].set_title("scatter()")


import seaborn as sns

#Jointplot
penguins = sns.load_dataset("penguins")
sns.jointplot(
    data=penguins,
    x="flipper_length_mm",
    y="bill_length_mm",
    #y="bill_length_mm ", whitespace after variable name causes an error
    hue="sex",
    height=10
)
plt.show


sns.pairplot(
    data=penguins,
    y="bill_length_mm",
    #y="flipper_length_mm",
    #y="sex"
    hue="species"
)