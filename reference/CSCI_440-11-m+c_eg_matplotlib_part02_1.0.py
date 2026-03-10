#https://www.youtube.com/watch?v=4O_o53ag3ag #7 py data viz libraries in 15 min
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,5))  #create figure w/1 axes
ax.plot([1,2,3,4,5],[1,4,2,3,6])
plt.show


#create 4 random normal distribution datasets
import numpy as np
data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data1, "o", label='data1')
ax.plot(data2, "d", label='data2')
ax.plot(data3, "v", label='data3')
ax.plot(data4, "s", label='data4')
ax.legend()
#note: did not need plot.show

#histogram data with overlay
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


#football field

def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):

    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
            [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
            53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
            color='white')

    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    #endzones
    if endzones:
        ez1 = patches.Rectangle((0,0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110,0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
        plt.xlim(0, 120)
        plt.ylim(-5, 58.3)
        plt.axis('off')
        if linenumbers:
            for x in range(20, 110, 10):
                numb = x
                if x > 50:
                    numb = 120 - x
                plt.text(x, 5, str(numb - 10),
                        horizontalalignment='center',
                        fontsize=20, #fontname='Arial',
                        color='white')
                plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                        horizontalalignment='center',
                        fontsize=20, #fontname='Arial',
                        color='white', rotation=180)
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)
        
        for x in hash_range:
            ax.plot([x, x], [0.4, 0.7], color='white')
            ax.plot([x, x], [89.8, 52.5], color='white')
            ax.plot([x, x], [22.91, 23.57], color='white')
            ax.plot([x, x], [29.73, 30.39], color='white')

        if highlight_line:
            hl = highlight_line_number +10
            plt.plot([h1, h1], [0, 53.3], color='yellow')
            plt.text(h1 + 2, 50, '<- {}', format(highlighted_name),
                    color='yellow')
    return fig, ax

create_football_field()
plt.show()
        