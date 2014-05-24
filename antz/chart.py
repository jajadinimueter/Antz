"""
Adds some charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def live_chart(data_gen):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 800)
    ax.set_xlim(0, 100)
    ax.grid()

    xdata, ydata = [], []

    def run(data):
        # update the data
        t, y = data
        xdata.append(t)
        ydata.append(y)

        xmin, xmax = ax.get_xlim()

        if t >= xmax:
            # xmin += (xmax - xmin) / 2
            ax.set_xlim(xmin, xmax + ((xmax - xmin) / 2))
            ax.figure.canvas.draw()

        line.set_data(xdata, ydata)

        return line,

    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
                                  repeat=False)

    plt.show()