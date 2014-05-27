"""
Adds some charts
"""
import threading

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def live_chart(data_gen):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 300)
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

    # noinspection PyUnusedLocal
    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
                                  repeat=False)

    plt.show()


class SolutionChartThread(threading.Thread):
    def __init__(self, ctx):
        threading.Thread.__init__(self)
        self._ctx = ctx

    def run(self):
        def data_gen():
            t = data_gen.t
            ctx = data_gen.ctx
            while True:
                if ctx.runner:
                    sollen = len(ctx.runner.solutions)
                    yield t, sollen
                    t += 0.5

        data_gen.t = 0
        data_gen.ctx = self._ctx

        live_chart(data_gen)
