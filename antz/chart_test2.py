import random
import threading
from antz.chart import live_chart


class SolutionChartThread(threading.Thread):
    def run(self):
        def data_gen():
            t = data_gen.t
            while True:
                yield t, random.random()
                t += 0.5

        data_gen.t = 0

        live_chart(data_gen)


solution_chart_thread = SolutionChartThread()
solution_chart_thread.start()