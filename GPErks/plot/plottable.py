from abc import ABCMeta, abstractmethod

from GPErks.plot.options import PlotOptions


class Plottable(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, plot_options: PlotOptions = None):
        if plot_options is None:
            plot_options = PlotOptions()
        pass
