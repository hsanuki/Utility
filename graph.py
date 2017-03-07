# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fp_axis = FontProperties(fname='/Library/Fonts/HelveticaNeue.ttf', size=40)
fp_ticks = FontProperties(fname='/Library/Fonts/HelveticaNeue.ttf', size=32)
fp_title = FontProperties(fname='/Library/Fonts/HelveticaNeue.ttf', size=32)
fp_legend = FontProperties(fname='/Library/Fonts/HelveticaNeue.ttf', size=32)


def plot_timeseries(clf, series1):
    series1.index = Utility.convert_indexdatetime(series1.index, "min")
    plt.rcParams['xtick.major.pad']='8'
    plt.rcParams['ytick.major.pad']='8'
    plt.figure(1, figsize=(clf.fig_size, 5), facecolor='white')
    plt.subplot('111', axisbg='white')
    plt.plot(series1.index.values, series1.values, "-", c="k", linewidth=3)
    plt.xlabel('Time [min]', fontproperties=clf.fp_axis, labelpad=10)
    plt.ylabel('Amplitude [a.u.]', fontproperties=clf.fp_axis, labelpad=10)
    plt.xlim(0, 36)
    plt.xticks(fontproperties=clf.fp_ticks)
    plt.yticks(fontproperties=clf.fp_ticks)
    plt.grid(True)
    plt.savefig('/Users/sanukiharuyuki/Desktop/tmp.png', bbox_inches='tight', pad_inches=0.2)
    plt.show()
