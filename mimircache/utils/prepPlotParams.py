# coding=utf-8

import time


def prepPlotParams(title0, xlabel0, ylabel0, figname0, **kwargs):
    plot_dict = {}
    if 'title' in kwargs:
        plot_dict['title'] = kwargs['title']
    else:
        plot_dict['title'] = title0

    if 'xlabel' in kwargs:
        plot_dict['xlabel'] = kwargs['xlabel']
    else:
        plot_dict['xlabel'] = xlabel0

    if 'ylabel' in kwargs:
        plot_dict['ylabel'] = kwargs['ylabel']
    else:
        plot_dict['ylabel'] = ylabel0

    if 'figname' in kwargs:
        plot_dict['figname'] = kwargs['figname']
    else:
        pos = figname0.rfind('.')
        assert pos!=-1, "please provide suffix for output image"
        plot_dict['figname'] = "{}_{}.{}".format(figname0[:pos], int(time.time()), figname0[pos+1:])

    return plot_dict

