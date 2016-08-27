

import time


def prepPlotParams(title, xlabel, ylabel, figname, **kwargs):
    plot_dict = {}
    if 'title' in kwargs:
        plot_dict['title'] = kwargs['title']
    else:
        plot_dict['title'] = title

    if 'xlabel' in kwargs:
        plot_dict['xlabel'] = kwargs['xlabel']
    else:
        plot_dict['xlabel'] = xlabel

    if 'ylabel' in kwargs:
        plot_dict['ylabel'] = kwargs['ylabel']
    else:
        plot_dict['ylabel'] = ylabel

    if 'figname' in kwargs:
        plot_dict['figname'] = kwargs['figname']
    else:
        pos = figname.rfind('.')
        assert pos!=-1, "please provide suffix for output image"
        plot_dict['figname'] = "{}_{}.{}".format(figname[:pos], int(time.time()), figname[pos+1:])

    return plot_dict

