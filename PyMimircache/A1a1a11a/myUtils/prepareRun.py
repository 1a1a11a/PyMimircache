# coding=utf-8

import os, sys, time


def prepare(figname, folder, plot_on_exist=False):

    if not folder:
        folder = os.path.dirname(figname)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if (not plot_on_exist) and os.path.exists("{}/{}".format(folder, figname)):
        return True

    return False


