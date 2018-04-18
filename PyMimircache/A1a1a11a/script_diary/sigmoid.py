# coding=utf-8

import math
import numpy as np
from scipy.optimize import curve_fit


def sigmoid_fit(xdata, ydata, func_name):
    """
    this function fits xdata and ydata
    :param xdata:
    :param ydata:
    :return:
    """
    sigmoid = globals()[func_name]
    # print(sigmoid)

    if xdata is None or ydata is None:
        xdata = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
        ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    # print(popt)
    return popt, sigmoid


def get_func(func_name):
    return globals()[func_name]


def logistic(x, beta, gamma):
    y = 1 / (1 + np.exp(-beta * (x - gamma)))
    return y


def gompertz(x, b, c):
    y = np.exp(-b * np.exp(-c * x))
    return y


def richard(x, b, c, d):
    y = (1 + (b - 1) * np.exp(-c * (x - d))) ** (1 / (1 - b))
    return y


def sigmoid1(x, a, b):
    y = x / (a + np.abs(x)) ** b
    return y


def tanh(x, a, b, c):
    y = a * np.tanh(b * (x + c))
    return y

def tanh2(x, b, c):
    y = 1 * np.tanh(b * (x + c))
    return y

def arctan2(x, a, b, c):
    y = a * np.arctan(b * (x + c))
    return y

def arctan(x, b, c):
    # don't use this, it will cause fitting to fail more often
    # y = 1/(math.pi/2) * math.atan(b * (x + c))
    y = 1/(math.pi/2) * np.arctan(b * (x + c))
    return y

def arctan_inv(y, b, c):
    x = math.tan(y * (math.pi/2)) / b - c
    return x



def sigmoid2(x, a):
    y = x / np.sqrt(a + x * x)
    return y


if __name__ == "__main__":
    print(arctan(1000, 5.9e-5, 692.52))