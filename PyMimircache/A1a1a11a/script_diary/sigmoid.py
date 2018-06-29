# coding=utf-8

import math
import numpy as np
from scipy.optimize import curve_fit
from  scipy.optimize import minimize
from functools import partial


def sigmoid_fit(xdata, ydata, func_name):
    """
    this function fits xdata and ydata
    :param xdata:
    :param ydata:
    :return:
    """
    sigmoid = globals()[func_name]

    if xdata is None or ydata is None:
        xdata = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
        ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])

    if "min" in func_name:
        partial_func = partial(sigmoid, x=xdata, y=ydata)
        res = minimize(partial_func, np.array((2, -6)))
        popt = res.x
        func = globals()[func_name[4:]]
    else:
        if "tanh2" in func_name:
            popt, pcov = curve_fit(sigmoid, xdata, ydata, p0=(0.001, 0), maxfev=24000)
        else:
            popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=24000)
            # popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=2400)
        func = sigmoid

    return popt, func


def sigmoid_fit2(xdata, ydata, func_name):
    """
    this function fits xdata and ydata
    :param xdata:
    :param ydata:
    :return:
    """
    assert "min" in func_name, "sigmoid_fit2 can only be applied to min func"
    sigmoid = globals()[func_name]

    partial_func = partial(sigmoid, x=xdata, y=ydata)
    res = minimize(partial_func, np.array((2, -6)))
    # print(res)


    return res.x, globals()[func_name[4:]]



def get_func(func_name):
    return globals()[func_name]

def min_arctan(arg, x, y):
    b, c = arg
    x = np.array(x)
    y_cal = 1/(np.pi/2) * np.arctan(b * (x + c))
    return np.sum((y_cal-y) ** 2)

def min_arctan3(arg, x, y):
    b, c = arg
    x = np.array(x)
    y_cal = 1/(np.pi) * (np.arctan(b * (x + c))) + 0.5
    return np.sum((y_cal-y) ** 2)



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

def tanh2(x, a, b):
    y = np.tanh(a * x + b) / 2 + 0.5
    return y

def arctan0(x, a, b, c):
    y = a * np.arctan(b * (x + c))
    return y

def arctan(x, b, c):
    # don't use this, it will cause fitting to fail more often
    # y = 1/(math.pi/2) * math.atan(b * (x + c))
    y = 1/(np.pi/2) * np.arctan(b * (x + c))
    return y

def arctan_inv(y, b, c):
    x = math.tan(y * (math.pi/2)) / b - c
    return x

# def arctan2(x, b, c, d):
#     y = 1/(np.pi) * (np.arctan(b * (x + c)) + d)
#     return y

# def arctan_inv2(y, b, c, d):
#     x = math.tan((y-d) * (math.pi)) / b - c
#     return x

def arctan3(x, b, c):
    y = 1/(np.pi) * (np.arctan(b * (x + c))) + 0.5
    return y

def arctan_inv3(y, b, c):
    x = math.tan((y-0.5) * (math.pi)) / b - c
    return x

def sigmoid2(x, a):
    y = x / np.sqrt(a + x * x)
    return y


if __name__ == "__main__":
    # print(arctan(1000, 5.9e-5, 692.52))
    xdata = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
    ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])
    print(sigmoid_fit2(xdata, ydata, "min_arctan"))
    sigmoid_fit(xdata, ydata, "arctan")