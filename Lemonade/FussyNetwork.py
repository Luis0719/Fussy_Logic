'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

import math
import numpy as np

MEDIAN = 0
DESV_STD = 1
GAUSSY_VALUES = 2

PARAM_P = 0
PARAM_Q = 1
PARAM_R = 2
PARAM_X = 3
PARAM_Y = 4

class GaussyModel():
    def __init__(self, x_elements, y_elements, rules, step, step_range, logger, debuglevel=0):
        self.x_elements = x_elements
        self.y_elements = y_elements
        self.rules = rules
        self.step = step
        self.step_range = step_range
        self.total_Steps = int((step_range[1] - step_range[0]) / self.step)
        self.logger = logger
        self.debuglevel = debuglevel

        for i in range(len(x_elements)):
            x_elements[i][GAUSSY_VALUES] = [0] * self.total_Steps
            y_elements[i][GAUSSY_VALUES] = [0] * self.total_Steps


    def log(self, text, debuglevel=0, logtype="INFO"):
        return
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            self.logger.write(msg + '\n')


    @staticmethod
    def evaluate_gaussy(x, median, desv_std):
        return math.e ** ((-0.5) * (((x-median) / desv_std) ** 2))

    @staticmethod
    def evaluate_fussy(x, y, p, q, r, gaussy_value):
        return gaussy_value * ((p * x) + (q * y) + r)

    @staticmethod
    def plot_ranges(axs, bottom_limit, top_limit, step, values):        
        axs.plot(np.arange(bottom_limit, top_limit, step), values)
    
    @staticmethod
    def plot_fussy_network(axs, bottom_limit, top_limit, step, Z):
        X = np.arange(bottom_limit, top_limit, step)
        Y = np.arange(bottom_limit, top_limit, step)
        X, Y = np.meshgrid(X, Y)
        Z = np.array(Z)

        axs.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)

    def fit(self):
        fussy_network_values = [[0 for x in range(self.total_Steps)] for y in range(self.total_Steps)]

        
        for i in range(self.step_range[0], self.step_range[1]):
            x = i*self.step

            for j in range(self.step_range[0], self.step_range[1]):
                y = j*self.step

                gaussy_subtotal = 0
                fussy_subtotal = 0

                self.log(f"X={x} Y={y}", 1)
                for element in self.rules:
                    x_gaussy_value = GaussyModel.evaluate_gaussy(x, element[PARAM_X][MEDIAN], element[PARAM_X][DESV_STD])
                    y_gaussy_value = GaussyModel.evaluate_gaussy(y, element[PARAM_Y][MEDIAN], element[PARAM_Y][DESV_STD])
                    
                    element[PARAM_X][GAUSSY_VALUES][i-self.step_range[0]] = x_gaussy_value
                    element[PARAM_Y][GAUSSY_VALUES][j-self.step_range[0]] = y_gaussy_value
                    xy_gaussy_product = x_gaussy_value * y_gaussy_value
                    self.log(f"{x_gaussy_value} * {y_gaussy_value} = {xy_gaussy_product}", 0)

                    gaussy_subtotal += xy_gaussy_product

                    fussy_value = GaussyModel.evaluate_fussy(x, y, element[PARAM_P], element[PARAM_Q], element[PARAM_R], xy_gaussy_product)
                    self.log(f"P={element[PARAM_P]} Q={element[PARAM_Q]} R={element[PARAM_R]} XY={xy_gaussy_product} Result={fussy_value}", 0)
                    fussy_subtotal += fussy_value

                if gaussy_subtotal == 0:
                    gaussy_subtotal = 1E-20  # Prevent division by 0 exception
                    
                self.log(f"{fussy_subtotal} / {gaussy_subtotal} network[{j-self.step_range[0]}][{i-self.step_range[0]}]={fussy_subtotal / gaussy_subtotal}", 1)
                fussy_network_values[j-self.step_range[0]][i-self.step_range[0]] = fussy_subtotal / gaussy_subtotal
                self.log("****************", 1)  

        return fussy_network_values


