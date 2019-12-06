'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

import math
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    plt_found = True
except:
    plt_found = False


logger = open('logs.log', 'w')
MEDIAN = 0
DESV_STD = 1
GAUSSY_VALUES = 2

PARAM_P = 0
PARAM_Q = 1
PARAM_R = 2
PARAM_X = 3
PARAM_Y = 4

class GaussyModel():
    def __init__(self, x_elements, y_elements, rules, step, step_range, debuglevel=0):
        self.x_elements = x_elements
        self.y_elements = y_elements
        self.rules = rules
        self.step = step
        self.step_range = step_range
        self.total_Steps = int((step_range[1] - step_range[0]) / self.step)
        self.debuglevel = debuglevel

        for i in range(len(x_elements)):
            x_elements[i][GAUSSY_VALUES] = [0] * self.total_Steps
            y_elements[i][GAUSSY_VALUES] = [0] * self.total_Steps


    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')


    @staticmethod
    def evaluate_gaussy(x, median, desv_std):
        return math.e ** ((-0.5) * (((x-median) / desv_std) ** 2))

    @staticmethod
    def evaluate_fussy(x, y, p, q, r, gaussy_value):
        return gaussy_value * ((p * x) + (q * y) + r)


    def config_plots(self):
        fig = plt.figure(figsize=plt.figaspect(.4))
        # fig, axs = plt.subplots(2)
        # fig.figsize = (100,100)

        # axs[0].set_xlabel('Time')
        # axs[0].set_ylabel('Membership Grade')
        # axs[0].grid(True)

        # axs[0].set_xlabel('Money')
        # axs[0].set_ylabel('Membership Grade')
        # axs[0].grid(True)

        # return fig, axs
        ranges_axs = fig.add_subplot(1, 2, 1)
        network_axs = fig.add_subplot(1, 2, 2, projection='3d')
        return fig, ranges_axs, network_axs


    def plot_ranges(self, axs, values):        
        # plt.plot([i*self.step for i in range(self.step_range[0], self.step_range[1])], values)
        axs.plot(np.arange(self.step_range[0], self.step_range[1], self.step), values)
    

    def plot_fussy_network(self, axs):
        X = np.arange(self.step_range[0], self.step_range[1], self.step)
        Y = np.arange(self.step_range[0], self.step_range[1], self.step)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(X + Y)

        # fig = plt.figure()
        # ax = Axes3D(fig)
        axs.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)

    def fit(self):
        fig, ranges_axs, network_axs = self.config_plots()

        fussy_network_values = [[0 for x in range(self.total_Steps)] for y in range(self.total_Steps)]
        # fussy_network_values = [[0] * self.total_Steps]  * self.total_Steps
        self.log(fussy_network_values)
        for i in range(self.step_range[0], self.step_range[1]):
            x = i*self.step

            for j in range(self.step_range[0], self.step_range[1]):
                y = j*self.step

                gaussy_subtotal = 0
                fussy_subtotal = 0

                self.log(f"X={x} Y={y}")
                for element in self.rules:
                    x_gaussy_value = GaussyModel.evaluate_gaussy(x, element[PARAM_X][MEDIAN], element[PARAM_X][DESV_STD])
                    y_gaussy_value = GaussyModel.evaluate_gaussy(y, element[PARAM_Y][MEDIAN], element[PARAM_Y][DESV_STD])
                    
                    element[PARAM_X][GAUSSY_VALUES][i-self.step_range[0]] = x_gaussy_value
                    element[PARAM_Y][GAUSSY_VALUES][j-self.step_range[0]] = y_gaussy_value
                    xy_gaussy_product = x_gaussy_value * y_gaussy_value
                    self.log(f"{x_gaussy_value} * {y_gaussy_value} = {xy_gaussy_product}")

                    gaussy_subtotal += xy_gaussy_product

                    fussy_value = GaussyModel.evaluate_fussy(x, y, element[PARAM_P], element[PARAM_Q], element[PARAM_R], xy_gaussy_product)
                    self.log(f"P={element[PARAM_P]} Q={element[PARAM_Q]} R={element[PARAM_R]} XY={xy_gaussy_product} Result={fussy_value}")
                    fussy_subtotal += fussy_value

                self.log(f"{fussy_subtotal} / {gaussy_subtotal} network[{j-self.step_range[0]}][{i-self.step_range[0]}]={fussy_subtotal / gaussy_subtotal}")
                fussy_network_values[j-self.step_range[0]][i-self.step_range[0]] = fussy_subtotal / gaussy_subtotal
                self.log("****************")  

        # Plot temperature graph
        for i in range(len(self.x_elements)):
            self.plot_ranges(ranges_axs, self.x_elements[i][GAUSSY_VALUES])
            self.plot_ranges(ranges_axs, self.y_elements[i][GAUSSY_VALUES])
            

        self.log("--------------------------X1---------------------------")
        self.log(self.x_elements[0][GAUSSY_VALUES])
        self.log("--------------------------X2---------------------------")
        self.log(self.x_elements[1][GAUSSY_VALUES])
        self.log("--------------------------X3---------------------------")
        self.log(self.x_elements[2][GAUSSY_VALUES])
        self.log("--------------------------Y1---------------------------")
        self.log(self.y_elements[0][GAUSSY_VALUES])
        self.log("--------------------------Y2---------------------------")
        self.log(self.y_elements[1][GAUSSY_VALUES])
        self.log("--------------------------Y3---------------------------")
        self.log(self.y_elements[2][GAUSSY_VALUES])

        # Plot fussy inference network
        # self.plot_fussy_network(axs[0], fussy_network_values)
        self.plot_fussy_network(network_axs)

        if self.debuglevel <= 4:
            plt.show()

        return fussy_network_values


