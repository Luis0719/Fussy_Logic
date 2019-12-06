'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

import math
from numpy import arange

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    plt_found = True
except:
    plt_found = False

logger = open('logs.log', 'w')

class GaussyModel():
    def __init__(self, variables, x_step, bottom_limit, top_limit):
        self.variables = variables
        self.x_step = x_step
        self.bottom_limit = bottom_limit
        self.top_limit = top_limit
        self.range = top_limit - bottom_limit

        for element in variables:
            element['gaussy_values'] = [0] * self.range


    @staticmethod
    def evaluate_x(x, median, desv_std):
        return math.e ** ((-0.5) * (((x-median) / desv_std) ** 2))


    @staticmethod
    def evaluate_fussy(x, p, q, gaussy_value):
        return gaussy_value * ((p * x) + q)


    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')


    def config_plots(self):
        fig, axs = plt.subplots(1, 2)
        fig.figsize = (100,100)

        axs[0].set_xlabel('Temperature')
        axs[0].set_ylabel('Fussy Inference Network')
        axs[0].grid(True)

        axs[1].set_xlabel('Temperature')
        axs[1].set_ylabel('Membership Grade')
        axs[1].grid(True)

        return fig, axs


    def plot_ranges(self, plt, values):        
        plt.plot([i*self.x_step for i in range(self.bottom_limit, self.top_limit)], values)
        

    def fit(self):
        fig, axs = self.config_plots()

        fussy_network_values = [0] * self.range
        for i in range(self.bottom_limit, self.top_limit):
            x = i*self.x_step
            gaussy_subtotal = 0
            fussy_subtotal = 0

            for element in self.variables:
                gaussy_value = GaussyModel.evaluate_x(x, element['median'], element['desv_std'])
                element['gaussy_values'][i-self.bottom_limit] = gaussy_value
                gaussy_subtotal += gaussy_value

                fussy_value = GaussyModel.evaluate_fussy(x, element['p'], element['q'], gaussy_value)
                fussy_subtotal += fussy_value

            fussy_network_values[i-self.bottom_limit] = fussy_subtotal / gaussy_subtotal

            

        # Plot temperature graph
        for element in self.variables:
            self.plot_ranges(axs[1], element['gaussy_values'])

        # Plot fussy inference network
        self.plot_ranges(axs[0], fussy_network_values)

        plt.show()


