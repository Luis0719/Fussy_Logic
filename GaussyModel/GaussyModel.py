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
    def __init__(self, variables, x_step):
        self.variables = variables
        self.x_step = x_step


    @staticmethod
    def evaluate_x(x, median, desv_std):
        return math.e ** ((-1/2) * (((x-median) / desv_std) ** 2))

    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')


    def plot_ranges(self, variables):
        plt.rcParams.update({'font.size': 10})
        
        for temp in variables:
            values = self.get_function_values(temp)

            if not plt_found:
                print("Matplotlib libary not found. Install it to see the graph. Install: 'pip install matplotlib'")
                return

            plt.plot([i*self.x_step for i in range(0, 500)], values)
        
        plt.show()


    def get_function_values(self, variable):
        bottom = variable['bottom']
        top = variable['top']
        desv_std = variable['desv_std']
        median = bottom + ((top - bottom)/2)

        print(f"Bottom: {bottom} top: {top} median: {median} dsv_std: {desv_std}")
        top = int(top / self.x_step)

        result = [0] * 500
        for i in range(0, 500):
            x = i*self.x_step
            value = variableModel.evaluate_x(x, median, desv_std)

            result[i] = value

        # print(result)
        return result

    
    def plot_fussy_inference_network():
        pass

    def fit(self):
        self.plot_ranges(self.variables)


