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
    def __init__(self, variables, x_step, bottom_limit=-200, top_limit=500):
        self.variables = variables
        self.x_step = x_step
        self.bottom_limit = bottom_limit
        self.top_limit = top_limit


    @staticmethod
    def evaluate_x(x, median, desv_std):
        return math.e ** ((-0.5) * (((x-median) / desv_std) ** 2))

    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')


    def plot_ranges(self, plt):        
        for temp in self.variables:
            values = self.get_function_values(temp)

            if not plt_found:
                print("Matplotlib libary not found. Install it to see the graph. Install: 'pip install matplotlib'")
                return

            plt.plot([i*self.x_step for i in range(self.bottom_limit, self.top_limit)], values)

        
        plt.set_xlabel('Temperature')
        plt.set_ylabel('Membership Grade')
        plt.grid(True)


    def get_function_values(self, variable):
        desv_std = variable['desv_std']
        median = variable['median']

        result = [0] * (self.top_limit - self.bottom_limit)
        for i in range(self.bottom_limit, self.top_limit):
            x = i*self.x_step
            value = GaussyModel.evaluate_x(x, median, desv_std)

            result[i-self.bottom_limit] = value

        return result


    def get_fussy_values(self):
        result = [0] * (self.top_limit - self.bottom_limit)
        function_values = [0] * len(self.variables) 
        processed_values = [0] * len(self.variables)

        print(self.variables)

        for i in range(self.bottom_limit, self.top_limit):
            x = i*self.x_step
            for j in range(len(self.variables)):
                median = self.variables[j]['median']
                desv_std = self.variables[j]['desv_std']
                function_values[j] = GaussyModel.evaluate_x(x, median, desv_std)

            for j in range(len(self.variables)):
                p = self.variables[j]['p']
                q = self.variables[j]['q']
                processed_values[j] = p*function_values[j] + q

            result[i-self.bottom_limit] = sum(processed_values) / sum(function_values)
            
        return result
    

    def plot_fussy_inference_network(self, plt):
        values = self.get_fussy_values()
        plt.plot([i*self.x_step for i in range(self.bottom_limit, self.top_limit)], values)
        plt.set_xlabel('Temperature')
        plt.set_ylabel('')
        plt.grid(True)

    def plot(self):
        fig, axs = plt.subplots(1, 2)
        fig.figsize = (100,100)
        # axs[0].title.set_text(f"Generation {generation}")
        self.plot_fussy_inference_network(axs[0])
        self.plot_ranges(axs[1])

        plt.show()

    def fit(self):
        self.plot()


