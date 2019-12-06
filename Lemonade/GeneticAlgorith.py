'''
    Owner: Luis Eduardo Hernandez Ayala
    Email: luis.hdez97@hotmail.com
    Python Version: 3.7.x, but shouldn't have problems with 2.7.x
'''

from math import sqrt, cos, sin
from FussyNetwork import GaussyModel
import numpy as np
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    plt_found = True
except:
    plt_found = False

logger = open('logs.log', 'w')


RESOLUTION_MEDIAN = 0
RESOLUTION_DESV_STD = 1
RESOLUTION_P = 2
RESOLUTION_Q = 3
RESOLUTION_R = 4

MEDIAN_START = 0
MEDIAN_END = 6
DESV_STD_START = 6
DESV_STD_END = 12
PARAM_P_START = 12
PARAM_P_END = 21
PARAM_Q_START = 21
PARAM_Q_END = 30
PARAM_R_START = 30
PARAM_R_END = 39


class FussyParameters():
    def __init__(self, cromosome):
        X = [
            [
                cromosome[0], # Median
                cromosome[6], # Desv_std
                None          # Gaussy_values
            ],
            [
                cromosome[0], # Median
                cromosome[6], # Desv_std
                None          # Gaussy_values
            ],
            [
                cromosome[2], # Median
                cromosome[8], # Desv_std
                None          # Gaussy_values
            ]
        ]

        Y = [
            [
                cromosome[3],
                cromosome[9],
                None
            ],
            [
                cromosome[4],
                cromosome[10],
                None
            ],
            [
                cromosome[5],
                cromosome[11],
                None
            ]
        ]


        fussy_rules = [
            [
                cromosome[12], # p
                cromosome[21], # q
                cromosome[30], # r
                X[0], # X
                Y[0]  # Y
            ],
            [
                cromosome[13],
                cromosome[22],
                cromosome[31],
                X[0],
                Y[1]
            ],
            [
                cromosome[14],
                cromosome[23],
                cromosome[32],
                X[0],
                Y[2]
            ],

            [
                cromosome[15],
                cromosome[24],
                cromosome[33],
                X[1],
                Y[0]
            ],
            [
                cromosome[16],
                cromosome[25],
                cromosome[34],
                X[1],
                Y[1]
            ],
            [
                cromosome[17],
                cromosome[26],
                cromosome[35],
                X[1],
                Y[2]
            ],

            [
                cromosome[18],
                cromosome[27],
                cromosome[36],
                X[2],
                Y[0]
            ],
            [
                cromosome[19],
                cromosome[28],
                cromosome[37],
                X[2],
                Y[1]
            ],
            [
                cromosome[20],
                cromosome[29],
                cromosome[38],
                X[2],
                Y[2]
            ]
        ]

        self.X = X
        self.Y = Y
        self.fussy_rules = fussy_rules
    
class PSmodel():
    def __init__(self, population_size, cromosome_size, target_values, resolutions, generations, x_bottom_limit, x_top_limit, x_step, competidors_percentage=0.05, mutation_percentage=0, gen_bit_length=8, elitism=False, graph_generations=False, debuglevel=0):
        # Validate population size to be an even number
        if population_size % 2 == 1:
            raise Exception("Population size must be an even number")
        self.population_size = population_size

        # Validate the competidors percentage to be between 1 and 100
        if competidors_percentage < 0.1 and competidors_percentage > 0.99:
            raise Exception("Competidor percentage must be a number between 0.1 and 0.99")
        self.competidors_percentage = competidors_percentage

        if mutation_percentage < 0 and mutation_percentage > 1:
            raise Exception("Mutations percentage must be a number between 0 and 1")
        self.total_mutations = int(population_size * mutation_percentage)

        self.cromosome_size = cromosome_size
        self.generations = generations
        self.gen_bit_length = gen_bit_length
        self.resolutions = resolutions
        self.elitism = elitism
        self.x_bottom_limit = x_bottom_limit
        self.x_top_limit = x_top_limit
        self.x_step = x_step
        self.graph_generations = graph_generations
        self.debuglevel = debuglevel

        self.population = None
        self.fittiests_history = None
        self.fittiest = None

        print(f"Target = {target_values}")
        self.target = target_values


    @staticmethod
    def get_y(x, cromosome):
        # return cromosome[A] * (cromosome[B] * sin(x/cromosome[C]) + cromosome[D] * sin(x/cromosome[E])) + cromosome[F] * x - cromosome[D]
        return cromosome[A] * (cromosome[B] * sin(x/cromosome[C]) + cromosome[D] * sin(x/cromosome[E])) + cromosome[F] * x - cromosome[D]

    @staticmethod
    def gen_to_binary(gen, gen_bit_length):
        format_schema = "{0:" + "{0:02d}".format(gen_bit_length) + "b}"
        return format_schema.format(gen)

    @staticmethod
    def bin_to_dec(binary_value):
        return int(binary_value, 2)

    @staticmethod
    def flip_bit(bit):
        if bit == "1":
            return "0"
        
        return "1"
        
    @staticmethod
    def is_valid_cromosome(cromosome):
        return not 0 in cromosome[DESV_STD_START:DESV_STD_END]:

    @staticmethod
    def resolutionate(values, resolution):
        result = [0] * len(values)
        for i in range(len(values)):
            result[i] = values[i] / resolution

        return result


    def get_resolutionated_cromosome(self, cromosome):
        resolutionated_cromosome = self.resolutionate(cromosome[MEDIAN_START:MEDIAN_END], self.resolutions[RESOLUTION_MEDIAN])
        resolutionated_cromosome += self.resolutionate(cromosome[DESV_STD_START:DESV_STD_END], self.resolutions[RESOLUTION_DESV_STD])
        resolutionated_cromosome += self.resolutionate(cromosome[PARAM_P_START:PARAM_P_END], self.resolutions[RESOLUTION_P])
        resolutionated_cromosome += self.resolutionate(cromosome[PARAM_Q_START:PARAM_Q_END], self.resolutions[RESOLUTION_Q])
        resolutionated_cromosome += self.resolutionate(cromosome[PARAM_R_START:PARAM_R_END], self.resolutions[RESOLUTION_R])
        
        return resolutionated_cromosome


    def get_function_values(self, cromosome):
        resolutionated_cromosome = self.get_resolutionated_cromosome(cromosome)

        parameters = FussyParameters(resolutionated_cromosome)
        fussy_model = GaussyModel(
                        x_elements = parameters.X, 
                        y_elements = parameters.Y, 
                        rules = parameters.fussy_rules, 
                        step = self.x_step, 
                        step_range = [self.x_bottom_limit,self.x_top_limit], 
                        logger = logger,
                        debuglevel = self.debuglevel)

        return fussy_model.fit()
    

    def evaluate(self, cromosome):
        error = 0
        cromosome_values = self.get_function_values(cromosome)
        for j in range(len(self.target)):
            for i in range(len(self.target[j])):
                error += abs(cromosome_values[j][i] - self.target[j][i])
        
        return error


    def log(self, text, debuglevel=0, logtype="INFO"):
        if self.debuglevel <= debuglevel:
            msg = "{} - {}".format(logtype, text)
            print(msg)
            logger.write(msg + '\n')     
        

    def generate_population(self):  
        '''
            Generate a population where each individual consists of an array of 3 numbers which can go from 0 to 2^gen_bit_length-1 (gen_bit_length is received as a parameter, default=8)
        '''
        max_value = 2 ** self.gen_bit_length
        temp_population = [0] * self.population_size
        for i in range(self.population_size):
            temp_population[i] = [0] * (self.cromosome_size + 1)
            valid_cromosome = False
            while not valid_cromosome:
                for j in range(self.cromosome_size):
                    temp_population[i][j] = random.randrange(0, max_value)
                valid_cromosome = PSmodel.is_valid_cromosome(temp_population[i])

        return temp_population


    def get_fittiest(self):
        fittiest = None
        for cromosome in self.population:
            if fittiest == None or cromosome[-1] < fittiest[-1]: 
                fittiest = cromosome

        return fittiest


    def calculate_aptitud_function(self):
        '''
            Calculate the aptitud function for each cromosome and store the result in the last position of the list
        '''
        for cromosome in self.population:
            cromosome[-1] = self.evaluate(cromosome)


    def tournament_compete(self, total_competidors):
        '''
            Get an 'n' number of 'randomly' selected cromosomes and get the one with the lowest value of the aptitud function
            'n' is given by the parameter 'total_competidors'
        '''
        self.log("-----start of tournament with {:d} competidors-----".format(total_competidors))
        winner = None
        for i in range(total_competidors):
            rndindex = random.randrange(self.population_size)
            self.log("{}".format(str(self.population[rndindex])))

            # If there's still no winner (first item in the loop) just assign it to be the temporal winner
            if winner == None or self.population[rndindex][-1] < winner[-1]:
                winner = self.population[rndindex]
                continue

        self.log("Winner = {}".format(str(winner)))
        self.log("-----end of tournament-----")
        return winner


    def tournament(self):
        '''
            Make a tournament to get a new list of 'n' cromosomes. 'n' is the total population size / 2. 
            The self.logic to get the cromosomes for this new list is delegated to the method 'tournament_compete'
        '''
        total_competidors = int(self.population_size * self.competidors_percentage)
        total_winners = int(self.population_size / 2)

        winners = [0] * total_winners
        for i in range(total_winners):
            winners[i] = self.tournament_compete(total_competidors)

        self.log('------winners tournament----')
        self.print_population(winners)
        return winners


    def breeding_operator1(self, father, mother):
        '''
            This breeding method will get a pivot randomely and will use it to 'break' each cromosome (father's and mother's cromosomes)
            Child1 will consist on father's binary value from position 0 to pivot, and mother's binary value from pivot to last position
            Child2, on the other hand, will consist on mothers's binary value from position 0 to pivot, and fathers's binary value from pivot to last position
        '''
        def cromosome_to_binary(cromosome, gen_bit_length):
            result = ""
            for i in range(len(cromosome)-1):
                result += PSmodel.gen_to_binary(cromosome[i], gen_bit_length)
            return result

        def binary_to_cromosome(binary, gen_bit_length):
            result = []
            for i in range(0, len(binary), gen_bit_length):
                result.append(PSmodel.bin_to_dec(binary[i:i+gen_bit_length]))

            return result


        # Get random pivot which will divide the cromosome
        pivot = random.randrange(1, self.gen_bit_length * self.cromosome_size)

        # Convert each cromsome to it's binary equivalent. This get the binary value of each gen and will merge them into one
        father_binary = cromosome_to_binary(father, self.gen_bit_length)
        mother_binary = cromosome_to_binary(mother, self.gen_bit_length)

        # Do the breeding
        child1_binary = father_binary[:pivot] + mother_binary[pivot:]
        child2_binary = mother_binary[:pivot] + father_binary[pivot:]

        # Split the final binary value into each gen
        child1 = binary_to_cromosome(child1_binary, self.gen_bit_length) + [0]
        child2 = binary_to_cromosome(child2_binary, self.gen_bit_length) + [0]

        self.log("{} & {} ({:d})= {} & {}".format(str(father), str(mother), pivot, str(child1), str(child2)), 2)
        
        return child1, child2


    def breeding_operator2(self, father, mother):
        def cromosome_to_binary(cromosome, gen_bit_length):
            result = ""
            for i in range(len(cromosome)-1):
                result += PSmodel.gen_to_binary(cromosome[i], gen_bit_length)
            return result

        def binary_to_cromosome(binary, gen_bit_length):
            result = []
            for i in range(0, len(binary), gen_bit_length):
                result.append(PSmodel.bin_to_dec(binary[i:i+gen_bit_length]))

            return result


        # Get random pivot which will divide the cromosome
        pivot1 = random.randrange(1, self.gen_bit_length * self.cromosome_size)
        pivot2 = random.randrange(1, self.gen_bit_length * self.cromosome_size)

        if pivot1 > pivot2:
            tmp = pivot1
            pivot1 = pivot2
            pivot2 = tmp

        # Convert each cromsome to it's binary equivalent. This get the binary value of each gen and will merge them into one
        father_binary = cromosome_to_binary(father, self.gen_bit_length)
        mother_binary = cromosome_to_binary(mother, self.gen_bit_length)

        # Do the breeding
        child1_binary = father_binary[:pivot1] + mother_binary[pivot1:pivot2] + father_binary[pivot2:]
        child2_binary = mother_binary[:pivot1] + father_binary[pivot1:pivot2] + mother_binary[pivot2:]

        # Split the final binary value into each gen
        child1 = binary_to_cromosome(child1_binary, self.gen_bit_length) + [0]
        child2 = binary_to_cromosome(child2_binary, self.gen_bit_length) + [0]

        self.log("{} & {} ({:d})= {} & {}".format(str(father), str(mother), pivot, str(child1), str(child2)), 2)
        
        return child1, child2


    def breether_factory(self, father, mother):
        i = random.randrange(0, 100)
        # i = 20
        if i > 20:
            return self.breeding_operator1(father, mother)
        else:
            return self.breeding_operator1(father, mothers)


    def breed(self, fathers, mothers):
        '''
            select the breeding method
        '''
        self.log('------breeding-------')
        newpopulation = [0] * self.population_size

        for i in range(int(self.population_size/2)):
            valid_children = False
            while not valid_children:
                child1, child2 = self.breeding_operator1(fathers[i], mothers[i])
                valid_children = PSmodel.is_valid_cromosome(child1) and PSmodel.is_valid_cromosome(child2)

            newpopulation[i*2], newpopulation[i*2+1] = child1, child2

        self.log('------end of breeding-------')
        return newpopulation


    def apply_elitism(self, fathers, mothers, children):
        self.log("**** Applying elitism ****", debuglevel=0)
        tmp = fathers + mothers + children
        self.print_population(tmp, debuglevel=0)
        self.log("Sorted population", debuglevel=0)
        tmp = sorted(tmp, key=lambda x:x[-1])
        self.print_population(tmp, debuglevel=0)
        result = tmp[:self.population_size]
        self.log("**** Elitism result ****")
        self.print_population(result, debuglevel=0)
        self.log("**** End of Elitism ****")
        return result  


    def mutate_gen(self, gen, bit_index):
        binary = list(PSmodel.gen_to_binary(gen, self.gen_bit_length))
        binary[bit_index] = PSmodel.flip_bit(binary[bit_index])
        binary = "".join(binary)
        return PSmodel.bin_to_dec(binary)


    def mutate_cromosome(self, cromosome, gen_index, bit_index=None):
        # Get which bit to mutate
        if bit_index == None:
            bit_index = random.randrange(0, self.gen_bit_length)

        # Apply mutation to gen
        cromosome[gen_index] = self.mutate_gen(cromosome[gen_index], bit_index)
        
        return cromosome

    
    def mutation(self):
        for i in range(self.total_mutations):
            # Get which cromosome to mutate
            rnd_index = random.randrange(0, self.population_size)
            selected_cromosome = self.population[rnd_index]

            log_builder = f"Cromosome mutated from {selected_cromosome} to -> "

            # Get which gen to mutate from the cromosome
            rnd_gen_index = random.randrange(0, self.cromosome_size)

            # Apply mutation
            mutated_cromosome = self.mutate_cromosome(selected_cromosome, rnd_gen_index)
            while(not PSmodel.is_valid_cromosome(mutated_cromosome)):
                mutated_cromosome = self.mutate_cromosome(selected_cromosome, rnd_gen_index)

            self.population[rnd_index] = mutated_cromosome
            self.log(log_builder + str(selected_cromosome), 1)


    def graph_history(self):
        if not plt_found:
            print("Matplotlib libary not found. Install it to see the graph. Install: 'pip install matplotlib'")
            return

        plt.rcParams.update({'font.size': 6})
        plt.plot([i+1 for i in range(self.generations)], self.fittiests_history)
        plt.ylabel('Error')
        plt.xlabel('Generations')
        plt.show()


    def plot_data(self, generation):
        fig = plt.figure(figsize=plt.figaspect(0.3))
        fig.suptitle(f"Generation #{generation}")

        X = np.arange(self.x_bottom_limit, self.x_top_limit, self.x_step)
        Y = np.arange(self.x_bottom_limit, self.x_top_limit, self.x_step)
        X, Y = np.meshgrid(X, Y)

        # Target
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        Z = np.array(self.target)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)

        # Generation fittiest
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        Z = np.array(self.get_function_values(self.fittiest))
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)

        # History
        ax = fig.add_subplot(1, 3, 3)
        ax.plot([i+1 for i in range(generation)], self.fittiests_history)

        # fig.tight_layout()
        plt.show()

    
    def fit(self):
        '''
            Main method of the algorithm. This method will coordinate each step to make the ordinary genetic algorithm work
        '''
        # Initialize aptituds list. This will store all the values of the aptitud function
        self.aptituds = [0] * self.population_size
        self.fittiests_history = []

        # Get initial population randomly and store it in the global variable
        self.population = self.generate_population()

        # Calculate the aptitud function for each cromosome
        self.calculate_aptitud_function()
        
        for i in range(self.generations):    
            # Get a new list of cromosomes with a tournament
            fathers = self.tournament()
            mothers = self.tournament()
            self.log('----fathers----', 1)
            self.print_population(fathers, 1)
            self.log('----mothers----', 1)
            self.print_population(mothers, 1)

            # from the winners of the tournament, get new cromosomes by a 'breeding' process and overwrite the actual population with the new population originated from the 'winners'
            self.population = self.breed(fathers, mothers)

            # Apply mutations
            self.mutation()

            # Calculate the aptitud function for each new cromosome
            self.calculate_aptitud_function()

            # Apply elitism if required
            if self.elitism:
                self.population = self.apply_elitism(fathers, mothers, self.population)

            # Get fittiest to make the graph
            self.fittiest = self.get_fittiest()
            self.fittiests_history.append(self.fittiest[-1])

            # self.graph_history()
            self.log(f"---------Generation {i+1}", 2)
            self.print_population(debuglevel=2)
            
            if self.graph_generations:
                self.plot_data(generation=i+1)

            
            self.log(self.fittiest, 5, f"Generation {i+1}")

        self.log(self.fittiests_history, 2)
        # self.plot_functions(generation_fittiest, i)

        return self.fittiest


    def print_population(self, population=None, debuglevel=0):
        '''
            Used for debug
        '''
        if population == None:
            population = self.population

        for i in range(len(population)):
            self.log(str(population[i]), debuglevel)