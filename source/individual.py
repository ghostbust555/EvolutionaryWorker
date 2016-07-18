import random

from layertype import LayerType
from neural import Neural


class Individual:
    def __init__(self, dna=None):
        self.score = 0
        self.norm_score = 0
        if dna:
            self.dna = dna
        else:
            r = random.randrange(2, 10)
            self.dna = [LayerType.random_layer() for _ in range(r)]
            self.dna.append(LayerType.loss_layer())

    def __run_network_(self, thorough=False):
        try:
            score = Neural(self, thorough).run_network()
        except:
            return 0

        return score

    def get_fitness(self):
        self.score = self.__run_network_()
        return self.score

    def get_fitness_thorough(self):
        self.score = self.__run_network_(True)
        return self.score

    def get_network(self):
        for x in self.dna:
            print(x[0].name, end=" ")
