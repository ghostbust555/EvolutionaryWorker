from individual import Individual
from layertype import LayerType, Loss

x = Individual([(LayerType.elu, 667), (LayerType.loss, Loss.binary_crossentropy)]).get_fitness_thorough()

print(x)