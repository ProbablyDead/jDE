from typing import Callable
from .individual import Individual
from tqdm import tqdm


class DifferentialEvolution:
    def __init__(self,
                 fitness_function: Callable[[list[float]], float],
                 N: int = 50,
                 dim: int = 10,
                 max_iter: int = 1000,
                 fitness_function_satisfying_value: float = 0,
                 fl: float = 0.1,
                 fu: float = 1.,
                 t1: float = 0.1,
                 t2: float = 0.1,
                 bounds: list[tuple[float]] = None,
                 visialize_progress: bool = True
                 ):
        self.__fitness_function = fitness_function
        self.__N = N
        self.__dim = dim
        self.__max_iter = max_iter
        self.__fitness_function_satisfying_value = \
            fitness_function_satisfying_value
        self.__fl = fl
        self.__fu = fu
        self.__t1 = t1
        self.__t2 = t2
        self.__bounds = bounds
        self.__visualize_progress = visialize_progress

    def __create_random_population(self):
        return [Individual(self.__fitness_function,
                           dim=self.__dim,
                           bounds=self.__bounds,
                           fl=self.__fl,
                           fu=self.__fu,
                           t1=self.__t1,
                           t2=self.__t2
                           ) for _ in range(self.__N)]

    def optimize(self):
        population = self.__create_random_population()
        best_solution = population[0]

        rng = range(self.__max_iter)
        if self.__visualize_progress:
            rng = tqdm(rng)

        for iter in rng:
            if best_solution.get_fitness_score() < \
                    self.__fitness_function_satisfying_value:
                break
            for i in range(self.__N):
                v_trail = population[i].get_trail_vector()
                population[i] = min(population[i],
                                    v_trail,
                                    key=lambda x: x.get_fitness_score())
                best_solution = min(best_solution,
                                    population[i],
                                    key=lambda x: x.get_fitness_score())

        return best_solution.get_fitness_score(), \
            best_solution.get_coordinates()
