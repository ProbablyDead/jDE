import numpy as np


class Individual:
    def __init__(self,
                 fitness_function,
                 dim: int = 10,
                 bounds: list[tuple[float]] = None,
                 f: float = None,
                 fl: float = 0.1,
                 fu: float = 1.,
                 t1: float = 0.1,
                 t2: float = 0.1,
                 coordinates: list[float] = None
                 ):
        self.__fitness_function = fitness_function
        self.__dim = dim
        self.__bounds = bounds if bounds else [(0, 1)]*dim
        self.__lower_bounds = [b[0] for b in bounds]
        self.__upper_bounds = [b[1] for b in bounds]

        self.__coordinates = coordinates if coordinates is not None \
            else self.__get_random_solution()
        self.__fl = fl
        self.__fu = fu
        self.__f = f if f is not None else np.random.uniform(
            low=self.__fl, high=self.__fu)
        self.__t1 = t1
        self.__t2 = t2
        self.__cr = np.random.rand()

    def get_fitness_score(self):
        return self.__fitness_function(self.__coordinates)

    def get_coordinates(self):
        return np.array(self.__coordinates)

    def __clip(self, coordinates):
        return np.clip(
            coordinates,
            a_min=self.__lower_bounds,
            a_max=self.__upper_bounds,
        )

    def __get_random_solution(self):
        return self.__clip(np.random.uniform(
            low=self.__lower_bounds,
            high=self.__upper_bounds,
            size=(self.__dim,)
        ))

    def __get_three_random_solutions(self):
        vs = []

        for _ in range(3):
            v_gen = self.__get_random_solution()
            while np.array_equal(v_gen, self.__coordinates):
                v_gen = self.__get_random_solution()
            vs.append(v_gen)

        return vs

    def __get_mutated_vector(self):
        v1, v2, v3 = self.__get_three_random_solutions()
        return self.__clip(v3 - self.__f*(v1 - v2))

    def __update_f_and_cr(self):
        self.__f = np.random.uniform(low=self.__fl, high=self.__fu) if \
            np.random.rand() < self.__t1 else self.__f

        self.__cr = np.random.rand() if \
            np.random.rand() < self.__t2 else self.__cr

    def get_trail_vector(self):
        vy = self.__get_mutated_vector()
        v_trail = Individual(
            self.__fitness_function,
            dim=self.__dim,
            bounds=self.__bounds,
            f=self.__f,
            fl=self.__fl,
            fu=self.__fu,
            t1=self.__t1,
            t2=self.__t2,
            coordinates=[]
        )

        for x, y in zip(self.__coordinates, vy):
            v_trail.__coordinates.append(
                y if np.random.rand() < self.__cr else x)

        v_trail.__update_f_and_cr()

        return v_trail
