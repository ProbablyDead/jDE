# Differential Evolution implementation with jDE modification
##### This work solves the **task 07** for the *mark 9*

### Input data
Input data (functions implementations, dimensions and bounds) is in the **`src/functions.py`** file

### Output data
Output data stored in the **`output.txt`** file
Parameters for algorithm benchmark are: 
- `N`: population size = 50
- `max_iter`: maximum iterations count = 1000
- both `f` and `cr` are random values in bounds of [fl, fu) and [0, 1) respectively
- `fitness_function_satisfying_value` = 0

### Algorithm implementation
Realization of mutating and crossover stages are in Individual class, placed in **`src/individual.py`**
Whole algorithm (finding the best solution) is implemented in **`differential_evolution.py`**, under the **DifferentialEvaluation** class
