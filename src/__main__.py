from .benchmark import Benchmark
from .functions import benches
from .differential_evolution import DifferentialEvolution

Benchmark(DifferentialEvolution).bench(
    benches, N=50, max_iter=1000, visualize_progress=False)
