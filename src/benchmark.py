class Benchmark:
    def __init__(self, optimizer):
        self.__optimizer = optimizer

    def bench(self, benches, N=50, max_iter=1000, visualize_progress=True):
        for bench in benches:
            res, res_a = self.__optimizer(bench.f,
                                          N=N,
                                          max_iter=max_iter,
                                          dim=bench.dim,
                                          bounds=bench.bounds,
                                          visialize_progress=visualize_progress
                                          ).optimize()
            print(f"{bench.name}\t{res}\n{res_a}\n")
