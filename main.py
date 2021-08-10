from BenchmarkModel import BenchmarkModel
import utils
import BenchmarkModel

benchmark_models = utils.raed_benchmarks()

for model in benchmark_models:
    model.get_metrics()