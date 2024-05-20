from memory_profiler import profile
from modules.pipeline import *
import time

graph = read_graph("/Users/djiemboutienctheuvictornico/Documents/MyFolders/ACADEMIC/M2_thesis/scripting/outputs1/AFB/withClass/0.1/qualitative/mlna_1/graph_storage/Motif_mln_1436_9_2024_04_24_21_01_19.gml.gz")

@profile
def gpuP():
    for i in range(12):
        gpu  = dict(sorted(pagerank(graph).items(), key=lambda x: abs(x[1]), reverse=False)[-5:])
@profile
def cpuP():
    for i in range(12):
        cpu  = dict(sorted(nx.pagerank(graph).items(), key=lambda x: abs(x[1]), reverse=False)[-5:])

start_time = time.time()
gpuP()
end_time = time.time()

execution_time = end_time - start_time
print(f"GPU Execution time: {execution_time:.2f} seconds")

start_time = time.time()
cpuP()
end_time = time.time()

execution_time = end_time - start_time
print(f"CPU Execution time: {execution_time:.2f} seconds")
