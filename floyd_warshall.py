import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

# Generate a random adjacency matrix for a graph with 500 nodes and 0.5 probability of connection
def generate_graph(num_nodes=500, connection_prob=0.5):
    graph = np.random.rand(num_nodes, num_nodes)
    graph = (graph < connection_prob).astype(float)
    np.fill_diagonal(graph, 0)  # Distance from a node to itself is 0
    graph[graph == 0] = float('inf')  # No connection means infinite distance
    np.fill_diagonal(graph, 0)
    return graph

# CUDA kernel for Floyd-Warshall algorithm
@cuda.jit
def floyd_warshall_cuda(dist, k):
    i, j = cuda.grid(2)  # Get thread indices
    n = dist.shape[0]

    # Ensure threads are within matrix bounds
    if i < n and j < n:
        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

# Host function for CUDA-based Floyd-Warshall
def floyd_warshall_cuda_host(graph, threads_per_block=(16, 16)):
    num_nodes = graph.shape[0]
    blocks_per_grid = (
        (num_nodes + threads_per_block[0] - 1) // threads_per_block[0],
        (num_nodes + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Allocate memory on the device and copy the graph to it
    dist_device = cuda.to_device(graph)

    # Launch kernel for each intermediate node 'k'
    for k in range(num_nodes):
        floyd_warshall_cuda[blocks_per_grid, threads_per_block](dist_device, k)
        cuda.synchronize()  # Ensure all threads complete before moving to the next iteration

    # Copy the result back to the host
    return dist_device.copy_to_host()

# CPU-based Floyd-Warshall for verification
def floyd_warshall(graph):
    num_nodes = len(graph)
    dist = graph.copy()
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist

# Main execution
if __name__ == "__main__":
    np.random.seed(seed=42)
    num_nodes = 500
    graph = generate_graph(num_nodes=num_nodes)

    # Non-parallel execution (CPU)
    start_time = time.time()
    dist_non_parallel = floyd_warshall(graph)
    cpu_time = time.time() - start_time
    print(f"Non-parallel execution time (CPU): {cpu_time:.2f} seconds")

    # CUDA execution with different thread configurations
    thread_configs = [(8, 8), (16, 16), (32, 32)]
    cuda_times = []
    speedups = []

    for threads_per_block in thread_configs:
        try:
            start_time = time.time()
            dist_cuda = floyd_warshall_cuda_host(graph, threads_per_block)
            cuda_time = time.time() - start_time
            cuda_times.append((threads_per_block, cuda_time))
            speedups.append(cpu_time / cuda_time)
            print(f"CUDA execution time with threads {threads_per_block}: {cuda_time:.2f} seconds")
        except cuda.CudaAPIError as e:
            print(f"Error with threads {threads_per_block}: {e}")

    # Verify correctness
    print("Verifying results...")
    if np.allclose(dist_non_parallel, dist_cuda, atol=1e-8):
        print("Verification successful: CPU and GPU results match!")
    else:
        print("Verification failed: Results do not match.")

    # Plot Speedup
    configs = [f"{x[0]}x{x[1]}" for x, _ in cuda_times]
    plt.figure(figsize=(10, 6))
    plt.plot(configs, speedups, marker='o', label="Speedup")
    plt.title("Speedup vs Thread Configuration")
    plt.xlabel("Thread Configuration (Threads Per Block)")
    plt.ylabel("Speedup (CPU Time / CUDA Time)")
    plt.grid(True)
    plt.legend()
    plt.show()