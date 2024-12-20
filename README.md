# Floyd-Warshall Algorithm Parallelization with CUDA

This project demonstrates the parallelization of the **Floyd-Warshall** algorithm using **CUDA** to accelerate the computation of shortest paths in a graph, resulting in significant performance improvements over the non-parallel (CPU) implementation.

## 1. Algorithm and Parallelization Method

### Algorithm
The **Floyd-Warshall** algorithm is used to find the shortest paths between all pairs of nodes in a graph. It iterates over all possible intermediate nodes and updates the shortest distance between all pairs of nodes using the following relation:` 

dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

markdown

Copy code

 `Where:`
- `dist[i, j]` represents the shortest distance between nodes `i` and `j`.
- `k` is an intermediate node, and the algorithm iterates over all nodes as intermediates.

### Parallelization Method
To accelerate the Floyd-Warshall algorithm, we parallelize the core computation of updating the distance matrix using **CUDA**. The approach involves the following steps:

- **CUDA Kernel**: The kernel is responsible for performing distance updates in parallel on the GPU. Each thread calculates the shortest path for a pair of nodes `(i, j)` at each intermediate node `k`.
- **Grid and Block Configuration**: The kernel is organized into a 2D grid of threads. Each thread corresponds to a unique `(i, j)` pair and is responsible for updating the distance in the distance matrix.
- **Synchronization**: After the update for a specific intermediate node `k`, synchronization ensures that all threads complete their computations before proceeding to the next iteration.

This parallelization significantly reduces the runtime of the algorithm, especially for larger graphs.

## 2. Instructions to Reproduce Results

### Prerequisites
Make sure you have the following tools and libraries installed:
- **Python 3.x**
- **Numba** for CUDA programming
- **NumPy** for matrix operations
- **CUDA Toolkit** installed with a compatible GPU

### Installation
1.  Clone the repository.
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
    
2.  Install the required dependencies as mentioned below.
```bash
pip install numba numpy
```
### Running the Script

[](https://github.com/captainamiedi/parallel-programming-project#running-the-script)

1.  Execute the script directly:
    ```bash
    python floyd_warshall.py
    ```
    
2.  The script generates a random graph (500 nodes by default, 0.5 probability of connection), runs both the non-parallel and parallel versions of the algorithm, verifies the correctness of the results, and plots the speedup achieved by parallel execution.

## 3. Parallelized Part of the Algorithm

The parallelization focuses on the **distance matrix update** at each step of the Floyd-Warshall algorithm. The specific tasks parallelized are:

-   **Matrix Updates**: For each intermediate node `k`, the distance between all pairs `(i, j)` is updated in parallel.
-   **Thread-Level Parallelism**: Each thread is assigned a unique pair `(i, j)` to compute the shortest path for the current node `k`.

The parallel computation of all pairs of nodes allows the algorithm to process large graphs in much less time compared to the sequential version.

## 4. Speedup Calculation and Figure

### Speedup Calculation

The speedup is calculated by comparing the execution time of the non-parallel CPU version with the parallel CUDA version. The formula for speedup is:

makefile

`Speedup = Time_non_parallel / Time_parallel` 

Where:

-   `Time_non_parallel` is the execution time for the non-parallel CPU version.
-   `Time_parallel` is the execution time for the parallel CUDA version.

### Speedup vs Number of Threads

The speedup increases as the number of threads increases, but at a certain point, the speedup may begin to plateau due to the overhead of managing too many threads or hardware limitations.

#### Speedup Plot

A plot showing the relationship between the number of threads and speedup can be generated. The X-axis represents the number of threads (e.g., `(8, 8)`, `(16, 16)`, `(32, 32)`), and the Y-axis represents the speedup (the ratio of CPU time to GPU time).


### Results

The following table illustrates the execution times for the non-parallel (CPU) and parallelized (CUDA) versions of the Floyd-Warshall algorithm using different thread configurations. The **Speedup** is calculated as the ratio of the CPU execution time to the CUDA execution time.

| Thread Configuration | CPU Time (s) | CUDA Time (s) | Speedup |
|----------------------|--------------|---------------|---------|
| Non-parallel (CPU)   | 97.54        | N/A           | N/A     |
| (8, 8)               | N/A          | 0.13          | 751.08  |
| (16, 16)             | N/A          | 0.04          | 2438.5  |
| (32, 32)             | N/A          | 0.06          | 1625.67 |

- **CPU Time (s)**: The time taken for the non-parallel execution of the algorithm on the CPU.
- **CUDA Time (s)**: The time taken for the parallel execution of the algorithm on the GPU using CUDA.
- **Speedup**: The speedup achieved by using the parallel version (calculated as `CPU Time / CUDA Time`).

The results show a significant speedup with increasing thread configurations. The **(16, 16)** configuration achieved the highest speedup of **2438.5x** compared to the CPU execution.


This table shows the speedup achieved with different thread configurations.

## Conclusion

### **GPU Acceleration is Highly Effective**

-   **Observation:** CUDA execution is significantly faster than the CPU-based implementation.
    -   CPU time: ~97.54 seconds
    -   Best CUDA time: ~0.04 seconds
-   **Conclusion:** The GPU provides massive speedup, with a **speedup factor of approximately 97.54 / 0.04 = 2438x** for the best thread configuration (`16x16`). This demonstrates the power of parallelism and the advantage of leveraging CUDA for computationally intensive tasks.

### **Thread Configuration Affects Performance**

-   **Observation:**
    -   `(8, 8)` threads: 0.13 seconds
    -   `(16, 16)` threads: 0.04 seconds
    -   `(32, 32)` threads: 0.06 seconds
-   **Conclusion:** The choice of thread configuration impacts performance. For this problem size:
    -   `(16, 16)` provides the optimal balance of workload distribution and resource utilization.
    -   `(8, 8)` underutilizes the GPU, leading to longer execution times.
    -   `(32, 32)` may result in diminishing returns due to potential oversubscription of GPU resources or memory constraints.
    

Below is the graph showing the speedup achieved by different thread configurations.

![Speedup Graph](download.png)