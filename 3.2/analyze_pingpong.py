import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data for intra-node communication
sizes_intra = np.array([
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
    8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
    536870912, 1073741824
])
times_intra = np.array([
    0.000001181, 0.000000642, 0.000000598, 0.000000599, 0.000000678,
    0.000000763, 0.000000968, 0.000001335, 0.000001943, 0.000002996,
    0.000003209, 0.000004768, 0.000008808, 0.000015774, 0.000026724,
    0.000047919, 0.000087073, 0.000161961, 0.000311406, 0.000624942,
    0.001255687, 0.002517168, 0.004719127, 0.009077462, 0.017646981,
    0.034974375, 0.069307755, 0.139068583
])

# Data for inter-node communication
times_inter = np.array([
     0.000002475, 0.000002507, 0.000002395, 0.000002421, 0.000003101,
     0.000003087, 0.000003275, 0.000003331, 0.000003412, 0.000003535,
     0.000003916, 0.000004586, 0.000007910, 0.000010556, 0.000014358,
     0.000024367, 0.000031720, 0.000053753, 0.000103852, 0.000193444,
     0.000379552, 0.000752024, 0.001503135, 0.003076158, 0.007413638,
     0.015109757, 0.030281336, 0.061136750
])

# Function to perform linear regression and return bandwidth and latency
def analyze_performance(sizes, times):
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    model = LinearRegression().fit(log_sizes.reshape(-1, 1), log_times)
    latency = np.exp(model.intercept_)
    bandwidth = 1 / model.coef_[0]
    return bandwidth, latency

# Analyze both intra-node and inter-node
bandwidth_intra, latency_intra = analyze_performance(sizes_intra, times_intra)
bandwidth_inter, latency_inter = analyze_performance(sizes_intra, times_inter)

# Plotting results
plt.figure(figsize=(10, 5))
plt.loglog(sizes_intra, times_intra, 'o-', label=f'Intra-node (Bandwidth={bandwidth_intra:.2f}, Latency={latency_intra:.2e})')
plt.loglog(sizes_intra, times_inter, 'o-', label=f'Inter-node (Bandwidth={bandwidth_inter:.2f}, Latency={latency_inter:.2e})')
plt.xlabel('Message Size (bytes)')
plt.ylabel('Ping-Pong Time (seconds)')
plt.title('MPI Ping-Pong Times Comparison')
plt.legend()
plt.grid(True)
plt.show()
