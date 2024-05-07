import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

def perform_regression_and_plot(sizes, times, title):

    slope, intercept, r_value, p_value, std_err = linregress(sizes, times)
    

    bandwidth = 1 / slope
    latency = intercept


    print(f"{title} - Bandwidth: {bandwidth:.2f} bytes/sec, Latency: {latency:.6f} sec")


    plt.figure()
    plt.loglog(sizes, times, 'o', label='Measured Times')
    

    best_fit_line = (slope * sizes) + intercept
    plt.loglog(sizes, best_fit_line, 'r', label=f'Best Fit Line\nBandwidth={bandwidth:.2f}, Latency={latency:.6f}')
    

    plt.xlabel('Message Size (bytes)')
    plt.ylabel('Ping-Pong Time (seconds)')
    plt.title(f'Ping-Pong Times vs. Message Size - {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():

    sizes_intra = np.array([
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824], dtype=np.float64)

    times_intra = np.array([
        0.000001181, 0.000000642, 0.000000598, 0.000000599, 0.000000678,
        0.000000763, 0.000000968, 0.000001335, 0.000001943, 0.000002996,
        0.000003209, 0.000004768, 0.000008808, 0.000015774, 0.000026724,
        0.000047919, 0.000087073, 0.000161961, 0.000311406, 0.000624942,
        0.001255687, 0.002517168, 0.004719127, 0.009077462, 0.017646981,
        0.034974375, 0.069307755, 0.139068583], dtype=np.float64)

    sizes_inter = np.array([
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
        536870912, 1073741824], dtype=np.float64)

    times_inter = np.array([
        0.000002475, 0.000002507, 0.000002395, 0.000002421, 0.000003101,
        0.000003087, 0.000003275, 0.000003331, 0.000003412, 0.000003535,
        0.000003916, 0.000004586, 0.000007910, 0.000010556, 0.000014358,
        0.000024367, 0.000031720, 0.000053753, 0.000103852, 0.000193444,
        0.000379552, 0.000752024, 0.001503135, 0.003076158, 0.007413638,
        0.015109757, 0.030281336, 0.061136750], dtype=np.float64)

    # Perform analysis and plotting for both intra and inter-node communications
    perform_regression_and_plot(sizes_intra, times_intra, "Intra-Node Communication")
    perform_regression_and_plot(sizes_inter, times_inter, "Inter-Node Communication")

if __name__ == "__main__":
    main()
