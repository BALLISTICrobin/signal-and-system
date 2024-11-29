import numpy as np
import matplotlib.pyplot as plt
import time

# Provided FFT function
def FFT(x):
    N = len(x)
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        X = np.concatenate([
            X_even + factor[:N // 2] * X_odd,
            X_even + factor[N // 2:] * X_odd
        ])
        return X

# Provided DFT function
def dft(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Implement IDFT
def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

# Implement IFFT
def IFFT(X):
    N = len(X)
    if N == 1:
        return X
    else:
        X_even = IFFT(X[::2])
        X_odd = IFFT(X[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)  # Inverse factor
        x = np.concatenate([
            X_even + factor[:N // 2] * X_odd,
            X_even + factor[N // 2:] * X_odd
        ])
        return x / 2

# Function to compute average runtime
def measure_runtime(func, signal, repetitions=10):
    runtimes = []
    for _ in range(repetitions):
        start_time = time.time()
        func(signal)
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return np.mean(runtimes)

# Function to generate a signal (combination of sinusoids + noise)
def generate_signal(n, frequencies=[5, 15, 30], amplitudes=[1.0, 0.5, 0.3]):
    if len(frequencies) != len(amplitudes):
        raise ValueError("Frequencies and amplitudes must have the same length.")
    t = np.arange(n) / n  # Normalized time vector
    signal = np.zeros(n, dtype=complex)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)  # Real component
    signal += 0.1 * (np.random.rand(n) + 1j * np.random.rand(n))  # Add small noise
    return signal

# Main analysis
signal_lengths = [2**k for k in range(2, 10)]  # Powers of 2 from 4 to 1024
dft_runtimes = []
idft_runtimes = []
fft_runtimes = []
ifft_runtimes = []

for n in signal_lengths:
    # Generate signal of length n
    signal = generate_signal(n)
    
    # Measure runtimes
    dft_runtimes.append(measure_runtime(dft, signal))
    idft_runtimes.append(measure_runtime(idft, dft(signal)))
    fft_runtimes.append(measure_runtime(FFT, signal))
    ifft_runtimes.append(measure_runtime(IFFT, FFT(signal)))

# Convert runtimes to milliseconds (ms)
dft_runtimes_ms = [t * 1000 for t in dft_runtimes]
idft_runtimes_ms = [t * 1000 for t in idft_runtimes]
fft_runtimes_ms = [t * 1000 for t in fft_runtimes]
ifft_runtimes_ms = [t * 1000 for t in ifft_runtimes]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
fig.suptitle("Runtime Comparison: DFT/IDFT vs FFT/IFFT", fontsize=16)

# Plot DFT and FFT runtimes on the first subplot
axs[0].plot(signal_lengths, dft_runtimes_ms, marker='o', color='r', label="DFT")
axs[0].plot(signal_lengths, fft_runtimes_ms, marker='o', color='g', label="FFT")
axs[0].set_title("DFT and FFT Runtime")
axs[0].set_xlabel("Signal Length (n)")
axs[0].set_ylabel("Runtime (ms)")
axs[0].grid()
axs[0].legend()

# Plot IDFT and IFFT runtimes on the second subplot
axs[1].plot(signal_lengths, idft_runtimes_ms, marker='o', color='b', label="IDFT")
axs[1].plot(signal_lengths, ifft_runtimes_ms, marker='o', color='m', label="IFFT")
axs[1].set_title("IDFT and IFFT Runtime")
axs[1].set_xlabel("Signal Length (n)")
axs[1].set_ylabel("Runtime (ms)")
axs[1].grid()
axs[1].legend()

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


