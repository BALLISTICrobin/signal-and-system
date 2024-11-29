import numpy as np
import matplotlib.pyplot as plt

n = 50
samples = np.arange(n)
sampling_rate = 100
wave_velocity = 8000

# Use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):
    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz
    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40]
    amplitudes2 = [0.3, 0.2, 0.1]
    
    # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_signal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                             for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_signal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                             for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_signal_A
    noisy_signal_B = signal_A + noise_for_signal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B, shift_samples

# Compute the DFT of a signal
def dft(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Compute the IDFT of a signal
def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

# Calculate cross-correlation using DFT
def cross_correlation(signal_A, signal_B):
    dft_A = dft(signal_A)
    dft_B = dft(signal_B)
    cross_corr_freq = np.conj(dft_A) * dft_B
    cross_corr = idft(cross_corr_freq)
    return np.roll(cross_corr.real, n // 2)  # Shift zero lag to center

# Generate signals
signal_A, signal_B, shift_samples = generate_signals()

# Cross-correlation
cross_corr = cross_correlation(signal_A, signal_B)

# Detect sample lag
lag_index = np.argmax(cross_corr)
sample_lag = lag_index - n // 2  # Correct for zero-centered lags

# Distance estimation
time_lag = sample_lag / sampling_rate
distance = abs(sample_lag) * (1 / sampling_rate) * wave_velocity
print(f"Estimated Sample Lag: {sample_lag}")
print(f"Estimated Distance: {distance:.2f} meters")

# Plot Signal A and Signal B
plt.figure(figsize=(10, 5))
plt.stem(samples, signal_A, linefmt='b-', markerfmt='bo', basefmt=" ", label="Signal A")
plt.xlabel("Sample Number (n)")
plt.ylabel("Amplitude")
plt.title("Signal A")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.stem(samples, signal_B, linefmt='r-', markerfmt='ro', basefmt=" ", label="Signal B")
plt.xlabel("Sample Number (n)")
plt.ylabel("Amplitude")
plt.title("Signal B")
plt.legend()
plt.grid()
plt.show()

# Stem plot for magnitude spectrum of Signal A and Signal B
dft_A = dft(signal_A)
dft_B = dft(signal_B)
plt.figure(figsize=(10, 5))
plt.stem(np.abs(dft_A), linefmt='b-', markerfmt='bo', basefmt=" ", label="Magnitude Spectrum of Signal A")
plt.xlabel("Sample Indices")
plt.ylabel("Amplitude")
plt.title("Magnitude Spectrum of Signal A and Signal B")
plt.legend()
plt.grid()
plt.show()

# dft_A = dft(signal_A)
# dft_B = dft(signal_B)
plt.figure(figsize=(10, 5))
plt.stem(np.abs(dft_B), linefmt='r-', markerfmt='ro', basefmt=" ", label="Magnitude Spectrum of Signal B")
plt.xlabel("Sample Indices")
plt.ylabel("Amplitude")
plt.title("Magnitude Spectrum of Signal A and Signal B")
plt.legend()
plt.grid()
plt.show()

# Plot Cross-Correlation
plt.figure(figsize=(10, 5))
lags = np.arange(-n // 2, n // 2)
plt.stem(lags, cross_corr, linefmt='m-', markerfmt='mo', basefmt=" ", label="Cross-Correlation")
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation")
plt.title("Cross-Correlation")
plt.legend()
plt.grid()
plt.show()
