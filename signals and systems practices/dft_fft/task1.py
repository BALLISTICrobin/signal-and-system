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

# # Discrete sample indices
# dt = 1 / sampling_rate  # Sampling interval in seconds
# time = samples * dt  # Time points corresponding to each sample

# original_signal = np.sin(2 * np.pi * 5 * time)
# #frequency spectrum of the original signal
# dft_original_signal = dft(original_signal)

# # Plot frequency spectrum of the original signal
# plt.stem(np.abs(dft_original_signal), linefmt='b-', markerfmt='bo', basefmt=" ", label="Frequency Spectrum of Original Signal")
# plt.xlabel("Sample Indices")
# plt.ylabel("Amplitude")
# plt.title("Frequency Spectrum of Original Signal")
# plt.legend()
# plt.grid()
# plt.show()

# # Generate signals
signal_A, signal_B, shift_samples = generate_signals()

# # Cross-correlation
cross_corr = cross_correlation(signal_A, signal_B)

# # Detect sample lag
lag_index = np.argmax(cross_corr)
sample_lag = lag_index - n // 2  # Correct for zero-centered lags

# # Distance estimation
time_lag = sample_lag / sampling_rate
distance = abs(sample_lag) * (1 / sampling_rate) * wave_velocity
print(f"Estimated Sample Lag: {sample_lag}")
print(f"Estimated Distance: {distance:.2f} meters")

# # Set up the figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns

# # Plot Signal A
axs[0, 0].stem(samples, signal_A, linefmt='b-', markerfmt='bo', basefmt=" ", label="Signal A")
axs[0, 0].set_xlabel("Sample Number (n)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].set_title("Signal A")
axs[0, 0].legend()
axs[0, 0].grid()

# # Plot Signal B
axs[0, 1].stem(samples, signal_B, linefmt='r-', markerfmt='ro', basefmt=" ", label="Signal B")
axs[0, 1].set_xlabel("Sample Number (n)")
axs[0, 1].set_ylabel("Amplitude")
axs[0, 1].set_title("Signal B")
axs[0, 1].legend()
axs[0, 1].grid()

# # Plot Magnitude Spectrum of Signal A
dft_A = dft(signal_A)
dft_B = dft(signal_B)
axs[1, 0].stem(np.abs(dft_A), linefmt='b-', markerfmt='bo', basefmt=" ", label="Magnitude Spectrum of Signal A")
axs[1, 0].set_xlabel("Sample Indices")
axs[1, 0].set_ylabel("Amplitude")
axs[1, 0].set_title("Magnitude Spectrum of Signal A")
axs[1, 0].legend()
axs[1, 0].grid()

# # Plot Magnitude Spectrum of Signal B
axs[1, 1].stem(np.abs(dft_B), linefmt='r-', markerfmt='ro', basefmt=" ", label="Magnitude Spectrum of Signal B")
axs[1, 1].set_xlabel("Sample Indices")
axs[1, 1].set_ylabel("Amplitude")
axs[1, 1].set_title("Magnitude Spectrum of Signal B")
axs[1, 1].legend()
axs[1, 1].grid()

# # Plot Cross-Correlation
lags = np.arange(-n // 2, n // 2)
axs[2, 0].stem(lags, cross_corr, linefmt='m-', markerfmt='mo', basefmt=" ", label="Cross-Correlation")
axs[2, 0].set_xlabel("Lag (samples)")
axs[2, 0].set_ylabel("Correlation")
axs[2, 0].set_title("Cross-Correlation")
axs[2, 0].legend()
axs[2, 0].grid()

# Hide the empty subplot (bottom right)
fig.delaxes(axs[2, 1])

# Adjust layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()


# Function to denoise a signal using a low-pass filter
def denoise_signal_with_custom_filter(signal, zero_indices):
    # Compute the DFT of the signal
    signal_dft = dft(signal)
    
    # Number of samples
    n = len(signal)
    
    # Create a mask to zero out specific indices
    filter_mask = np.ones(n, dtype=bool)  # Start with all components included
    filter_mask[zero_indices] = False     # Exclude the specified indices
    
    # Apply the filter mask
    filtered_dft = signal_dft * filter_mask  # Zero out specified indices
    
    # Compute the IDFT to get the filtered signal
    filtered_signal = idft(filtered_dft).real  # Take the real part of the result
    
    return filtered_signal

zero_indices = range(10, 41)

# Apply the custom filter to both signals
filtered_signal_A = denoise_signal_with_custom_filter(signal_A, zero_indices)
filtered_signal_B = denoise_signal_with_custom_filter(signal_B, zero_indices)

#plot the signal A and signal B after filtering
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows, 2 columns

# Plot Signal A after filtering
axs[0, 0].stem(samples, filtered_signal_A, linefmt='b-', markerfmt='bo', basefmt=" ", label="Signal A after Filtering")
axs[0, 0].set_xlabel("Sample Number (n)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].set_title("Signal A after Filtering")
axs[0, 0].legend()
axs[0, 0].grid()

# Plot Signal B after filtering
axs[0, 1].stem(samples, filtered_signal_B, linefmt='r-', markerfmt='ro', basefmt=" ", label="Signal B after Filtering")
axs[0, 1].set_xlabel("Sample Number (n)")
axs[0, 1].set_ylabel("Amplitude")
axs[0, 1].set_title("Signal B after Filtering")
axs[0, 1].legend()
axs[0, 1].grid()

# Adjust layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()


# # Cross-correlation
cross_corr = cross_correlation(signal_A, signal_B)

# # Detect sample lag
lag_index = np.argmax(cross_corr)
sample_lag = lag_index - n // 2  # Correct for zero-centered lags

# # Distance estimation
time_lag = sample_lag / sampling_rate
distance = abs(sample_lag) * (1 / sampling_rate) * wave_velocity
print(f"Estimated Sample Lag after filtering: {sample_lag}")
print(f"Estimated Distance after filtering: {distance:.2f} meters")
