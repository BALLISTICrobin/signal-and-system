import numpy as np
import matplotlib.pyplot as plt

def givenfunction(x):
    return 2 * np.sin(14 * np.pi * x) - np.sin(2 * np.pi * x) * (4 * np.sin(2 * np.pi * x) * np.sin(14 * np.pi * x) - 1)

x_values = np.linspace(-3, 3, 1000)
y_values = givenfunction(x_values)

# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original given wave")
plt.title("Original Function (given wave)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

sampled_times = x_values
frequencies = np.linspace(-10, 10, 500)

def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    for i, f in enumerate(frequencies):
        ft_result_real[i] = np.trapz(signal * np.cos(2 * np.pi * f * sampled_times), sampled_times)
        ft_result_imag[i] = np.trapz((-1) * signal * np.sin(2 * np.pi * f * sampled_times), sampled_times)
    
    return ft_result_real, ft_result_imag

# Perform Fourier Transform
ft_real, ft_imag = fourier_transform(y_values, frequencies, sampled_times)

# Store individual sine and cosine functions
sine_components = []
cosine_components = []

for i, f in enumerate(frequencies):
    cosine_components.append(ft_real[i] * np.cos(2 * np.pi * f * sampled_times))
    sine_components.append(ft_imag[i] * np.sin(2 * np.pi * f * sampled_times))

# Combine the components to reconstruct the signal
reconstructed_signal = np.sum(cosine_components, axis=0) + np.sum(sine_components, axis=0)

# Plot the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_real**2 + ft_imag**2))
plt.title("Frequency Spectrum of given wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# Plot the reconstructed signal
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original given wave", color="blue")
plt.plot(sampled_times, reconstructed_signal, label="Reconstructed given wave", color="red", linestyle="--")
plt.title("Original vs Reconstructed Function (given wave)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
#     ft_real, ft_imag = ft_signal
#     n = len(sampled_times)
#     reconstructed_signal = np.zeros(n)
    
#     for t_idx, t in enumerate(sampled_times):
#         # Sum over all frequencies
#         real_part = ft_real * np.cos(2 * np.pi * frequencies * t)
#         imag_part = ft_imag * np.sin(2 * np.pi * frequencies * t)
#         reconstructed_signal[t_idx] = np.trapz(real_part - imag_part, frequencies)
    
#     return reconstructed_signal



# # Reconstruct the signal from the FT data
##reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# Plot the original and reconstructed functions for comparison
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, y_values, label="Original given wave", color="blue")
# plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed given wave", color="red", linestyle="--")
# plt.title("Original vs Reconstructed Function (given wave)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()