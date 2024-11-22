import numpy as np
import matplotlib.pyplot as plt

# Create the parabolic function
def parabolic_function(x):
    return np.where((x >= -2) & (x <= 2), x**2, 0)

# Create the triangular function
def triangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1 - np.abs(x / 2), 0)

# Create the sawtooth function
def sawtooth_function(x):
    return np.where((x >= -2) & (x <= 2), (x + 2) / 4, 0)

# Create the rectangular function
def rectangular_function(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)

# Define the interval and function and generate appropriate x values and y values
# Define the interval and generate x values
x_values = np.linspace(-10, 10, 1000)

# Choose one function at a time for demonstration
y_values = triangular_function(x_values)  # Change to triangular_function, sawtooth_function, or rectangular_function

# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# Define the sampled times and frequencies
sampled_times = x_values
frequencies = np.linspace(-1, 1, 1000)

# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    for i, f in enumerate(frequencies):
        # Compute real and imaginary parts using trapezoidal integration
        ft_result_real[i] = np.trapz(signal * np.cos(2 * np.pi * f * sampled_times), sampled_times)
        ft_result_imag[i] = np.trapz((-1)*signal * np.sin(2 * np.pi * f * sampled_times), sampled_times)
    
    return ft_result_real, ft_result_imag

# Apply FT to the sampled data
ft_data = fourier_transform(y_values, frequencies, sampled_times)
#  plot the FT data
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
plt.title("Frequency Spectrum of y = x^2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()


# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    ft_real, ft_imag = ft_signal
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    
    for t_idx, t in enumerate(sampled_times):
        # Sum over all frequencies
        real_part = ft_real * np.cos(2 * np.pi * frequencies * t)
        imag_part = ft_imag * np.sin(2 * np.pi * frequencies * t)
        reconstructed_signal[t_idx] = np.trapz(real_part - imag_part, frequencies)
    
    return reconstructed_signal



# Reconstruct the signal from the FT data
reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# Plot the original and reconstructed functions for comparison
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
plt.title("Original vs Reconstructed Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
