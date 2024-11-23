import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import trapz

def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    for i, f in enumerate(frequencies):
        # Compute real and imaginary parts using trapezoidal integration
        ft_result_real[i] = np.trapz(signal * np.cos(2 * np.pi * f * sampled_times), sampled_times)
        ft_result_imag[i] = np.trapz((-1)*signal * np.sin(2 * np.pi * f * sampled_times), sampled_times)
    
    return ft_result_real, ft_result_imag 

def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    ft_real, ft_imag = ft_signal
    reconstructed_signal = np.zeros(len(sampled_times))
    
    for t_idx, t in enumerate(sampled_times):
        real_part = ft_real * np.cos(2 * np.pi * frequencies * t)
        imag_part = ft_imag * np.sin(2 * np.pi * frequencies * t)
        reconstructed_signal[t_idx] = np.trapz(real_part - imag_part, frequencies)
    
    return reconstructed_signal  

# Load and preprocess the image
image = plt.imread('signals and systems practices/onlineFT/noisy_image.png')  # Replace with your image file path
# show the image
# plt.figure()
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.show()

if image.ndim == 3:
    image = np.mean(image, axis=2)  # Convert to grayscale

image = image / 255.0  # Normalize to range [0, 1]
print (image.shape)

sample_rate = 1000 
interval_step = 1  # Adjust for sampling every 'interval_step' image points
image_sampled = image[::interval_step]
max_time = len(image_sampled) / (sample_rate / interval_step)
sampled_times = np.linspace(0, max_time, num=len(image_sampled))

max_freq = sample_rate / (2 * interval_step)
num_freqs = len(image_sampled)
frequencies = np.linspace(0, max_freq, num=num_freqs)

ft_data = []
reconstructed_data = []
for i,signal in enumerate(image):
    ft_data.append(fourier_transform(signal, frequencies, sampled_times))
    # # Step 2.1: Visualize the frequency spectrum
    # plt.figure(figsize=(12, 6))
    # plt.plot(frequencies, np.sqrt(ft_data[i][0]**2 + ft_data[i][1]**2))
    # plt.title("Frequency Spectrum of the Audio Signal (Custom FT with Trapezoidal Integration)")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude")
    # plt.show()
    filtered_ft_data = np.zeros((2, num_freqs))
    filtered_ft_data[0] = ft_data[i][0].copy()
    filtered_ft_data[1] = ft_data[i][1].copy()
    magnitudes = np.sqrt(filtered_ft_data[0]**2 + filtered_ft_data[1]**2)
    mask = ((frequencies >= 20) & (frequencies <= 55))
    filtered_ft_data[0] = np.where(mask, 0 ,filtered_ft_data[0])
    filtered_ft_data[1] = np.where(mask, 0, filtered_ft_data[1])
    reconstructed_data.append(inverse_fourier_transform(filtered_ft_data, frequencies, sampled_times))
    
# plt.figure(figsize=(12, 4))
# plt.plot(reconstructed_data)
# plt.title("Reconstructed (Denoised)  image (Time Domain)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()




plt.imsave('signals and systems practices/onlineFT/denoised_image.png', reconstructed_data, cmap='gray')


plt.figure()
plt.title('Denoised Image')
plt.imshow(reconstructed_data, cmap='gray')
plt.show()
