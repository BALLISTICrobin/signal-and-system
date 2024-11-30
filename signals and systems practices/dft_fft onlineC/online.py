import numpy as np
import matplotlib.pyplot as plt

#implement the necessary functions here

def fft(x):

    N = len(x)
    
    if N & (N - 1) != 0:
        next_power_of_2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_power_of_2 - N))
        N = next_power_of_2

    if N == 1:
        return x
    
    even = fft(x[::2])
    odd = fft(x[1::2])
    
    result = np.zeros(N, dtype=complex)
    
    for k in range(N // 2):
        twiddle_factor = np.exp(-2j * np.pi * k / N)
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return result

def ifft(x):

    N = len(x)

    if N & (N - 1) != 0:
        next_power_of_2 = 1 << (N - 1).bit_length() 
        x = np.pad(x, (0, next_power_of_2 - N))
        N = next_power_of_2

    if N == 1:
        return x

    even = ifft(x[::2])
    odd = ifft(x[1::2])

    result = np.zeros(N, dtype=complex)

    for k in range(N // 2):
        twiddle_factor = np.exp(2j * np.pi * k / N) 
        result[k] = even[k] + twiddle_factor * odd[k]
        result[k + N // 2] = even[k] - twiddle_factor * odd[k]

    return result / 2

def fft2d(matrix):
    fft_rows = np.array([fft(row) for row in matrix])
    fft_cols = np.array([fft(col) for col in fft_rows.T]).T
    return fft_cols

def ifft2d(matrix):
    ifft_rows = np.array([ifft(row) for row in matrix])
    ifft_cols = np.array([ifft(col) for col in ifft_rows.T]).T
    return ifft_cols

def cross_corr_2d(image,shifted_image):
    
    # F = image.copy()
    # G = shifted_image.copy()

    # for i in range(F.shape[0]):
    #     F[i] = fft(F[i])
    # for i in range(G.shape[0]):
    #     G[i] = fft(G[i])

    # for i in range(G.shape[0]):
    #     G[i] = np.conj(G[i])

    # cross_corr = np.zeros((F.shape[0],F.shape[1]),dtype=complex)

    # for i in range(F.shape[0]):
    #     for j in range(G.shape[1]):
    #         cross_corr[i][j] = F[i][j] * G[i][j]
    
    # return np.real(cross_corr)

    F = fft2d(image)
    G = fft2d(shifted_image)
    cross_corr = F * np.conj(G)
    return np.real(ifft2d(cross_corr))

def find_peak(cross_corr):
    mxi=0
    mxj=0
    for i in range(cross_corr.shape[0]):
        for j in range(cross_corr.shape[1]):
            if cross_corr[i][j] > cross_corr[mxi][mxj]:
                mxi,mxj = i,j
    
    return mxi,mxj

def reverse_shift(image,horizontal,vertical):
    revesed_image = image.copy()
    # vertical = 0
    print(horizontal,vertical)
    
    revesed_image = np.roll(revesed_image,horizontal,axis=0)
    revesed_image = np.roll(revesed_image,vertical,axis=1)

    return revesed_image
    

image = plt.imread("signals and systems practices/dft_fft onlineC/image.png")
shifted_image = plt.imread("signals and systems practices/dft_fft onlineC/shifted_image.png")
print(image.shape)
plt.figure(figsize=(12, 8))

cross_corr = cross_corr_2d(image,shifted_image)
mxi,mxj = find_peak(cross_corr)
print(mxi,mxj)
rev = reverse_shift(image=shifted_image,horizontal=mxi,vertical=mxj)

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Shifted Image
plt.subplot(2, 3, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title(f"Shifted Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(rev, cmap='gray')
plt.title("Reversed Shifted Image")
plt.axis('off')


# Reversed Shifted Image
#plt.subplot(2, 3, 3)
#plt.imshow(reversed_shifted_image, cmap='gray')
#plt.title("Reversed Shifted Image")
#plt.axis('off')

plt.tight_layout()
plt.show()
