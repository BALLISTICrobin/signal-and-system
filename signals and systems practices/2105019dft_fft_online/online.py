
import numpy as np
import math
# Example usage
x = 123
y = 456

x_digits = [int(d) for d in str(x)]
y_digits = [int(d) for d in str(y)]

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

def circular_convolution(signal_A, signal_B):
    
    n=len(signal_A)+len(signal_B)-1
    if(len(signal_A)&(len(signal_A)-1)!=0):
        no_of_zeros = 2**math.ceil(math.log2(n))-len(signal_A)
        for i in range (0,no_of_zeros,1):
            signal_A.append(0)
    
    if(len(signal_B)&(len(signal_B)-1)!=0):
        no_of_zeros = 2**math.ceil(math.log2(n))-len(signal_B)
        for i in range (0,no_of_zeros,1):
            signal_B.append(0)
    print(signal_A,signal_B)        
    # print(len(signal_A),len(signal_B))  
    fft_A = FFT(signal_A)
    fft_B = FFT(signal_B)
    cicular_convolve_freq =  fft_A*fft_B
    ifft_signal = IFFT(cicular_convolve_freq)
    return np.array(np.rint(ifft_signal.real), dtype=int)[:n]


# len = (len(x_digits)-1)*(len(y_digits)-1)+1
convo = circular_convolution(x_digits,y_digits)
print(convo)
length = convo.size
result = np.zeros(length+1,dtype=int)

carry = 0
resultRIndex = result.size-1

for i in range(convo.size-1,-1,-1):
    value= convo[i]+carry
    result[resultRIndex] = value%10
    carry = value//10
    resultRIndex-=1
    
result[resultRIndex] = carry
for i in result:
    print(i,end="")




