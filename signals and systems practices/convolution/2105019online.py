import numpy as np
import matplotlib.pyplot as plt
class discreteSignal:
    
    def __init__(self,INF):
        self.INF = INF
        self.values = np.zeros(2*INF+1)
        self.mid = INF
        self.time = np.arange(-INF,INF+1,1)
        
    def setValueAtTime(self, time, value):
        index = self.mid + time
        if(-self.INF <= time <= self.INF):
            self.values[index] = value
            
    def shiftSignal(self, shift):
        newDiscreteSignal = discreteSignal(self.INF)
        newDiscreteSignal.values = np.roll(self.values, shift)
    
        if shift > 0:
            newDiscreteSignal.values[:shift] = 0
        elif shift < 0:
            
            newDiscreteSignal.values[len(newDiscreteSignal.values) + shift:] = 0
    
        return newDiscreteSignal
    
    
    def add(self,other):
        if(self.INF==other.INF):
            newDiscreteSignal = discreteSignal(self.INF)
            newDiscreteSignal.values = self.values+other.values
            return newDiscreteSignal
    
    def multiply(self, other):
        if(self.INF==other.INF):
            newDiscreteSignal = discreteSignal(self.INF)
            newDiscreteSignal.values = self.values*other.values
            return newDiscreteSignal
        
    def multiplyconstfactor(self, scaler): 
        newDiscreteSignal = discreteSignal(self.INF)
        newDiscreteSignal.values = self.values*scaler
        return newDiscreteSignal
    
    def plot(self,x_values, title, y_ticks=True,ax=None):
        tick_values = np.arange(min(self.values), max(self.values) + 1, 1)
        if(ax!=None):
            ax.stem(x_values,self.values)
            ax.set_title(title)
            ax.grid(axis='both')
            ax.set_yticks(tick_values)
        else:
            plt.stem(x_values,self.values)
            plt.title(title)
            plt.grid(axis='both')
            plt.yticks(tick_values)

class DiscreteLTI:
    def __init__(self,impulseResponse:discreteSignal):
        self.impulseResponse = impulseResponse
        
    def linear_combination_of_impulses(self, inputSignal:discreteSignal):
        impulses = []
        coefficients = []
        for index,value in enumerate(inputSignal.values):
            impulse = discreteSignal(inputSignal.INF)
            impulse.setValueAtTime(index-inputSignal.INF,1)
            impulses.append(impulse)
            coefficients.append(value)
            
        return impulses, coefficients
    
    def output(self, input_signal: discreteSignal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        output_signal = discreteSignal(input_signal.INF)
        impulse_responses = []
    
        for impulse, coeff in zip(impulses, coefficients):
            nonzero_indices = np.nonzero(impulse.values)[0]
        
            # Select a specific index to use for shifting, like the first non-zero index
            if len(nonzero_indices) > 0:
                actual_nonzero_index = nonzero_indices[0] - impulse.mid
                shifted_impulse_response = self.impulseResponse.shiftSignal(int(actual_nonzero_index))
                scaled_impulse_response = shifted_impulse_response.multiplyconstfactor(coeff)
                impulse_responses.append(scaled_impulse_response)
    
        output_signal.values = np.convolve(input_signal.values, self.impulseResponse.values)
        return impulse_responses, output_signal


# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split(',')))
n = int(input("Window size: "))
alpha = float(input("Alpha: "))

inputSignal = discreteSignal(len(price_list))
impulseSignal = discreteSignal(len(price_list))

for index, value in enumerate(price_list):
    inputSignal.setValueAtTime(index,value)
    
for k in range(n):
    h = alpha*(1-alpha)**k
    impulseSignal.setValueAtTime(k,h)

discreteLTI = DiscreteLTI(impulseSignal)
x, outputSignal = discreteLTI.output(inputSignal)
outputSignal.values[len(price_list) - n + 1:]
print(outputSignal.values)