import numpy as np
import matplotlib.pyplot as plt

class ContinuousSignal:
    def __init__(self, npFunc, func=None):
        self.func = func 
        self.npFunc = npFunc
    
    def shift(self, shift):
        
        shifted_func = lambda t: self.npFunc(t - shift)
        return ContinuousSignal(shifted_func)
    
    def add(self,other):
        if not isinstance(other, ContinuousSignal):
            raise ValueError("The operand must be an instance of ContinuousSignal.")
        
        added_func = lambda t: self.npFunc(t) + other.npFunc(t)
        return ContinuousSignal(added_func)
    
    def multiply(self, other):
        if not isinstance(other, ContinuousSignal):
            raise ValueError("The operand must be an instance of ContinuousSignal.")
        
        multiplied_func = lambda t: self.npFunc(t) * other.npFunc(t)
        return ContinuousSignal(multiplied_func)
    
    def multiply_const_factor(self, scaler):
      
        scaled_func = lambda t: scaler * self.npFunc(t)
        return ContinuousSignal(scaled_func)
    
    def plot(self, t_start, t_end, ax=None, color='blue'):
        t_values = np.linspace(t_start, t_end, 1000)
        signal_values = self.npFunc(t_values)
        x_ticks = np.arange(t_start, t_end+1,1)
        y_ticks = np.arange(0, signal_values.max() + 0.1, 0.1)
        if ax is None:
            plt.plot(t_values, signal_values,color=color)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            plt.grid(axis="both")
            # plt.title(title)
        else:
            ax.plot(t_values, signal_values,color=color)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.grid(axis="both")
            # ax.title(title)