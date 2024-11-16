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
            
            