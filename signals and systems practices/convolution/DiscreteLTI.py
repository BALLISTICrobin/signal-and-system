import numpy as np
import matplotlib.pyplot as plt
from discreteSignal import discreteSignal

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
