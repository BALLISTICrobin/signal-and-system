import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from continuousSignal import ContinuousSignal

class continuousLTI:
    def __init__(self, impulse_response):
        
        if not isinstance(impulse_response, ContinuousSignal):
            raise ValueError("Impulse response must be an instance of ContinuousSignal.")
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal, delta):
        t_values = np.arange(-3/delta, 3/delta,1)
        impulses = []
        coefficients = []
        for t0 in t_values:
            coeff = input_signal.func(t0 * delta)
            coefficients.append(coeff) 
            # print(coeff)
            rectangular_impulse_signal = ContinuousSignal(
                npFunc=lambda t, t0=t0, coeff=coeff: 1 * ((t >= t0 * delta) & (t < (t0 + 1) * delta)).astype(int)
            )
            impulses.append(rectangular_impulse_signal)
        
        return impulses, coefficients


    def output(self, input_signal, delta):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal, delta)
        output_signal = ContinuousSignal(npFunc=lambda t: 0)  # Initialize an output signal as zero
        impulse_responses = []
        for i, impulse in enumerate(impulses):
            t0 = -(3/delta-i+1)*delta  # adjust t0 based on delta and position in impulses
            
            # Get the impulse response shifted by t0 and scaled by coefficient
            shifted_impulse_response = self.impulse_response.shift(t0).multiply_const_factor(coefficients[i]*delta)
            impulse_responses.append(shifted_impulse_response)
            output_signal = output_signal.add(shifted_impulse_response)  # Accumulate the output signal

        return output_signal,impulse_responses