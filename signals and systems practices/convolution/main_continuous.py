import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
from continuousLTI import continuousLTI
from continuousSignal import ContinuousSignal

INF =5
def input_signal_func(t):
    if(t>=0):
        return np.exp(-t)
    else:
        return 0
    
def nparray_input_signal(t):
    return np.exp(-t) * (t >= 0)
def impulse_response_func(t):
    if(t>=0):
        return 1
    else:
        return 0
def np_impulse_response_func(t):
    return (t >= 0).astype(int)
def convolution(t):
        integrand = lambda tau: nparray_input_signal(tau) * np_impulse_response_func(t - tau)
        result, _ = quad(integrand, 0, t)  # Integrate from 0 to t
        return result
def main():
    img_root_path = '/Users/niloydas/Desktop/python programming/signals and systems practices/convolution/images/continuous'
    
    impulse_response = ContinuousSignal(np_impulse_response_func,impulse_response_func)
    input_signal = ContinuousSignal(nparray_input_signal,input_signal_func)

    input_signal.plot(-5,5)
    plt.savefig(f'{img_root_path}/input.png')
    lti_system = continuousLTI(impulse_response)

    index1=0
    index2=0
    # Set delta value and retrieve impulses and coefficients
    delta = 0.5
    impulses, coefficients = lti_system.linear_combination_of_impulses(input_signal, delta)

    # Set up plot grid
    index = 0
    num_columns = 3 
    num_rows = (len(impulses) // num_columns) + (len(impulses) % num_columns > 0)+1
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))

    # Plot each scaled impulse (impulse * coefficient)
    for i in range(num_rows):
        for j in range(num_columns):
            if index < len(impulses):
                # Scale impulse by corresponding coefficient
                scaled_impulse = impulses[index].multiply_const_factor(coefficients[index])
                scaled_impulse.plot(-INF, INF, ax=axes[i, j])
                axes[i, j].set_title(f"Impulse {index + 1}")
                index += 1
            else:
                index1 = i
                index2 = j

    # Combine the scaled impulses to create the reconstructed impulse
    reconstructed_impulse = impulses[0].multiply_const_factor(coefficients[0])
    for i in range(1, len(impulses)):
        scaled_impulse = impulses[i].multiply_const_factor(coefficients[i])
        reconstructed_impulse = reconstructed_impulse.add(scaled_impulse)

    # Plot the reconstructed impulse in the last available subplot
    reconstructed_impulse.plot(-INF, INF, ax=axes[index1, index2])
    axes[index1, index2].set_title("Reconstructed Signal")

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(f'{img_root_path}/linear_combination_of_impulses.png')
    

    
    
   
    plt.figure(figsize=(10, 6))

    #Reconstruction with delta
    delta_values = np.array([0.5, 0.1, 0.05, 0.01])
    index_delta = 0
    row_num = math.ceil(len(delta_values) / 2)
    column_num = 2
    fig, axes = plt.subplots(row_num, column_num, figsize=(12, 12))

 
    for row_index in range(row_num):
        for column_index in range(column_num):
            if index_delta < len(delta_values):
                delta = delta_values[index_delta]
            
              
                impulses, coefficients = lti_system.linear_combination_of_impulses(input_signal, delta)
            
                
                scaled_impulse = impulses[0].multiply_const_factor(coefficients[0])
                for i in range(1, len(impulses)):
                    scaled_impulse = scaled_impulse.add(impulses[i].multiply_const_factor(coefficients[i]))

                
                scaled_impulse.plot(-INF, INF, ax=axes[row_index, column_index])
                input_signal.plot(-INF, INF, ax=axes[row_index, column_index],color='red')
                axes[row_index, column_index].grid("both")
                axes[row_index, column_index].set_title(f"Reconstruction with delta = {delta}")

                index_delta += 1

 
    plt.tight_layout()
    plt.savefig(f'{img_root_path}/Reconstruction_of_input_signal_with_varying_delta.png')

    #output impulse response
    delta = 0.5
    output_signal, impulse_responses = lti_system.output(input_signal, delta)
    
    index = 0
    num_columns = 3 
    num_rows = (len(impulse_responses) // num_columns) + (len(impulse_responses) % num_columns > 0)+1
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))
    
    for i in range(num_rows):
        for j in range(num_columns):
            if index < len(impulse_responses):
                impulse_responses[index].plot(-INF, INF, ax=axes[i, j])
                axes[i, j].set_title(f"Impulse_response {index + 1}")
                index += 1
            else:
                index1 = i
                index2 = j
    
    output_signal.plot(-INF, INF, ax=axes[index1, index2])
    axes[index1, index2].set_title("output sum")
    axes[index1,index2].grid(True)
    plt.tight_layout()
    plt.savefig(f'{img_root_path}/Returned impulses multiplied by their coefficients.png')
    
    
    
    #output impulse response with different delta
    delta_values = np.array([0.5, 0.1, 0.05, 0.01])
    index_delta = 0
    row_num = math.ceil(len(delta_values) / 2)
    column_num = 2
    fig, axes = plt.subplots(row_num, column_num, figsize=(12, 12))
    
    #for manual integration
    time_values = np.linspace(0, 10, 100)
    output_values = [convolution(t) for t in time_values]
    
    for row_index in range(row_num):
        for column_index in range(column_num):
            if index_delta < len(delta_values):
                delta = delta_values[index_delta]
            
              
                output_signal, impulse_responses = lti_system.output(input_signal, delta)
            
                
                output_signal.plot(-INF, INF, ax=axes[row_index, column_index])
                axes[row_index, column_index].set_title(f"delta={delta}")
                axes[row_index, column_index].grid(True)
                axes[row_index, column_index].plot(time_values, output_values, label='Output y(t)', color='orange')
                index_delta += 1
                
    plt.tight_layout()
    plt.savefig(f'{img_root_path}/Approximate output with varying delta.png')
    
    
    

    
    
if __name__ == "__main__":
    main()