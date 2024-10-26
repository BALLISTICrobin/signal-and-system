import numpy as np
import matplotlib.pyplot as plt
from discreteSignal import discreteSignal  
from DiscreteLTI import DiscreteLTI


def main():
    img_root_path = '/Users/niloydas/Desktop/python programming/signals and systems practices/convolution/images/discrete'
    
    impulse_response_signal = discreteSignal(5) 
    impulse_response_signal.setValueAtTime(0, 1) 
    impulse_response_signal.setValueAtTime(1, 1)
    impulse_response_signal.setValueAtTime(2, 1)
    
    dtsLTI = DiscreteLTI(impulse_response_signal)
    
    input_signal = discreteSignal(5)
    input_signal.setValueAtTime(0, 0.5)
    input_signal.setValueAtTime(1, 2)
    
    time_indices = np.arange(-input_signal.INF, input_signal.INF + 1, 1)
    plt.figure(figsize=(10,6))
    input_signal.plot(time_indices,"sum")
    plt.savefig(f'{img_root_path}/input.png')
    impulses, coefficients = dtsLTI.linear_combination_of_impulses(input_signal)
    

    fig, axes = plt.subplots(input_signal.INF + 1, 2, figsize=(8, 8))
    
    
    index = 0
    for row in range(input_signal.INF + 1):
        for column in range(2):
            if index >= len(impulses):
                break
            input_impulses = impulses[index].multiplyconstfactor(coefficients[index])
            title = rf"$\delta[n - ({index-input_signal.INF})] x[{index-input_signal.INF}]$"
            input_impulses.plot(time_indices, title,ax=axes[row, column])
            index += 1
    
    input_impulse_sum = impulses[0].multiplyconstfactor(coefficients[0])
    for i in range(1,len(impulses),1):
        input_impulse_sum = input_impulse_sum.add(impulses[i].multiplyconstfactor(coefficients[i]))
    
    input_impulse_sum.plot(time_indices,"sum",ax=axes[input_signal.INF, 1])
    plt.tight_layout()
    plt.savefig(f'{img_root_path}/linear_combination_of_impulses.png')
    # plt.show()


    impulse_responses, output_signal = dtsLTI.output(input_signal)
    fig, axes = plt.subplots(input_signal.INF + 1, 2, figsize=(8, 8))
    
    index = 0
    for row in range(input_signal.INF + 1):
        for column in range(2):
            if index >= len(impulse_responses):
                break
            title = f"h[n - ({index-input_signal.INF})] x[{index-input_signal.INF}]"
            impulse_responses[index].plot( time_indices,title,ax =axes[row, column])
            index += 1
    

    output_time_indices = np.arange(len(output_signal.values))
    output_signal.plot(output_time_indices, "Output Sum",ax=axes[input_signal.INF, 1])
    
    plt.tight_layout()
    plt.savefig(f'{img_root_path}/impulse_responses_output.png')
    # plt.show()

if __name__ == "__main__":
    main()