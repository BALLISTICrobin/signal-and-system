# discrete time step signal
# u[n-a] = {1; n>=a} and {0 ; otherwise}

import numpy as np
import matplotlib.pyplot as mpl

a = int(input("enter a: "))
LL = -10
HL = 10
size = HL-LL
u = np.zeros(size,dtype=np.int32)
print(u)