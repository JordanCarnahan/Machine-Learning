#CSCI_440-10-m+c_eg_py_range_vs_numpy_arange

#Py: range()
# This loop will run 5 times, but the value of _ (0 through 4) is ignored inside the loop
#Basically it means you are not interested in how many times the loop is run till, just that it should run some specific number of times overall.
for _ in range(5):
    # do something that does not depend on the loop counter
    print("Loop iteration (index not used)")

#Numpy: arange()
#1. np.arange(stop): Generates values from 0 up to (but not including) stop.
import numpy as np
arr = np.arange(5)
# Result: array([0, 1, 2, 3, 4])

#2. np.arange(start, stop): Generates values from start up to (but not including) stop
import numpy as np
arr = np.arange(2, 7)
# Result: array([2, 3, 4, 5, 6])

#3. np.arange(start, stop, step): Generates values with a specified spacing.
import numpy as np
arr = np.arange(0, 10, 2)
# Result: array([0, 2, 4, 6, 8])

#4. Floating-point steps: numpy.arange() is more suitable for use with non-integer step sizes than Python's range function, though the results can sometimes be length-unstable due to floating-point arithmetic.
#For floating-point ranges where you want to ensure the endpoint is handled correctly or the number of samples is precise, numpy.linspace() is often preferred. 
import numpy as np
arr = np.arange(0.1, 0.5, 0.1)
# Result: array([0.1, 0.2, 0.3, 0.4])
