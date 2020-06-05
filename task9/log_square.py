import numpy as np


def log_square(array):
    return np.log(array**2)


array=np.random.randint(10,20,(4,4))

print(log_square(array))
