import numpy as np

def transpose_mult(array):
    return np.dot(array.transpose(),array)
    # return np.dot(array,array.transpose())

array=np.random.randint(1,10,(2,3))
print(transpose_mult(array))
