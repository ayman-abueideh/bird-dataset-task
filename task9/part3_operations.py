import numpy as np
import math
# array=np.arange(100,200,3.5).reshape(7,4)


def print_attr(array):
    print('shape', array.shape)
    print('dimension', array.ndim)
    print('length', array.size)

def get_column(array,column):
    return array[:,column]

def get_odd_rows_even_columns(array):
    return array[::2,1::2]

def get_sqrt(array):
    return np.sqrt(array)

def split_array(array):
    array1 = array[:math.floor(array.shape[0] / 2), :]
    array2 = array[math.floor(array.shape[0] / 2):array.shape[0], :]
    return array1,array2

def max_min(array,axis=(0,1)):
    max=np.max(array,axis=axis[0])
    min=np.max(array,axis=axis[1])
    return max,min

def delete_insert(array,column,axis=1):
    tmp_array=np.delete(array,column,axis)
    tmp_array = np.insert(tmp_array, column, np.zeros(tmp_array.shape[(axis+1)%2]), axis=axis)
    return tmp_array


# testing

array=np.random.randint(100,200,(7,3))
print('array',array)

# print_attr(array)
# print(get_column(array,0))
# print(get_odd_rows_even_columns(array))
# print(get_sqrt(array))
# print(max_min(array))
# print(delete_insert(array,0,0))
