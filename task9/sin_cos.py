import  numpy as np
import math
import matplotlib.pyplot as plt

range=360
array=np.arange(0,range)
angle_array=(array*math.pi)/180
sin_array=np.sin(angle_array)
cos_array=np.cos(angle_array)

plt.scatter(array,sin_array)
plt.scatter(array,cos_array)
plt.show()
