import numpy as np


a = np.random.randint(2,10,(2,3))
b = np.random.randint(2,10,(2,3))

row_wise=np.concatenate((a,b),axis=0)
column_wise=np.concatenate((a,b),axis=1)

print(a,'\n')
print(b,'\n')
print(row_wise,'\n')
print(column_wise,'\n')
