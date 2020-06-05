import numpy as np
import pandas as pd
file_path='../Bird_image_dataset/classes.txt'

data_df =pd.read_csv(file_path,delim_whitespace=True,header=None,names=['class_id','class_name'])
classes_number=data_df.shape[0]
one_hot=[]
for index,row in data_df.iterrows():
    array=np.zeros((classes_number,))
    array[index]=1
    one_hot.append(array)
one_hot=np.array(one_hot)
# print(one_hot)
print(one_hot.shape)


