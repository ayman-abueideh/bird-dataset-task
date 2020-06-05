import numpy as np

def jacSim(box_a,box_b):
    sim_box=np.ndarray(shape=(box_a.shape))
    for i in range(sim_box.shape[0]):
        # sim_box[i]=box_a[i]*box_b/(box_a[i]**2+box_b**2-box_a[i]*box_b)
        sim_box[i]=np.sum(box_a[i]*box_b/(box_a[i]**2+box_b**2-box_a[i]*box_b))/sim_box[i].shape[0]
    return sim_box



#Testing
a_box=np.random.randint(1,10,(4,4),'uint8')
a_box[0]=[1,2,3,1]

b_box=a_box[0]+a_box[0]*.05
b_box=np.array([9,1,4,1])
print('box_a \n',a_box,'\n')
print('box_b \n',b_box,'\n')

print('Jacsim\n',jacSim(a_box,b_box))

