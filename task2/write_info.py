import pandas as pd
from pathlib import Path
import time
import numpy as np

dir_path='../Bird_image_dataset/'
classes_ids_path=dir_path+'classes.txt'
images_ids_path=dir_path+'images.txt'
image_labels_path=dir_path+'image_class_labels.txt'
bounding_boxes_path=dir_path+'bounding_boxes.txt'
parts_path=dir_path+'parts/parts.txt'
parts_loc_path=dir_path+'parts/part_locs.txt'
parts_click_locs_path=dir_path+'parts/part_click_locs.txt'
attribute_path=dir_path+'attributes/attributes.txt'

start_time=time.time()

#read images.txt file
images_df=pd.read_csv(images_ids_path,delim_whitespace=True,header=None,names=['image_id','image_name'])
get_training= lambda image_name: 1 if Path(dir_path+'images/'+image_name).exists() else 0
images_df['is_training_image']=np.vectorize(get_training)(images_df['image_name'].values)
print('image_id','image_name','is_training_image data finished in ', time.time()-start_time)

Time=time.time()
# #read classes.txt file
classes_df=pd.read_csv(classes_ids_path,delim_whitespace=True,header=None,names=['class_id','class_name'])

#read image_class_labels.txt
image_labels_df=pd.read_csv(image_labels_path,delim_whitespace=True,header=None,names=['image_id','class_id'])
images_df['class_id']=image_labels_df.loc[images_df['image_id']==image_labels_df['image_id'],'class_id']

class_name=images_df.merge(classes_df)['class_name']
print('class id and class names finished in ', time.time()-Time)
Time=time.time()

# read bounding_boxes.txt
bounding_boxes_df=pd.read_csv(bounding_boxes_path,delim_whitespace=True,header=None,names=['image_id','x','y','width','height'])
ground_truth_bounding_boxes=bounding_boxes_df.loc[images_df['image_id']==bounding_boxes_df['image_id'],:].values[:,1:].tolist()
detected_bounding_box=(bounding_boxes_df.loc[images_df['image_id']==bounding_boxes_df['image_id'],:].values[:,1:]+5).tolist()

print('ground and detected finished in ', time.time()-Time)
Time=time.time()

#read parts.txt file
parts_df=pd.read_csv(parts_path,delim_whitespace=True,header=None,names=['part_id','part_name'])
parts_lockup=parts_df['part_name'].values

#read parts_loc.txt file
parts_loc_df=pd.read_csv(parts_loc_path,delim_whitespace=True,header=None,names=['image_id','part_id','x','y','visible'])


#add parts_locations info to default_df
part_annotations=images_df.apply(lambda row:parts_loc_df.loc[row.image_id==parts_loc_df['image_id']].values[:,1:],axis=1)
print('part_annotations finished in ', time.time()-Time)
Time=time.time()

#add parts_locations info to default_df
image_parts_loc=np.loadtxt(parts_click_locs_path,dtype=float)
image_muturk_dict=dict()
for parts_click in image_parts_loc:
    image_id=int(parts_click[0])
    part_id= int(parts_click[1])
    part_name=parts_lockup[part_id-1]
    image_muturk_dict.setdefault(image_id,{}).setdefault(str(part_id)+','+part_name,[]).append([parts_click[2],parts_click[3],parts_click[4],parts_click[5]])

muturk=list()
for row in image_muturk_dict.values():
    muturk.append(row)

print('muturk finished in ',time.time()-Time)
Time=time.time()

attributes_df=pd.read_csv(attribute_path,delim_whitespace=True,header=None,names=['attribute_id','attribute_name'])
attribute_lockup=attributes_df['attribute_name'].values

#convert not visible to not_visible in certainties.txt file
certainties_path=dir_path+'attributes/certainties.txt'
with open(certainties_path, "r+") as file:
     data = file.read().replace('not visible','not_visible')
     file.seek(0)
     file.write(data)


#read certainties.txt file
certainties_df=pd.read_csv(certainties_path,delim_whitespace=True,header=None,names=['certainties_id','certainties_name'])
certainties_lockup=certainties_df['certainties_name'].values

#read image_attributes_labels.txt file
image_attribute_path=dir_path+'attributes/image_attribute_labels.txt'
image_attributes_df=pd.read_csv(image_attribute_path,delim_whitespace=True,header=None,names=['image_id','attribute_id','is_present','certanity_id','time'])
image_attributes_values=image_attributes_df.values

image_attributes=dict()
for attribute in image_attributes_values:
    image_id=int(attribute[0])
    attribute_id=int(attribute[1])
    attribute_name=attribute_lockup[attribute_id-1]
    certanity_id=int(attribute[3])
    certanity_name=certainties_lockup[certanity_id-1]
    image_attributes.setdefault(image_id,[]).append([attribute_id,attribute_name,attribute[2],certanity_id,certanity_name,attribute[4]])


attributes=list()
for row in image_attributes.values():
    attributes.append(row)
print('attributes finished in ',time.time()-Time)

data=pd.DataFrame({
    'image_id':images_df['image_id'],
    'image_name':images_df['image_name'],
    'is_training_image':images_df['is_training_image'],
    'class_id':images_df['class_id'],
    'class_name':class_name,
    'ground_truth_bounding_boxes':ground_truth_bounding_boxes,
    'detected_bounding_box':detected_bounding_box,
    'part_annotations':part_annotations,
    'Muturk_part_locations':muturk,
    'attributes_annotations':attributes,
})
print('creating finished in ',time.time()-start_time)

print('saving...')
Time=time.time()
data.to_csv(dir_path+'bird_2020_1.csv')
data.to_pickle(dir_path+'bird_2020_1.pkl')
print('finished saving in ',time.time()-Time)
