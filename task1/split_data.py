import os
import shutil
from random import shuffle
import math

# initialize all directories
base_dir='../Bird_image_dataset'
image_dir= base_dir +'/images'
testing_dir=base_dir+'/testing_images'
training_dir=base_dir+'/training_images'

#reading image files
files=list(os.walk(image_dir))

#storing subdirectories paths
dirs_list=list(map(lambda dir: testing_dir+'/'+dir ,files[0][1]))

#create subdirectories for the testing_images
for dir in dirs_list:
    try:
        os.makedirs(dir)
    except:
        pass

#storing images paths
testing_images_list=[]
files_count=0
for path,subdir,name in files[1:]:
    files_count+=len(name)
    testing_count=int(math.ceil(len(name)*.3))
    shuffle(name)
    testing_images_list.extend(list(map(lambda x: path+'/'+x,name[:testing_count])))
shuffle(testing_images_list)

print('number of training images = ',files_count-len(testing_images_list))
print('number testing images = ',len(testing_images_list))

#move the testing files to testing folder
for file in testing_images_list:
    shutil.move(file,os.path.dirname(file.replace(image_dir,testing_dir)))

