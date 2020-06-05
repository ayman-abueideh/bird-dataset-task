from task6.resize_functions import *


images_info_dir='../Bird_image_dataset/bird_2020_1.pkl'
dir_path='../Bird_image_dataset/images/'
bird_info=pd.read_pickle(images_info_dir)
bird_info['augmentation']='NORMAL'

tmp_bird=bird_info.copy()
tmp_bird=resize_random_images(tmp_bird[tmp_bird['is_training_image'] == 1],dir_path,save=True,inplace=True)
print(tmp_bird.shape)



