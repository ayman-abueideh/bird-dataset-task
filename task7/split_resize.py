import time
from task6.resize_functions import *

images_info_dir = '../Bird_image_dataset/bird_2020_1.pkl'
dir_path = '../Bird_image_dataset/'
training_path = 'images/'
testing_path = 'testing_images/'
bird_info = pd.read_pickle(images_info_dir)

bird_info['augmentation'] = 'NORMAL'

start_time = time.time()
tmp_bird = bird_info.copy()
training_set = resize_images(tmp_bird[tmp_bird['is_training_image'] == 1], dir_path + training_path, save=False,
                             Index=tmp_bird.shape[0])
testing_set = resize_images(tmp_bird[tmp_bird['is_training_image'] == 0], dir_path + testing_path, save=False,
                            Index=tmp_bird.shape[0])
print(time.time() - start_time)

training_set.to_csv('training_dataset.csv')
testing_set.to_csv('testing_dataset.csv')
