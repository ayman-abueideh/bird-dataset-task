import pandas as pd
import time
from task3.agumentation_functions import *

images_info_dir = '../Bird_image_dataset/bird_2020_1.pkl'
dir_path = '../Bird_image_dataset/images/'

bird_info = pd.read_pickle(images_info_dir)
bird_info.head()

start_time = time.time()

def augmentation(image_df: pd.DataFrame,save=False):
    images = []
    base_id = image_df.shape[0]
    for index, row in image_df.iterrows():
        if row.is_training_image == 1:
            image = read_image(dir_path + row.image_name)
            aug_images = [
                horizental(row.copy(), image.copy(), dir_path, base_id + 1, save=save),
                vertical(row.copy(), image.copy(), dir_path, base_id + 2, save=save),
                crop(row.copy(), image.copy(), dir_path, base_id + 3, save=save),
                translate(row.copy(), image.copy(), dir_path, base_id + 4, save=save),
                rotate(row.copy(), image.copy(), dir_path, base_id + 5, save=save),
                add_noise(row.copy(), image.copy(), dir_path, base_id + 6, save=save),
                add_salt_noise(row.copy(), image.copy(), dir_path, base_id + 7, save=save),
                add_pepper_noise(row.copy(), image.copy(), dir_path, base_id + 8, save=save),
                add_salt_pepper_noise(row.copy(), image.copy(), dir_path, base_id + 9, save=save),
                contrast(row.copy(), image.copy(), dir_path, base_id + 10, save=save),
                coarse_dropout(row.copy(), image.copy(), dir_path, base_id + 11, save=save),
                change_brightness(row.copy(), image.copy(), dir_path, base_id + 12, save=save),
                blurring(row.copy(), image.copy(), dir_path, base_id + 13, save=save),
                sharpen(row.copy(), image.copy(), dir_path, base_id + 14, save=save),
                gray(row.copy(), image.copy(), dir_path, base_id + 15, save=save),
                invert(row.copy(), image.copy(), dir_path, base_id + 16, save=save),
                mult_channel(row.copy(), image.copy(), dir_path, base_id + 17, save=save)
            ]
            base_id += len(aug_images)
            images.extend(aug_images)
    images = [image for image in images if image is not None]
    new_df = pd.DataFrame(data=images, columns=image_df.columns)
    new_df.index = list(range(image_df.shape[0], len(images) + image_df.shape[0]))
    return new_df

tmp_bird_info = bird_info.copy()
print(tmp_bird_info.shape)
tmp_bird_info = tmp_bird_info.append(augmentation(tmp_bird_info))

print(time.time() - start_time)
Time=time.time()
tmp_bird_info.to_pickle('bird_2020_aug1.pkl')
tmp_bird_info.to_csv('bird_2020_aug1.csv')
print(time.time() - Time)







