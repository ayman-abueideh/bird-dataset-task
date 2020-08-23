import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_bar_chart(names, values, barPath, title='species', rotation='0', save=False, show=True):
    y_pos = np.arange(len(names))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, names, rotation=rotation)
    plt.ylabel('images number')
    plt.title(title)
    if save:
        plt.savefig(barPath)
    if show:
        plt.show()
    plt.clf() 

def draw_multiline():
    pass
# read bird data file
images_info_dir = '../Bird_image_dataset/bird_2020_aug.pkl'
dir_path = '../Bird_image_dataset/images/'
bird_info = pd.read_pickle(images_info_dir)

#make images dir
try:
    os.makedirs('./images/')
except:
    pass

# get all class names
class_names = bird_info[bird_info['class_name'].duplicated() == False]['class_name']

# get all the new and old number of images for each class
species_data = {}
for bird_class in class_names:
    normal_images = len(bird_info[(bird_info['class_name'] == bird_class) & (bird_info['augmentation'] == 'NORMAL')])
    augmeented_images = len(bird_info[(bird_info['class_name'] == bird_class)])
    species_data.update({bird_class: [normal_images, augmeented_images]})

# save the bar graph for each class
bar_names = ['normal_images', 'all_images']
for species in species_data:
    draw_bar_chart(bar_names, species_data[species],
                   barPath='./images/' + species + '.png', title=species, save=True, show=False)

