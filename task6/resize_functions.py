import numpy as np
from PIL import Image
from task4.AugFun import AugFun
from task3.agumentation_functions import write_image
import pandas as pd
import random
from task3.agumentation_functions import read_image


def resize_image(image, rescale_values):
    rescaled_image = np.array(Image.fromarray(image).resize((rescale_values[0], rescale_values[1])))
    return rescaled_image


def resize_box(box, scale):
    # print(box,scale)
    tmp_box = box.copy()
    box[0] = int(np.round(box[0] * scale[0]))
    box[1] = int(np.round(box[1] * scale[1]))
    box[2] = int(np.round(box[2] * scale[0]))
    box[3] = int(np.round(box[3] * scale[1]))
    return box


def resize_pixles(pixels, x, y, scale):
    pixels[:, (x - 1):x] = np.round(pixels[:, (x - 1):x] * scale[0])
    pixels[:, x:y] = np.round(pixels[:, x:y] * scale[1])
    return pixels


def resize_muturk_parts(parts, scale):
    for key in parts:
        parts[key] = np.array(parts[key])
        parts[key][:, :2] = resize_pixles(parts[key][:, :2], 1, 2, scale)
    return parts


def resize(image_data, image, image_path, base_id, image_name=AugFun.RESIZE + '.png',rescale_value=(250, 250), save=False):
    tmp_image = resize_image(image, rescale_value)
    scale = [tmp_image.shape[1] / image.shape[1], tmp_image.shape[0] / image.shape[0]]
    image_data.ground_truth_bounding_boxes = resize_box(image_data.ground_truth_bounding_boxes.copy(), scale)

    image_data.detected_bounding_box = np.array(image_data.ground_truth_bounding_boxes) + 5
    image_data.part_annotations = resize_pixles(np.array(image_data.part_annotations), 2, 3, scale)
    image_data.Muturk_part_locations = resize_muturk_parts(dict(image_data.Muturk_part_locations), scale)
    image_data.image_id = base_id + image_data.image_id
    image_data.image_name = image_data.image_name + image_name
    image_data.augmentation = AugFun.RESIZE
    if save:
        write_image(image_path + image_data.image_name, tmp_image)
    return image_data


def resize_random_images(image_df, dir_path,save=False,inplace=False):
    class_names = image_df[image_df['class_name'].duplicated() == False]['class_name']
    base_id = image_df.shape[0] + 1
    rows = []
    for species in class_names:
        index = random.choice(image_df[image_df['class_name'] == species].index)
        image = read_image(dir_path + image_df.loc[index].image_name)
        if inplace:
            image_df.loc[index] = resize(image_df.loc[index], image, dir_path, 0,image_name='', save=save)
        else:
            rows.append(resize(image_df.loc[index], image, dir_path, base_id, save=save))
            base_id += 1
    if inplace:
        return image_df
    new_df = pd.DataFrame(data=rows, columns=image_df.columns)
    new_df.index = list(range(image_df.shape[0], len(rows) + image_df.shape[0]))
    return new_df


def resize_images(image_df, dir_path,Index=0,save=False):
    rows = []
    for index,row in image_df.iterrows():
        image = read_image(dir_path + row.image_name)
        rows.append(resize(row.copy(), image, dir_path,Index, save=save))

    new_df = pd.DataFrame(data=rows, columns=image_df.columns)
    return new_df
