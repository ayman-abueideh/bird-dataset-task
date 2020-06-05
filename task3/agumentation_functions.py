import numpy as np
import random
from scipy import ndimage,signal
import math
import cv2
from task4.AugFun import AugFun
def read_image(image_path):
    image = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
    return image

def write_image(image_path,image):
    cv2.imwrite(image_path , image)

def convolve(image,kernel):
    shape=image.shape
    if len(shape)==3:
        for i in range(shape[2]):
            image[..., i] = np.clip(signal.convolve2d(image[..., i], kernel, mode='same', boundary="symm"),0, 255).astype('uint8')
    elif len(shape)==2:
        image = np.clip(signal.convolve2d(image, kernel, mode='same', boundary="symm"), 0, 255).astype('uint8')
    return image


# crop for a row
def change_point_position(parts: list, x, y, shape):
    parts[:, (x - 1):x] = parts[:, (x - 1):x] - shape[0]
    parts[:, x:y] = parts[:, x:y] - shape[1]

    minus_values = parts < 0
    parts[minus_values] = 0
    return parts


def change_mturk_parts_positions(parts: dict, image_shape):
    for key in parts:
        parts[key] = np.array(parts[key])
        parts[key][:, :2] = change_point_position(parts[key][:, :2], 1, 2, image_shape)
    return parts


def crop_box(box, shape, rescale_value):
    tmp_box = box.copy()
    rescale_shape = [int(shape[0] * rescale_value), int(shape[1] * rescale_value)]
    parts_shape = [0, 0]
    image_shape = [0, 0, 0, 0]
    # top
    if rescale_shape[0] < box[1]:
        tmp_box[1] = tmp_box[1] - (rescale_shape[0])
        parts_shape[1] = rescale_shape[0]
        image_shape[0] = rescale_shape[0]
    # bottom
    if shape[0] - rescale_shape[0] > box[1] + box[3]:
        image_shape[2] = rescale_shape[0]
    # left
    if rescale_shape[1] < box[0]:
        tmp_box[0] = tmp_box[0] - rescale_shape[1]
        parts_shape[0] = rescale_shape[1]
        image_shape[1] = rescale_shape[1]

    # right
    if shape[1] - rescale_shape[1] > box[0] + box[2]:
        image_shape[3] = rescale_shape[1]

    return tmp_box, image_shape, parts_shape


def crop_image(image, locations):
    tmp_image = image[locations[0]:image.shape[0] - locations[2], locations[1]:image.shape[1] - locations[3]]
    return tmp_image


def crop(image_data, image, image_path, base_id, save=False):
    box, image_shape, parts_shape = crop_box(np.array(image_data.ground_truth_bounding_boxes), image.shape, .2)
    tmp_image = crop_image(image, image_shape)
    image_data.ground_truth_bounding_boxes = box
    image_data.detected_bounding_box = np.array(box) + 5
    image_data.part_annotations = change_point_position(np.array(image_data.part_annotations), 2, 3, parts_shape)
    image_data.Muturk_part_locations = change_mturk_parts_positions(dict(image_data.Muturk_part_locations), parts_shape)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.SCALE + '.png'
    image_data.augmentation = AugFun.SCALE
    if save:
        write_image(image_path + image_data.image_name, tmp_image)
    return image_data

# flip horizental for a row
def image_horizental_flip(image:np.ndarray):
    # horizental_image=np.flip(image,1)
    horizental_image=image[:, ::-1]
    return horizental_image

def box_horizental_flip(box:list,image_shape):
    box[0] = image_shape[1] - box[2]-box[0]
    return box

def parts_horizental_flip(parts:list,image_shape):
    parts[:,1:2]=image_shape[1]-parts[:,1:2]
    return parts

def mturk_parts_horizental_flip(parts:dict,image_shape):
    for key in parts:
        parts[key]=np.array(parts[key])
        parts[key][:,:1]=image_shape[1]-parts[key][:,:1]
    return parts

def horizental(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    tmp_image= image_horizental_flip(image)
    box=box_horizental_flip(np.array(image_data.ground_truth_bounding_boxes),image.shape)
    image_data.ground_truth_bounding_boxes = box
    image_data.detected_bounding_box = np.array(box) + 5
    image_data.part_annotations = parts_horizental_flip(np.array(image_data.part_annotations),image.shape)
    image_data.Muturk_part_locations = mturk_parts_horizental_flip(dict(image_data.Muturk_part_locations), image.shape)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.HORIZENTAL + '.png'
    image_data.augmentation = AugFun.HORIZENTAL
    if save:
        write_image(image_path + image_data.image_name, tmp_image)
    return  image_data

# flip vertical for a row

def image_vertical_flip(image: np.ndarray, rand):
    vertical_image = np.flip(image, 0)
    if rand:
        vertical_image = np.flip(vertical_image, 1)
    return vertical_image


def box_vertical_flip(box: list, image_shape, rand):
    box[1] = image_shape[0] - box[3] - box[1]
    if rand:
        box[0] = image_shape[1] - box[2] - box[0]
    return box


def parts_vertical_flip(parts: list, image_shape, rand):
    parts[:, 2:3] = image_shape[0] - parts[:, 2:3]
    if rand:
        parts[:, 1:2] = image_shape[1] - parts[:, 1:2]

    return parts


def mturk_parts_vertical_flip(parts: dict, image_shape, rand):
    for key in parts:
        parts[key] = np.array(parts[key])
        parts[key][:, 1:2] = image_shape[0] - parts[key][:, 1:2]
        if rand:
            parts[key][:, :1] = image_shape[1] - parts[key][:, :1]

    return parts


def vertical_flip(image: np.ndarray, rand: bool):
    tmp_image = image_vertical_flip(image, rand)
    return tmp_image

def vertical(image_data,image, image_path, base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    rand = bool(random.getrandbits(1))
    tmp_image = vertical_flip(image, rand)
    box =  box_vertical_flip(np.array(image_data.ground_truth_bounding_boxes), image.shape, rand)
    image_data.ground_truth_bounding_boxes = box
    image_data.detected_bounding_box = np.array(box) + 5
    image_data.part_annotations =parts_vertical_flip(np.array(image_data.part_annotations),image.shape,rand)
    image_data.Muturk_part_locations = mturk_parts_vertical_flip(dict(image_data.Muturk_part_locations),image.shape,rand)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.VERTICAL + '.png'
    image_data.augmentation = AugFun.VERTICAL
    if save:
        write_image(image_path + image_data.image_name, tmp_image)
    return image_data


#translation  for a row

def translate_box(box,shift):
    box[0] -= shift[0]
    box[1] -= shift[1]
    return box

def translate_image(image: np.ndarray, box):
    sdx = random.randint(-(image.shape[1] - (box[0] + box[2])), box[0])
    sdy = random.randint(-(image.shape[0] - (box[1] + box[3])), box[1])
    mat = [[1, 0, sdy],
           [0, 1, sdx],
           [0, 0, 1]]
    image = image.copy()
    if len(image.shape) == 2:
        image = ndimage.affine_transform(image, mat)
    else:
        image[:, :, 0] = ndimage.affine_transform(image[:, :, 0], mat)
        image[:, :, 1] = ndimage.affine_transform(image[:, :, 1], mat)
        image[:, :, 2] = ndimage.affine_transform(image[:, :, 2], mat)
    return image ,[sdx,sdy]

def translate(image_data,image, image_path, base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    tmp_image, shift = translate_image(image, image_data.ground_truth_bounding_boxes)
    image_data.ground_truth_bounding_boxes = translate_box(np.array(image_data.ground_truth_bounding_boxes),shift)
    image_data.detected_bounding_box = np.array(image_data.ground_truth_bounding_boxes) + 5
    image_data.part_annotations = change_point_position(np.array(image_data.part_annotations), 2, 3, shift)
    image_data.Muturk_part_locations = change_mturk_parts_positions(dict(image_data.Muturk_part_locations), shift)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.TRANSLATE + '.png'
    image_data.augmentation = AugFun.TRANSLATE
    if save :
        write_image(image_path + image_data.image_name, tmp_image)
    return image_data


#rotate an image

def rotate_pixels(pixels, theta, center):
    # print(type(pixels),pixels,)
    # theta=-(180*angle)/math.pi
    pixels[:, :1] = pixels[:, :1] - center[1]
    pixels[:, 1:2] = pixels[:, 1:2] - center[0]

    pixels[:, :1] = pixels[:, :1] * math.cos(theta) - pixels[:, 1:2] * math.sin(theta)
    pixels[:, 1:2] = pixels[:, :1] * math.sin(theta) + pixels[:, 1:2] * math.cos(theta)

    pixels[:, :1] = np.round(pixels[:, :1] + center[1])
    pixels[:, 1:2] = np.round(pixels[:, 1:2] + center[0])
    return pixels


def rotate_mturk_parts(parts: dict, theta, center):
    for key in parts:
        parts[key] = np.array(parts[key])
        parts[key][:, :2] = rotate_pixels(parts[key][:, :2], theta, center)
    return parts


def is_rotatable(box,shape,theta,center):
    box_tmp = np.array([
        [box[0], box[1]],
        [box[0] + box[2], box[1]],
        [box[0], box[1] + box[3]],
        [box[0] + box[2], box[1] + box[3]]
    ])
    box_tmp = rotate_pixels(box_tmp, theta, center)
    tmp_box = [box_tmp[2][0],box_tmp[3][1]-box_tmp[2][1],box_tmp[1][0]-box_tmp[2][0],box_tmp[2][1]]
    condition=(box_tmp > 0).all() and (box_tmp[:, :1] < shape[1]).all() and (box_tmp[:, 1:2] < shape[0]).all()
    return condition,tmp_box


def rotate_image(image,theta):

    mat = [
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]]
    image = ndimage.affine_transform(image, mat)  # around 0,0
    # image = ndimage.rotate(image, angle, reshape=False) around image centre
    return image

def rotate(image_data,image, image_path, base_id,save=False):
    # image = read_image(image_path + image_data.image_name)

    center = [0, 0]
    # center = [round(image.shape[0] / 2), round(image.shape[1] / 2)]
    angle = random.randint(0, 15)
    theta = angle * math.pi / 180

    condition,box=is_rotatable(np.array(image_data.ground_truth_bounding_boxes),image.shape,theta,center)
    if condition:
        tmp_image = rotate_image(image,theta)
        image_data.ground_truth_bounding_boxes = box
        image_data.detected_bounding_box = np.array(box) + 5
        image_data.part_annotations = rotate_pixels(np.array(image_data.part_annotations), theta, center)
        image_data.Muturk_part_locations = rotate_mturk_parts(dict(image_data.Muturk_part_locations), theta, center)
        image_data.image_id = base_id
        image_data.image_name = image_data.image_name + AugFun.ROTATE + '.png'
        image_data.augmentation = AugFun.ROTATE

        if save:
            write_image(image_path + image_data.image_name, tmp_image)
            return image_data


# Blurring function


def blurring_image(image: np.ndarray, sigma=5):
    blurred = image.copy()
    n=5
    kernel=np.ones((n,n))/n**2
    blurred=convolve(blurred,kernel)
    return blurred


def blurring(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.BLUR + '.png'
    image_data.augmentation = AugFun.BLUR
    if save:
        blurred = blurring_image(image, sigma=random.randrange(1, 10, 2))
        write_image(image_path+ image_data.image_name, blurred)
    return image_data

#sharpen filter


def sharpen_image(image: np.ndarray):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpend=convolve(image,kernel)
    return sharpend

def sharpen(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.SHARPEN + '.png'
    image_data.augmentation = AugFun.SHARPEN

    if save:
        sharpend = sharpen_image(image)
        write_image(image_path + image_data.image_name, sharpend)
    return image_data

# gray filter

def to_gray(image):
    gray=image
    if len(image.shape)==3:
        gray=np.round(np.dot(image[...,:image.shape[2]],[0.2989, 0.5870, 0.1140])).astype('uint8')
    return gray

def gray(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.GRAYSCALE + '.png'
    image_data.augmentation = AugFun.GRAYSCALE
    if save:
        gray = to_gray(image)
        write_image(image_path + image_data.image_name, gray)
    return image_data

#adding noise

def random_noise(image):
    prop = random.uniform(.6, .8)
    gauss = np.random.normal(0, image.std(), image.shape).astype('uint8')
    choice = np.random.choice([0, 1], image.shape, replace=True, p=[prop, 1 - prop]).astype('uint8')
    gauss = gauss * choice
    noisy = image + gauss
    return noisy

def add_noise(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.NOISE + '.png'
    image_data.augmentation = AugFun.NOISE
    if save:
        noised = random_noise(image)
        write_image(image_path + image_data.image_name, noised)
    return image_data

# salt noise
def salt_noise(image,salt_prob=.5):
    # nums = np.random.choice([0, 255], size=image.shape, p=[1-salt_prob, salt_prob]).astype('uint8')
    # image = np.clip(image + nums,0,255)
    amount = random.uniform(.2,.4)*salt_prob
    num_salt = np.ceil(amount * image.size * salt_prob)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape[:2]]
    image[coords[0], coords[1]] = 255
    return image

def add_salt_noise(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.SALT + '.png'
    image_data.augmentation = AugFun.SALT
    if save:
        noised = salt_noise(image, random.uniform(.5, .6))
        write_image(image_path + image_data.image_name, noised)
    return image_data

#pepper noise


def pepper_noise(image,pepper_prob=.5):
    # nums = np.random.choice([0, 1], size=image.shape, p=[pepper_prob, 1-pepper_prob]).astype('uint8')
    # image = image * nums

    amount = random.uniform(.2, .4)*pepper_prob
    num_salt = np.ceil(amount * image.size * pepper_prob)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape[:2]]
    image[coords[0], coords[1]] = 0
    return image

def add_pepper_noise(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.PEPPER + '.png'
    image_data.augmentation = AugFun.PEPPER

    if save:
        noised = pepper_noise(image, random.uniform(.5, .6))
        write_image(image_path + image_data.image_name, noised)
    return image_data

# add slat and pepper

def add_salt_pepper_noise(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)

    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.SALT_PEPPER + '.png'
    image_data.augmentation = AugFun.SALT_PEPPER

    if save:
        noised = pepper_noise(image, random.uniform(.4, .5))
        noised = salt_noise(noised, random.uniform(.4, .5))
        write_image(image_path + image_data.image_name, noised)
    return image_data

# contrast

def channel_hist(channel):
    h_channel, bin_channel = np.histogram(channel.flatten(), 256, [0, 256])
    cdf_channel = np.cumsum(h_channel)
    cdf_m_channel = np.ma.masked_equal(cdf_channel, 0)
    cdf_m_channel = np.clip((cdf_m_channel - cdf_m_channel.min()) * 255 / (cdf_m_channel.max() - cdf_m_channel.min()),0,255)
    cdf_final_channel = np.ma.filled(cdf_m_channel, 0).astype('uint8')
    img_channel = cdf_final_channel[channel]
    return img_channel
def image_hist(image):

    if len(image.shape)==3:
        for i in range(image.shape[2]):
            channel=image[:,:,i]
            image[:,:,i]=channel_hist(channel)
    elif len(image.shape)==2:
        image=channel_hist(image)
    return image

def contrast(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.CONTRAST + '.png'
    image_data.augmentation = AugFun.CONTRAST
    if save:
        image = image_hist(image)
        write_image(image_path + image_data.image_name, image)
    return image_data

# brightness
def brightness(image,gamma):
    image = np.clip(255 * ((image / 255) ** gamma), 0, 255).astype('uint8')
    return image

def change_brightness(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.BRIGHTNESS + '.png'
    image_data.augmentation = AugFun.BRIGHTNESS
    if save:
        gamma = random.randrange(4, 15) / 10
        image = brightness(image, gamma)
        write_image(image_path + image_data.image_name, image)
    return image_data
#invert colors
def invert_image(image):
    image= 255-image
    return image

def invert(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.INVERT + '.png'
    image_data.augmentation = AugFun.INVERT
    if save:
        image = invert_image(image)
        write_image(image_path + image_data.image_name, image)
    return image_data

# coarse dropout

def image_coarse_dropout(image):

    dropout_num=random.randint(15,20)
    height_factor=int(image.shape[1]/dropout_num)
    width_factor=int(image.shape[0]/dropout_num)
    for i in range(dropout_num):
        y1 = random.randint(i,height_factor+i)* random.randint(10,height_factor+dropout_num)
        y1 = y1%image.shape[1]
        y2=y1+random.randint(10,30)
        x1 = random.randint(i,width_factor+i) * random.randint(10, width_factor+dropout_num)
        x1 = x1 % image.shape[0]
        x2 = x1 + random.randint(10,30)
        image[x1:x2,y1:y2]=0
    return image

def coarse_dropout(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id = base_id
    image_data.image_name = image_data.image_name + AugFun.COARSE_DROPOUT + '.png'
    image_data.augmentation = AugFun.COARSE_DROPOUT
    if save:
        image = image_coarse_dropout(image)

        write_image(image_path + image_data.image_name, image)
    return image_data

# channel multiplication
def channel_mult(image):
    mul_value=random.randrange(50,200,20)/100
    if len(image.shape)==3:
        channel=random.randint(0,2)
        image[:, :, 0] = np.clip(image[:, :, 0] * (mul_value) if channel==0 else image[:, :, 0]/mul_value,0,255)
        image[:, :, 1] = np.clip(image[:, :, 1] * (mul_value) if channel==1 else image[:, :, 1]/mul_value,0,255)
        image[:, :, 2] = np.clip(image[:, :, 2] * (mul_value) if channel==2 else image[:, :, 2]/mul_value,0,255)
    else :
        image=np.clip((image*mul_value).astype('uint8'),0,255)
    return image

def mult_channel(image_data,image,image_path,base_id,save=False):
    # image = read_image(image_path + image_data.image_name)
    image_data.image_id=base_id
    image_data.image_name = image_data.image_name + AugFun.CHANNEL_MULT + '.png'
    image_data.augmentation = AugFun.CHANNEL_MULT
    if save:
        image = channel_mult(image)
        write_image(image_path + image_data.image_name, image)

    return image_data


