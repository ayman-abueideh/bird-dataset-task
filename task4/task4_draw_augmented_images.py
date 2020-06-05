import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  task3.agumentation_functions import *
from task4.AugFun import AugFun
import random

augmentation_function=['HORIZENTAL','VERTICAL','SCALE','TRANSLATE','ROTATE','BLUR','SHARPEN',]


def draw_figure(images:dict,fig_path):
    row=math.ceil(math.sqrt(len(images)))
    column=math.ceil(len(images)/row)
    fig=plt.figure(figsize=(20,20))
    for index,image in enumerate(images):
        ax=fig.add_subplot(row,column,index+1)
        ax.imshow(images[image])
        ax.axis('off')
        ax.set_title(image,fontsize=20)

    plt.savefig(fig_path,dpi=100)
    plt.show()

def get_augmentation_images(image,box,augFun=AugFun.fun_list):

    images_dict={
        'normal':image,
    }
    if AugFun.HORIZENTAL in augFun:
        images_dict[AugFun.HORIZENTAL]=(image_horizental_flip(image.copy()))

    if AugFun.VERTICAL in augFun:
        images_dict[AugFun.VERTICAL]=(image_vertical_flip(image.copy(),random.randint(0,1)))

    if AugFun.SCALE in augFun:
        box, image_shape, parts_shape = crop_box(box, image.shape, .2)
        # print(image.shape)
        tmp_image = crop_image(image.copy(), image_shape)
        # print(tmp_image.shape , image_shape)
        images_dict[AugFun.SCALE]=tmp_image


    if AugFun.TRANSLATE in augFun:
        images_dict[AugFun.TRANSLATE]=(translate_image(image.copy(),box)[0])

    if AugFun.ROTATE in augFun:
        center = [0, 0]
        angle = random.randint(0, 15)
        theta = angle * math.pi / 180
        condition,box=is_rotatable(box,image.shape,theta,center)
        if condition:
            images_dict[AugFun.ROTATE]=(rotate_image(image.copy(),theta))

    if AugFun.BLUR in augFun:
        images_dict[AugFun.BLUR]=(blurring_image(image.copy(),random.randrange(1,10,2)))

    if AugFun.SHARPEN in augFun:
        images_dict[AugFun.SHARPEN]=(sharpen_image(image.copy()))

    if AugFun.GRAYSCALE in augFun:
        images_dict[AugFun.GRAYSCALE]=(to_gray(image.copy()))


    if AugFun.NOISE in augFun:
        images_dict[AugFun.NOISE]=(random_noise(image.copy()))

    if AugFun.SALT in augFun:
        images_dict[AugFun.SALT]=(salt_noise(image.copy()))

    if AugFun.PEPPER in augFun:
        images_dict[AugFun.PEPPER]=(pepper_noise(image.copy()))

    if AugFun.SALT_PEPPER in augFun:
        images_dict[AugFun.SALT_PEPPER]=(pepper_noise(salt_noise(image.copy())))

    if AugFun.CONTRAST in augFun:
        images_dict[AugFun.CONTRAST]=(image_hist(image.copy()))

    if AugFun.BRIGHTNESS in augFun:
        images_dict[AugFun.BRIGHTNESS]=(brightness(image.copy(),random.randrange(4, 15) / 10))

    if AugFun.INVERT in augFun:
        images_dict[AugFun.INVERT]=(invert_image(image.copy()))

    if AugFun.CHANNEL_MULT in augFun:
        images_dict[AugFun.CHANNEL_MULT]=(channel_mult(image.copy()))

    if AugFun.COARSE_DROPOUT in augFun:
        images_dict[AugFun.COARSE_DROPOUT]=(image_coarse_dropout(image.copy()))

    return images_dict
