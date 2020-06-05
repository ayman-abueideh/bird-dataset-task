import imageio
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def transpose_image(image):
    tmp_channel = image[:, :, 0]
    image[:, :, 0] = image[:, :, 2]
    image[:, :, 2] = tmp_channel
    return image

image_name='sunrise with large tree.jpg'



image=misc.imread(image_name)
plt.imshow(image)
plt.show()

plt.imshow(transpose_image(image.copy()))
plt.show()
