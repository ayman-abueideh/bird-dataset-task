from task4.task4_draw_augmented_images import *
from task3.agumentation_functions import read_image


image_path= '../Bird_image_dataset/images/016.Painted_Bunting/Painted_Bunting_0016_15200.jpg'

image=read_image(image_path)
box=[205.0,84.0,289.0,153.0]

dict=get_augmentation_images(image,box)
print(len(dict))
draw_figure(dict,"test_figure.png")
