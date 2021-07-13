import cv2
import numpy as np
import os
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL
import tensorflow as tf
import os
import math
import easyocr
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
#from IPython.display import display
import io
import os
from PIL import Image

os.chdir(r'C:\Users\Acer\Desktop')
IMAGE_PATH = r"C:\Users\Acer\Desktop\roi.png"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def preprocess_image(image_path):

    hr_image = tf.image.decode_image(tf.io.read_file(image_path)) #Image is decoded and its datatype and number of channels can be changed

    if hr_image.shape[-1] == 4:  #.REDUCING CHANNELS CAN VARY RESULTS FOR DIFF IMAGES
        #hr_image = hr_image[...,:-1]   #REDUCING CHANNELS CODE IS COMMENTED OUT-------------------------
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4   #convert to tensor
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1]) #limit image to a box
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)

def save_image(image, filename):

    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)   #limit colors min and max value
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())  #cast meaning converting to int format
    image.save("%s.png" % filename)   #CHANGED FILE FORMAT TO PNG-------------------
    print("Saved as %s.png" % filename)

hr_image = preprocess_image(IMAGE_PATH)
model = hub.load(SAVED_MODEL_PATH)

#fake_image = model(hr_image)
#fake_image = tf.squeeze(fake_image)
fake_image = tf.squeeze(hr_image)

save_image(tf.squeeze(fake_image), filename="Super Resolution")