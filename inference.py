import argparse
import os
import json
import io
import numpy as np
from tensorflow.keras import regularizers
from natsort import natsorted
import tensorflow as tf
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

model_dir = os.getcwd()+'unet_model1'

def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 6  # Including the "Unlabeled" class
classes_file = os.path.join(os.getcwd(), 'Semantic-segmentation-dataset/Semantic segmentation dataset/classes.json')


def load_image(path):
    img = load_img(path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def load_class_colors():
    with open(classes_file) as json_file:
        classes_data = json.load(json_file)
        class_colors = {cls['title']: cls['color'] for cls in classes_data['classes']}
    return class_colors
  
def convert_mask_to_image(mask):
    class_colors = load_class_colors()
    mask_indices = np.argmax(mask, axis=-1)
    mask_colors = np.zeros((mask_indices.shape[0], mask_indices.shape[1], 3), dtype=np.uint8)
    for cls, color in class_colors.items():
        class_index = next(
            (i for i, c in enumerate(classes_data['classes']) if c['title'] == cls and c['color'] == color), None)
        if class_index is not None:
            mask_colors[mask_indices == class_index] = mcolors.hex2color(color)
    return mask_colors


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Semantic Segmentation Inference')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('model_dir', type=str, help='Path to the saved model directory')
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_dir)

    # Load image
    img = load_image(args.image_path)

    # Perform inference
    mask = model.predict(img)
    mask_image = convert_mask_to_image(mask[0])

    # Display results
    plt.imshow(mask_image)
    plt.axis('off')
    plt.show()
