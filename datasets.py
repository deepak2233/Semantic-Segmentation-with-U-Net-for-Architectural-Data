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

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 6  # Including the "Unlabeled" class
classes_file = os.path.join(os.getcwd(), 'Semantic-segmentation-dataset/Semantic segmentation dataset/classes.json')
def load_data(dataset_dir):
    image_paths = []
    mask_paths = []
    tiles = natsorted(os.listdir(dataset_dir))
    for tile in tiles[:-1]:
        tile_folder = os.path.join(dataset_dir, tile)
        image_folder = os.path.join(tile_folder, "images")
        mask_folder = os.path.join(tile_folder, "masks")
        for filename in natsorted(os.listdir(image_folder)):
            if filename.endswith(".jpg"):
                image_files = os.path.join(image_folder, filename)
                mask_files = os.path.join(mask_folder, filename.replace(".jpg", ".png"))
                image_paths.append(image_files)
                mask_paths.append(mask_files)
                
    return image_paths,mask_paths

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_mask(path):
    with open(classes_file) as json_file:
        classes_data = json.load(json_file)
        class_colors = {cls['title']: cls['color'] for cls in classes_data['classes']}
    
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, IMAGE_SIZE)
    
    # Convert mask RGB values to class indices
    mask_indices = tf.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=tf.int32)
    for cls, color in class_colors.items():
        class_index = next((i for i, c in enumerate(classes_data['classes']) if c['title'] == cls and c['color'] == color), None)
        if class_index is not None:
            color = tf.constant(mcolors.hex2color(color), dtype=tf.float32)  # Convert color to float32
            class_mask = tf.reduce_all(tf.equal(mask, color), axis=-1)
            mask_indices = tf.where(class_mask, class_index, mask_indices)
    
    # Convert class indices to one-hot encoded mask
    mask = tf.one_hot(mask_indices, depth=NUM_CLASSES)
    
    return mask

def create_dataset(image_paths, mask_paths,batch_size):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.batch(batch_size)
    image_dataset = image_dataset.prefetch(tf.data.AUTOTUNE)
    
    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_paths)
    mask_dataset = mask_dataset.map(load_mask, num_parallel_calls=tf.data.AUTOTUNE)
    mask_dataset = mask_dataset.batch(batch_size)
    mask_dataset = mask_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf.data.Dataset.zip((image_dataset, mask_dataset))



def apply_data_augmentation(images, masks):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    augmented_images = []
    augmented_masks = []

    for image, mask in zip(images, masks):
        image_batch = np.expand_dims(image, axis=0)
        mask_batch = np.expand_dims(mask, axis=0)

        for img_aug, mask_aug in zip(datagen.flow(image_batch, batch_size=1),
                                      datagen.flow(mask_batch, batch_size=1)):
            augmented_images.append(img_aug[0])
            augmented_masks.append(mask_aug[0])

    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)

    return augmented_images, augmented_masks

def apply_class_weighting(masks):
    class_weights = class_weight.compute_class_weight("balanced", np.unique(masks.flatten()), masks.flatten())
    return class_weights

def apply_oversampling(images, masks):
    reshaped_images = images.reshape(images.shape[0], -1)
    reshaped_masks = masks.flatten()

    oversampler = RandomOverSampler()
    oversampled_images, oversampled_masks = oversampler.fit_resample(reshaped_images, reshaped_masks)

    oversampled_images = oversampled_images.reshape(oversampled_images.shape[0], images.shape[1], images.shape[2], images.shape[3])
    oversampled_masks = oversampled_masks.reshape(-1, masks.shape[1], masks.shape[2])

    return oversampled_images, oversampled_masks

def apply_undersampling(images, masks):
    reshaped_images = images.reshape(images.shape[0], -1)
    reshaped_masks = masks.flatten()

    undersampler = RandomUnderSampler()
    undersampled_images, undersampled_masks = undersampler.fit_resample(reshaped_images, reshaped_masks)

    undersampled_images = undersampled_images.reshape(undersampled_images.shape[0], images.shape[1], images.shape[2], images.shape[3])
    undersampled_masks = undersampled_masks.reshape(-1, masks.shape[1], masks.shape[2])

    return undersampled_images, undersampled_masks

