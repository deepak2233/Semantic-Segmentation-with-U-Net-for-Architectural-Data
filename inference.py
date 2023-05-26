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
    
   
    
    image_paths, mask_paths = load_data(args.dataset_dir)
    
    print()
    print("Toatal image size", len(image_paths),len(mask_paths))

    print("######## Split dataset into training and test sets")
    # Split dataset into training and test sets
    image_paths_train, image_paths_test, mask_paths_train, mask_paths_test = train_test_split(
        image_paths, mask_paths, test_size=args.test_size, random_state=args.random_state
    )
    print()
    print("######## Create TensorFlow Datasets")
    # Create TensorFlow Datasets
    train_dataset = create_dataset(image_paths_train, mask_paths_train, args.batch_size)
    test_dataset = create_dataset(image_paths_test, mask_paths_test, args.batch_size)

    # Load the trained model
    
    model = tf.keras.models.load_model(arg.model_dir)

    # Evaluate the model on the test dataset
    loss = model.evaluate(test_dataset)

    # Initialize the confusion matrix
    num_classes = NUM_CLASSES  # Including the "Unlabeled" class
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Iterate over the test dataset and compute IoU
    for image, true_mask in test_dataset:
        # Predict the mask using the trained model
        pred_mask = model.predict(image)
        pred_mask = tf.argmax(pred_mask, axis=-1)

        # Flatten the masks for computing confusion matrix
        true_mask = tf.reshape(true_mask, [-1])
        pred_mask = tf.reshape(pred_mask, [-1])

        # Update the confusion matrix
        confusion += confusion_matrix(true_mask, pred_mask, labels=np.arange(num_classes))

    # Compute the IoU matrix
    intersection = np.diag(confusion)
    union = np.sum(confusion, axis=0) + np.sum(confusion, axis=1) - intersection
    iou = intersection / np.maximum(union, 1)  # Add epsilon to avoid division by zero

    # Print the IoU matrix
    print("IoU matrix:")
    for i in range(num_classes):
        print(f"Class {i}: {iou[i]}")
    

    print("#### Inference on the image")
    # Load image
    img = load_image(args.image_path)

    # Perform inference
    mask = model.predict(img)
    mask_image = convert_mask_to_image(mask[0])

    # Display results
    plt.imshow(mask_image)
    plt.axis('off')
    plt.show()
