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

# Constants
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 6  # Including the "Unlabeled" class
classes_file = os.path.join(os.getcwd(), 'data','Semantic-segmentation-dataset','Semantic segmentation dataset','classes.json')
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


def unet_model1(input_shape, num_classes, dropout_rate):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def unet_model2(input_shape, num_classes, dropout_rate, l1_regularization, l2_regularization):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model



def train_model(model, train_dataset, epochs):
    for layer in model.layers:
        print(layer.name, layer)
    
    model.fit(train_dataset, epochs=epochs)
    
    # # Plot the training and validation loss
    # loss = history.history["loss"]
    # val_loss = history.history["val_loss"]
    # epochs = range(1, len(loss) + 1)

    # plt.figure()
    # plt.plot(epochs, loss, "bo", label="Training loss")
    # plt.plot(epochs, val_loss, "b", label="Validation loss")
    # plt.title("Training and validation loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    

save_dir =  os.path.join(os.getcwd()+'unet_model1.h5')
   
def save_model(model, save_dir):
    model.save(save_dir)
    print(f"Model saved to {save_dir}")
    
    
def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    return loss, accuracy




def visualize_predictions(model, test_dataset, num_samples):
    # Select random samples from the test dataset
    samples = test_dataset.take(num_samples)
    
    # Generate predictions for the selected samples
    predictions = model.predict(samples)
    
    # Plot the original images and their corresponding predictions
    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 8))
    for i, (image, mask) in enumerate(samples):
        # Print the shape of the image before reshaping
        print("Image shape before reshaping:", image.shape)
        
        # Ensure the image has the correct shape
        if image.shape != (256, 256, 3):
            # Skip this image and continue to the next one
            print("Skipping image with incorrect shape:", image.shape)
            continue
        
        # Reshape the image to (256, 256, 3)
        image = np.reshape(image, (256, 256, 3))
        
        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original')
        
        # Plot predicted mask
        predicted_mask = tf.argmax(predictions[i], axis=-1)
        axes[i, 1].imshow(predicted_mask, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Predicted Mask')
    
    plt.tight_layout()
    plt.show()






def main(args):
    # Load dataset and masks
    print("######## Loading the Datasets")
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

    #print("Toatal Train-Test image size", len(train_dataset),len(test_dataset))
    
    print()
    
    # Apply data augmentation if enabled
    if args.data_augmentation:
        print("######## Apply data augmentation")
        train_dataset = apply_data_augmentation(train_dataset)
        print("Toatal train data augmentation image size", len(train_dataset))
    
    print()
       # Apply class weighting if enabled

    if args.class_weighting:
        print("######## Apply class weighting") 
        train_dataset = apply_class_weighting(train_dataset)
        print("Toatal train class weighting image size", len(train_dataset))
        

    print()
    # Apply oversampling if enabled
    if args.oversampling:
        print("######## Apply oversampling")
        train_dataset = apply_oversampling(train_dataset)

    print()
    # Apply undersampling if enabled
    if args.undersampling:
        print("######## Apply undersampling")
        train_dataset = apply_undersampling(train_dataset)

    # Create and compile the model
    
    print()
    if args.model == "unet_model1":
        print("######## compile the model")
        model = unet_model1((args.image_height, args.image_width, 3), NUM_CLASSES, args.dropout_rate)
    elif args.model == "unet_model2":
        print("######## compile the model")
        model = unet_model2((args.image_height, args.image_width, 3), NUM_CLASSES, args.dropout_rate,
                       args.l1_regularization, args.l2_regularization)
    else:
        raise ValueError("Invalid model choice.")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    print()
    print("######## Train the model")
     # # Train the model
    train_model(model, train_dataset, epochs=args.epochs)
    print()
    save_dir =  os.path.join(os.getcwd()+'/unet_model1.h5')
    save_model(model, save_dir)
    
    print("######## Evaluate the model")
    # Evaluate the model
    loss, accuracy = evaluate_model(model, test_dataset)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    print("######## Visualize predictions")
    # Visualize predictions
    visualize_predictions(model, test_dataset, num_samples = 2)


if __name__ == "__main__":
    # Parser configuration
    parser = argparse.ArgumentParser(description="Train U-Net model for semantic segmentation")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory", required=True)
    parser.add_argument("--model", type=str, choices=["unet_model1", "unet_model2"], default="unet_model1", help="Choose the UNet model to use")
    parser.add_argument("--image_height", type=int, default=256, help="Image height")
    parser.add_argument("--image_width", type=int, default=256, help="Image width")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to use as validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for dataset shuffling")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--data_augmentation", action="store_true", help="Apply data augmentation")
    parser.add_argument("--class_weighting", action="store_true", help="Apply class weighting")
    parser.add_argument("--oversampling", action="store_true", help="Apply oversampling")
    parser.add_argument("--undersampling", action="store_true", help="Apply undersampling")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--l1_regularization", type=float, default=0.01, help="L1 regularization")
    parser.add_argument("--l2_regularization", type=float, default=0.01, help="L2 regularization")
    args = parser.parse_args()

    main(args)
