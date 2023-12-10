# Semantic Segmentation with U-Net for Architectural Data

This repository contains code semantic segmentation of architectural data! This project focuses on employing a U-Net model to perform precise masking of key elements such as buildings, lanes, roads, vegetation, and water in architectural imagery. The U-Net model is a popular architecture for image segmentation tasks and has been widely used in Robotics domains

## Installation

To use this code, you need to have Python 3.6.10 installed on your system. You can clone this repository using the following command:

```
git clone https://github.com/deepak2233/vertliner.git
```

Next, navigate to the project directory:

```
cd vertliner
```

Install the required dependencies by running the following command:

```
pip install -r requirement.txt
```

---

## Dataset

The code assumes that you have a dataset of images and their corresponding masks. The images should be in JPEG format, and the masks should be in PNG format. The directory structure should be as follows:

```
data/Semantic-segmentation-dataset/
  Semantic segmentation dataset/
    tile1/
      images/
        image001.jpg
        image002.jpg
        ...
      masks/
        image001.png
        image002.png
        ...
    tile2/
      images/
        image001.jpg
        image002.jpg
        ...
      masks/
        image001.png
        image002.png
        ...
    ...
```

You can change the dataset directory by modifying the `dataset_dir` variable in the code.

## Usage

The main script for training the U-Net model is `main.py`. You can run the script with the following command:

```
python3 main.py --dataset_dir data/Semantic-segmentation-dataset/Semantic\ segmentation\ dataset/ --model unet_model1 --epochs 5 --test_size 0.2
```

## To Trace Output Log

In order to trace and save run history use git log command to save the log

```
python3 main.py --dataset_dir data/Semantic-segmentation-dataset/Semantic\ segmentation\ dataset/ --model unet_model1 --epochs 5 --test_size 0.2>output.log
```


The `batch_size` argument specifies the batch size for training, and the `epochs` argument specifies the number of training epochs.

The script will train the model on the provided dataset and save the trained model weights to a file named `model.h5`.


## Command-line Arguments

- `--dataset_dir` (required): Path to the dataset directory.

- `--model`: Choose the UNet model to use. Available options: "unet_model1", "unet_model2". (Default: "unet_model1")

- `--image_height`: Image height. (Default: 256)

- `--image_width`: Image width. (Default: 256)

- `--test_size`: Proportion of the dataset to use as validation. (Default: 0.2)

- `--random_state`: Random state for dataset shuffling. (Default: 42)

- `--batch_size`: Batch size. (Default: 8)

- `--epochs`: Number of epochs. (Default: 50)

- `--learning_rate`: Learning rate. (Default: 0.0001)

- `--data_augmentation`: Apply data augmentation. (Flag, no value required)

- `--class_weighting`: Apply class weighting. (Flag, no value required)

- `--oversampling`: Apply oversampling. (Flag, no value required)

- `--undersampling`: Apply undersampling. (Flag, no value required)

- `--dropout_rate`: Dropout rate. (Default: 0.2)

- `--l1_regularization`: L1 regularization. (Default: 0.01)

- `--l2_regularization`: L2 regularization. (Default: 0.01)


---
After training the model, you can use it for inference on new images. The script `inference.py` provides an example of how to use the trained model for semantic segmentation. You can run the script with the following command:

```
python inference.py --image_path os.getcwd()+'Semantic-segmentation-dataset/Semantic segmentation dataset/tile1/images/    image001.jpg --model os.getcwd()+'unet_model1.h5' --output_path path/to/output.png'
```

---
## Model Architecture

The U-Net model used in this code is defined in the `unet_model.py` file. There are two versions of the model available:

- `unet_model1`: This is the basic U-Net model without any regularization techniques.

- `unet_model2`: This version of the model includes L1 and L2 regularization to reduce overfitting.

You can choose the model version by uncommenting the corresponding line in the `train.py` script.

## Data Augmentation

To increase the variability of the training data and improve the model's generalization ability, data augmentation techniques are applied. The `apply_data_augmentation` function in the code performs data augmentation by randomly applying transformations such as rotation, shifting, zooming, and flipping.

## Class Weighting

If the dataset is imbalanced, it is recommended to apply class weighting to give more importance to underrepresented classes during training. The `apply_class_weighting` function in the code computes the class weights based on the distribution of class labels in the training dataset.

## Oversampling and Undersampling

In cases where the dataset is highly imbalanced, oversampling or undersampling techniques can be used to balance the class distribution. The `apply_oversampling` and `apply_unders

.
