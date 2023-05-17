# Semantic Segmentation

This repository contains code for performing semantic segmentation using a U-Net model. The U-Net model is a popular architecture for image segmentation tasks and has been widely used in various domains, including medical imaging and remote sensing.

## Installation

To use this code, you need to have Python 3.x installed on your system. You can clone this repository using the following command:

```
git clone https://github.com/your-username/semantic-segmentation.git
```

Next, navigate to the project directory:

```
cd semantic-segmentation
```

Install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

argparse==1.4.0
natsort==7.1.1
numpy==1.21.0
tensorflow==2.6.0
scikit-learn==0.24.2
imbalanced-learn==0.8.1
matplotlib==3.4.2


## Dataset

The code assumes that you have a dataset of images and their corresponding masks. The images should be in JPEG format, and the masks should be in PNG format. The directory structure should be as follows:

```
Semantic-segmentation-dataset/
  Semantic segmentation dataset/
    tile1/
      images/
        image1.jpg
        image2.jpg
        ...
      masks/
        image1.png
        image2.png
        ...
    tile2/
      images/
        image1.jpg
        image2.jpg
        ...
      masks/
        image1.png
        image2.png
        ...
    ...
```

You can change the dataset directory by modifying the `dataset_dir` variable in the code.

## Usage

The main script for training the U-Net model is `train.py`. You can run the script with the following command:

```
python train.py --batch_size 32 --epochs 50
```

The `batch_size` argument specifies the batch size for training, and the `epochs` argument specifies the number of training epochs.

The script will train the model on the provided dataset and save the trained model weights to a file named `model.h5`.

## Results

After training the model, you can use it for inference on new images. The script `infer.py` provides an example of how to use the trained model for semantic segmentation. You can run the script with the following command:

```
python infer.py --image_path path/to/image.jpg --model_path path/to/model.h5 --output_path path/to/output.png
```

The `image_path` argument specifies the path to the input image, the `model_path` argument specifies the path to the trained model weights file, and the `output_path` argument specifies the path to save the output segmentation mask.

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
