# Qus: What are some common activation functions used in deep learning and when would you use each?
First of all, Lets discuss what is activation function ->It’s the function that takes as input the activation level of the unit, and returns some function of this activation to be used as the output of the unit

When will we use it and why?
In machine learning, we pass the output of every layer in the model through a non linear "activation" function, before we pass it on to the next layer. The reason we do that is a result of:
1.	Each layer in the model is a linear function by itself (multiplying the output of the previous layer with the weights of the current layer).
2.	If f(x) and g(x) are both linear, f(g(x)) is also linear.

So the combination of (1) and (2) means that even if you create 1000000 layers that feed into each other - their combined composite representation power is that of a single linear function applied on the input values. So we introduce non linearity between the layers, in order to allow the model to represent complex non linear functions of the input values.
1)	Sigmoid -> this activation function ranges from 0 to 1. These functions are commonly use in the output lyres when you have binary problem also, apart from this the update version of this problem is SoftMax that is used in the multiple classification problem 

•	`Sigmoid formula:  f(x) = 1 / (1 + exp(-x))`

SoftMax formula:  If input x = [x1, x2, ..., xn], outputs vector y = [y1, y2, ..., yn] 

•	`Then -> yi = exp(xi) / (exp(x1) + exp(x2) + ... + exp(xn))`

2)	Tanh: In this function, maps the input to a range between -1 and 1. It is given by the formula f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). Tanh functions are commonly used in hidden layers of a neural network, as they introduce non-linearity and help in capturing complex patterns.

3)	ReLU: this function return the output based on the input directly if it is positive, and 0 otherwise. Mathematically, ReLU is defined as `f(x) = max(0, x)`. ReLU is widely used in hidden layers of deep neural networks as it helps in mitigating the vanishing gradient problem and speeds up the training process.

Note: Nowadays, mostly Data Scientist and used ReLu over TanH in the hidden lyres to prevent the vanishing gradient and speed the training process. 

4)	Leaky ReLU: Leaky ReLU functions are a modification of ReLU functions that help mitigate the "dying ReLU" problem. Instead of setting negative values to 0, it allows a small negative value.

•	 `Leaky ReLU formula:  f(x) = max(0.01x, x).  `

---
# Qus: Describe a time when you used data augmentation to improve a model' performance.
Generally, I usually use data augmentation in two cases. 
•	Small amount of training datasets 
•	If image Datasets is biased Eg. If I am going classification (say Cat and Dog) and I saw I have training image Dog which has only one pose along with same pose of cat the training will biased and will give degrade the accuracy of classification 
Also, I have one more example. To improve the performance of a model that was being trained to classify images of cats and dogs. The original dataset only had 100 images of each class, which was not enough data to train a model that could generalize well to new images. I used data augmentation to create 1000 new images of each class by applying a variety of transformations to the original images, such as cropping, flipping, and rotating. This increased the size of the dataset by a factor of 10, which allowed the model to learn more about the different features of cats and dogs. As a result of using data augmentation, the model was able to achieve an accuracy of 95% on a test set of 100 images of cats and dogs.

---
# Qus: Explain how you would go about fine-tuning a pre-trained neural network for a new task.
Layer freezing: It is often beneficial to freeze some of the pre-trained layers, especially the early layers that capture common features such as edges, shapes, and textures. Freezing means that these layers are not changed and their weights are not updated during the fine-tuning process. This helps retain learned representations while allowing later levels to adapt to new tasks.
Training process: Initialize the modified model using the weights of the pre-trained model. Only unfrozen levels and newly added levels are updated during the training process. A dataset prepared for a new task is used to train a model using techniques such as backpropagation and gradient descent. Careful tuning of the learning rate, optimizer, and other hyperparameters is important to ensure successful tuning.
Fine-tuning strategies: Different fine-tuning strategies can be applied depending on the size of the data set and computational resources. A common approach is to fine-tune end-to-end where all layers of the model are trainable. Another approach is to use his two-step process of freezing and training the first layer separately, followed by thawing and training the remaining layers. This stabilizes the training process and helps prevent fatal forgetfulness. 	

---
# Qus: How would you approach diagnosing and addressing overfitting in a deep-learning
model?
To address this overfitting =, I usually take cares model performance.. Overfitting occurs when a model learns the training data too well, and as a result, it is unable to generalize to new data. This can lead to poor performance on the test set, and it can make the model unreliable.
In simpler manner-> training accuracy too high and testing or unseen data testing is not good.
In this assignment, I have also used few steps to address this: 
•	Using a validation set: A validation set is a subset of the training data that is not used to train the model. The model is evaluated on the validation set after each epoch of training. If the model's performance on the validation set starts to decrease, then this is a sign that the model is overfitting.
•	Early stopping: Early stopping is a technique that stops training the model before it has had a chance to overfit the training data. This is done by setting a maximum number of epochs, and then stopping training when the model's performance on the validation set stops improving.
•	Regularization: Regularization is a technique that adds a penalty to the loss function, which helps to prevent the model from learning the training data too well. There are a number of different regularization techniques, including L1 regularization, L2 regularization, and dropout.
By using these techniques, you can diagnose and address overfitting in deep learning models. This will help to improve the accuracy and reliability of your models, and it will make them more likely to generalize to new data.
Here are some additional details about the techniques mentioned above:
Validation set: A validation set is a subset of the training data that is not used to train the model. The model is evaluated on the validation set after each epoch of training. This helps to ensure that the model is not overfitting the training data, and that it is generalizing well to new data.
Early stopping: Early stopping is a technique that stops training the model before it has had a chance to overfit the training data. This is done by setting a maximum number of epochs, and then stopping training when the model's performance on the validation set stops improving. Early stopping can help to prevent overfitting by preventing the model from learning the training data too well.
Regularization: Regularization is a technique that adds a penalty to the loss function, which helps to prevent the model from learning the training data too well. There are a number of different regularization techniques, including L1 regularization, L2 regularization, and dropout.
L1 regularization: L1 regularization adds a penalty to the loss function that is proportional to the absolute value of the weights. This helps to prevent the model from learning weights that are too large, which can help to prevent overfitting.
L2 regularization: L2 regularization adds a penalty to the loss function that is proportional to the square of the weights. This helps to prevent the model from learning weights that are too large, and it can also help to improve the generalization performance of the model.
Dropout: Dropout is a technique that randomly drops out (or sets to zero) a fraction of the neurons in the model during training. This helps to prevent the model from relying on any particular set of neurons, and it can help to prevent overfitting.
