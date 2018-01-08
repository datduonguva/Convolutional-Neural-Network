# Convolutional Neural Network
This is an simplified implementation of Lenet Convolutional Neural Network for recognizing color images from CIFAR-10 data. The implementation does not use any ML/AI package.

## Package requirements
* Numpy: for matrix handling
* Matplotlib: for plotting the fluctuation of loss over each training epoch

## Design

* Input images are of size 32x32x3
* Two convolutional layers
* Two max pooling layers
* Fully connected layers has one hidden layers
* Data being used to train the model are CIFAR-10 data.

## Performance
For the limitted computational ability of the deviced being used (Intel Core i5-3427 U 1.8GHz, 4GB RAM), the network has been chosed to have:
* 6 filters for the 1st convolutional layer
* 16 filters for 2nd convolutional layer
* 100 hidden nodes
* 10 out put nodes

For the purpose of demonstration, only 1000 images were used to train. The running time for each epoch is about 150 (s). The plot below shows the variation of loss function after each epoch:

![alt text](https://github.com/datduonguva/Convolutional-Neural-Network/blob/master/loss_vs_epoch.png)

## Future implementation
For the future, I will find parts that can be computed in parallel to improve the running time.
