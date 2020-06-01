# **Behavioral Cloning** 

## Writeup - Thomas Smith

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture.png "Netwok Archicture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is heavily based on the NVIDIA model described [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The model is defined in model.py lines 53-88.

The network architecture is represented by the image below, taken from the NVIDIA paper referenced above.

![Network architecture][image1]

The final architecture consists of: a normalization layer; 3 convolutional layers with 5x5 kernel size and 2x2 stride; 2 convolutional layers with 3x3 kernel size and 1x1 stride; and 3 fully connected layers.

Padding is 'valid' where applicable throughout.

The input data is 66x220 pixel 3-channel RGB images (as opposed to YUV images as used in the NVIDIA paper) which go through a Keras lambda layer where it is normalized to a zero mean distribution between -0.5 and 0.5 using:

$$ x_{norm} = \frac{x}{255} - 0.5 $$

Each of the convolutional and fully connected layers, except for the final fully connected layer, have RELU activation functions to introduce non-linearity to the model.

Other network architectures were assessed but the NVIDIA model was found to perform best at driving in autonomous mode, although other more complex architecures did have lower training and validation losses. The other networks investigated were: LeNet, VGG16 and Inception Resnet. VGG16 and Inception Resnet were imported with weights trained on the ImageNet database, with the final layer removed and replaced by a global average pooling layer. A fully connected layer with one output was then added and the weights of this layer were trained using the Udacity provided dataset. These models can also be found in the repository.

#### 2. Attempts to reduce overfitting in the model

The model includes dropout layers after all convolutional and fully connected layers except for the final fully connected layer. A dropout layer is also used before the first convolutional layer to reduce overfitting.

The dataset was also augmented to reduce overfitting. The augmentation technique applied is to flip all images left to right, simulating driving around the circuit in the opposite direction. Steering angles for flipped images are mutliplied by a factor of -1 to reverse the direction. This technique is applied to all images in the dataset, including the left, center and right camera angles.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The model was optimised against a mean square error loss function (model.py line 88). The model was trained for 2 epochs as trials showed that the validation loss started to fluctuate if more than 2 epochs were used, indicating overfitting.

#### 4. Appropriate training data

Due to hardware constraints only used the Udacity provided dataset was used for this project. The training data proivded includes three camera views for each frame: left, center and right of vehicle.  The left and right angles are used to improve the generality of the model and improve its ability to return to the center of the lane. For these images, the steering angle was adjusted from the center angle by +/- 0.35 for the left and right images respectively.

The training data available was split into training and validation datasets with a 80:20 split (model.py line 21) to allow training performance to be better assessed. After augmenting the data there were to many training images to load into memory at once and so a Python generator was used to only load and augment images as they were required by the training (model.py lines 23-47).

The input images were cropped to remove areas of the image that don't contain information relevant to steering angles (model.py line 60) and resized to the NVIDIA network input size of 66x200.

#### 5. Outcome

The model was trained as outlined above and tested by driving the vehicle in autonomous mode in the simulator. The results can be seen in two videos: run1.mp4 for track 1 and run2.mp4 for track 2.

On track 1 the model performed well and the car remains on track at all times and, subjectively, drives with a smooth and controlled style.

On track 2 the model struggles more and leaves the road after a few turns. The model is unable to recover from this point. 

#### 6. Future work and improvements

The performance of the trained model on track 2 was unsatisfactory. To improve the performance more data could be collected, including from track 2, and additional data augmentation techniques to employed to increase the size of the dataset. This is expected to lead to better results with more complex network architectures, which are prone to overfitting with the current dataset.
