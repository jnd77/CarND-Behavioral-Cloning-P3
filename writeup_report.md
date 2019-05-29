# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidianet]: ./examples/nvidia-architecture.png "Model Visualization"
[training]: ./examples/training-curves.png "Training"
[original]: ./examples/original.png "Original training image"
[flipped]: ./examples/flipped.png "Flipped training image"

---
### Files Submitted & Code Quality

#### Submitted files

My project includes the following files:
* input_generator.py containing the methods to load the data in the form of a Python generator
* network.py containing the Keras model architecture
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4

#### Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven 
autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### Code is usable and readable

The input_generator.py file contains the code necessary to load the data, 
and prepare 2 generators, one for training, one for validation.

The network.py file builds the Keras model, with all the required layers.

The model.py file contains the code for training and saving the convolution neural network. 
It shows the pipeline I used for training and validating the model, 
and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### Model architecture

My model is based on the famous [Nvidia model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), 
which performs well for end-to-end models.

The model also includes the pre-processing layers to crop the input image, and also to normalize it. 

#### Reduce overfitting

To reduce overfitting, I use some augmentation techniques, like horizontal flipping (with a 50% chance).
I also make use of the side cameras.


#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


#### Appropriate training data

This was the most complex issue in this project.
The data need to be well balanced between good runs around the track in both directions.
It also requires some recovery segments, to ensure the vehicle returns to the center when 
drifting to the side.
After trying to gather my own data, and I'm not a very good driver, I decided to use the sample data
provided by Udacity, which were good enough to complete the first track.  

### Model Architecture and Training Strategy

#### Solution Design Approach

At first, I tried to use an inception model (as can be seen in the network.py file), followed by a 
series of fully connected layers.
I thought such a network would be good at recognizing the track features.
However, it never really performed well, even in straight lines.

After some internet search, I came across the Nvidia model, and decided to give it a try.
After the first training, BINGO, the vehicle was able to drive autonomously around the track 
without leaving the road.

#### Final Model Architecture

The final model architecture (network.py, with the nvidia architecture) consisted of the following:

```
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

It starts with pre-processing: first the Cropping2D layer to remove the sky and the hood, 
then the Lambda layer to normalize the data.

Or the original image from Nvidia:

![Nvidia network][nvidianet]



#### Creation of the Training Set

I tried to record my own data, with a couple of laps counter-clockwise, then one lap clockwise.
I also added some recovery data (when car is near the edge and returns to the center).

In the end, I used the Udacity sample data.

20% of the data were selected to be part of the validation set.
80% were kept for the training, amounting to 6,428 records (each record containing 3 images, one per camera).

During the construction of the generator, I randomly shuffle the training set.
I randomly select one of the cameras (left, right or center).
In case of the left or right cameras being selected, I apply a small correction of 0.2 to the angle.
In case of the center camera being selected, I flipped the image horizontally 50% of the time.

Here is an image before being flipped:
![Original][original]

And after being flipped:
![Flipped][flipped]

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was around 8 as evidenced by the following graph, where the validation error
starts incrasing after 8 epochs.

![Training Curves][training] 

I used an adam optimizer so that manually training the learning rate wasn't necessary.
