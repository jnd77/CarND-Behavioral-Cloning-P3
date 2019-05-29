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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
