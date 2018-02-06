# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./extras/graph.jpg "Visualization"
[image2]: ./extras/grid.jpg "Grayscaling"
[image3]: ./extras/normalization.jpg "Random Noise"
[image4]: ./extras/skew.jpg "Traffic Sign 1"
[image5]: ./extras/skew2.jpg "Traffic Sign 2"

## Rubric Points
###Dataset Exploration

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the good-old python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (unaugmented)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is not uniformly distributed.

![alt text][image1]

To make the project easier to follow, see the following grid:

![alt_text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It has been suggested that the data needs to be represented as real numbers ranging from (-1, 1), I didn't find that particularly useful in this setting, so I simply used the raw pixel values. My guess is that the bias term takes care of that.

That said, I did some preprocessing on the images. The main thing I noticed is how dim some of the images were, so I normalized all of the images' color intensities. For the training set though, I kept the original (non-normalized) images in addition to the normalized ones, since that would simulate presenting my network with different lighting scenarios.
![alt text][image3]

In order to account for different angles at which the picture of the traffic sign might be taken, I generated a number of randomly skewed images with randomly selected exposures and augmented them to the data set. Here are a couple of examples.

![alt text][image4]

![alt text][image5]

Note: The code for the image transformation logic came from this [source](https://github.com/vxy10/ImageAugmentation).

Now the training set got blown up in size to 765578 (10 randomly skewed images for each normalized and non-normalized image).

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x3 |
| Convolution 4x4		| 1x1 stride, same padding, outputs 32x32x32 |
| Max Pooling 			| 2x2 stride, outputs 16x16x32 |
| Convolution 4x4		| 1x1 stride, outputs 16x16x128 |
| Max pooling	      		| 2x2 stride,  outputs 8x8x128 |
| Flatten 				| outputs 8192 |
| Fully connected		| outputs 2048 |   
| Fully connected		| outputs 768  |   
| Fully connected		| outputs 384  |   
| Fully connected		| outputs 43   | 

Note that the activation function used was RELU, and it went between most of the layers above (except the conv 1x1 and the very last fc layer).  

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The networks aims to minimize the cross entropy between the predictions and the labels. With that, I experimented with a number of optimizers, but I ended up sticking with the Adam optimizer. 

I chose a batch size of 256, 10 epochs, and a learning rate of 0.0005.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 88.6% 
* test set accuracy of 88.0

I started with a LeNet model. I expiremented with adding a convolutional layers and fully connected layers with different parameters, changing the activation functions, and introducing dropout at various points in the model. None of those approaches seemed to give me a noticable improvement in accuracy, alas.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found while roaming the streets of Frankfurt on Google maps:
<img src="./extras/traffic/30.jpg" height="200" width="200">
<img src="./extras/traffic/NoEntry.jpg" height="200" width="200">
<img src="./extras/traffic/PriorityRoad.jpg" height="200" width="200">
<img src="./extras/traffic/RightOfWay.jpg" height="200" width="200">
<img src="./extras/traffic/Stop.jpg" height="200" width="200">

I chose these images to be blurry (I took them from a distance), and some of them, have awkward margins (like the no entry sign), and also distortions from being taken from an awkward angle (see the right-of-way sign) (for which the augmentations mentioned earlier helped quite a lot).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		| 30 km/h  		| 
| PriorityRoad     	| Priority Road |
| Road Work			| Road Work  |
| Stop	      			| Stop			|
| Right of Way		| Right of way	|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 60%, as compared to 88% on the test set

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is an 80 km/h speed limit sign (probability of 0.99), but the image is in fact a 30 km/h speed limit.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 80 km/h		| 
| 2e-6     				| 30 km/h		|



Here is the breakdown for the rest of the images. I'll omit extremely tiny probabilities (under 10^-6)

No Entry sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .874         			|  No passing for vehichles over 3.5 tons sign  | 
| .125     				|  No Entry		|
| .0002					| Stop				|
| .000006	      			| Priority Road |

Priority Road sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority Road	| 

Stop sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   	| 

Right-of-way at the next intersection:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way  	| 
