# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/data_visualization.png "Visualization"
[image2]: ./writeup_images/signs.png "Grayscaling"
[image3]: ./writeup_images/signs_pred.png "Random Noise"


---
### Writeup / README

#### 1. All files are included in this github repo, the most important of which being the ipython notebook located [here](https://github.com/AaronNBrock/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is a visualization the basic summery of the data set. It's pretty much just a bar graph of the numbers above... I'm not that creative.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocess Data

The preprocessing workflow consisted of two things:
1. Shuffling training data
This is done so that each batch contains a better sub sample of the entire data set.

2. Normalizing data between 0 and 1
This is done to even out the starting points of all the data so the network can more efficiently learn what's important.

(There's not really a fantastic way to visualize these changes)

(I didn't augment the data in anyway cause I'm already a bit behind in the class)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model consisted fo the following hyper parameters:
| Hyper-parameter    | Value            |
|:------------------:|:----------:|
| Epochs             | 20               |
| batch\_size           | 256        |
| Learning Rate         | 0.001      |
| Max pooling 2x2          |    0.75       |

And the following layers:

| Layer                       |     Description                                                 | Output shape |
|:---------------------:|:---------------------------------------------:|:------------:|
| Input                       | 32x32x3 RGB image                                                  | 32x32x3      |
| Convolution 5x5          | 5x5 filter, 1x1 stride, Valid padding            | 28x28x6      |
| Relu                     | ReLU Activation Function                         |              |
| Max pooling 2x2             |    2x2 filter, 2x2 stride, Valid padding            | 14x14x6      |
| Dropout                     |    Dropout                                       |              |
| Convolution 5x5             | 5x5 filter, 1x1 stride, Valid padding            | 10x10x16     |
| Relu                     | ReLU Activation Function                         |              |
| Max pooling 2x2             |    2x2 filter, 2x2 stride, Valid padding            | 5x5x16       |
| Dropout                     |    Dropout                                       |              |
| Flatten                     |    Flatten                                       | 400          |
| Fully Connected          |    400 to 120                                       | 120          |
| Relu                     | ReLU Activation Function                         |              |
| Dropout                     |    Dropout                                       |              |
| Fully Connected             |    120 to 84                                        | 84           |
| Relu                     | ReLU Activation Function                         |              |
| Dropout                     |    Dropout                                       |              |
| Fully Connected             |    84 to n\_classes (43)                            | n\_classes   |
| Softmax                           | Softmax Activation Function                                  |              |



#### 3. Training

For training I used an AdamOptimizer over Stochastic Gradient Descent, since it has better performance in certain situations.  Besides that training was pretty strait forward, I iterated over the data set ``epochs`` in ``batch_size`` segments and ran the Optimizer for each batch.

#### 4. Approach's used and difficulties encountered.

Originally my biggest issue was that of over fitting, to combat this I added dropout between each layer, this lowered over fitting immensely.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.948
* test set accuracy of 0.942
 

### Test a Model on New Images

#### 1. Choose German traffic signs found on the web.

Here are five German traffic signs that I found on the web:

![alt text][image2]

My guess was that the speed limit signs would be the hardest to classify due to traffic signs in general having similar colors & shapes.

#### 2. Output

Here are the results of the prediction:

![alt text][image3]

As you can see the two misclassified images were indeed the speed limits.  However, interestingly when I was testing this it seemed to think that the ``50km/h`` sign was a ``Roundabout mandatory``, however during the file training (and the one shown in the image above) it instead opted for a ``30km/h`` which makes more sense.  The accuracy being 5/7 or 0.714, it did indeed do worse then the testing data, but my guess is that this is just due to such a small sample.

Regarding the networks certainty, it was almost 100% certain on each image when I ran it through the softmax function (I'm not sure if this is a good or bad thing).  Anyway, I instead used the logits normalized linearly rather than via an exponential to make the certainty easier to understand and there you could see that the images it had gotten wrong (the speed limit signs) it was most uncertain about.

(I tried to get a nice visualization of the logits normalized, but it was taking too long so I scraped it)

