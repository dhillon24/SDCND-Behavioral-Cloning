
## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/figure_1.png "Unbalanced Histogram"
[image2]: ./examples/figure_3.png "Balanced Histogram"
[image3]: ./examples/nvidia-image.png "Model Visualization"
[image4]: ./examples/left.jpg "Left Image"
[image5]: ./examples/center.jpg "Center Image"
[image6]: ./examples/right.jpg "Right Image"
[image7]: ./examples/recovery-1.jpg "Recovery Image 1"
[image8]: ./examples/recovery-2.jpg "Recovery Image 2"
[image9]: ./examples/recovery-3.jpg "Recovery Image 3"
[image10]: ./examples/image.jpg "Original Image"
[image11]: ./examples/flipped-image.jpg "Flipped Image"
[image12]: ./examples/dark-image.jpg "Darkened Image"
[image13]: ./examples/cropped-image.jpg "Cropped Image"
[image14]: ./examples/figure_4.png "Loss Function"

## Pipeline
---
**Files Submitted & Code Quality**

*1. Submission includes all required files and can be used to run the simulator in autonomous mode*

My project includes the following files:
* model.py containing the script to create and train the model for track 1 (lake track).
* drive.py for driving the car in autonomous mode for track 1 and track 3 (mountain track from old simulator).
* model.h5 containing a trained convolution neural network for track 1 and track 3.
* model-throttle.py containing the script to create and train the model for track 2 (jungle track).
* drive-throttle.py for driving the car in autonomous mode for track 2.
* model-throttle.h5 containing a trained convolution neural network for track 2.
* writeup_report.ipynb summarizing the results
* supporting graphics and linked video files


*2. Submission includes functional code*

Using the Udacity provided simulator (version1 or version2) and drive.py or drive-throttle.py file, the car can be driven autonomously around the specified tracks by executing 
```sh
python drive.py model.h5
```
           or
```sh
python drive-throttle.py model-throttle.h5
```

*3. Submission code is usable and readable*

The model.py and model-throttle.py files contains the code for training and saving the convolution neural networks. The files shows the pipelines I used for training and validating the models, and it contains comments to explain how the code works.

**Model Architecture and Training Strategy**

*1. An appropriate model architecture has been employed*

My model for the lake and mountain tracks consists of a convolution neural network based on the Nvidia's end-to-end steering architecture consisting of 5 convolutional layers with 5x5 and 3x3 filter sizes and depths between 24 and 64. The Nvidia model was chosen for its lower memory footprint than models with a higher number of parameters like VGG and its benchmarked performance in the given task of end-to-end steering angle prediction.   

The model includes RELU layers to introduce nonlinearity and the data is normalized (between -0.5 and 0.5) in the model using a Keras lambda layer. The input is cropped as well to weed out the effect of surroundings on steering angle prediction. This cropping is also implemented through a Keras layer for higher performance.

*2. Attempts to reduce overfitting in the model*

The model contains dropout layers in order to reduce overfitting. This is implemented after the the convolutional layers' output is flattened before the fully connected layers. Introducing more than one dropout layers or batch normalization did not improve results markedly and in fact lead to higher training loss. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Early stopping was implemented as well and the number of epochs were kept low. 

*3. Model parameter tuning*

Mean squared error was chosen as the loss function. The model used an adam optimizer, so the learning rate was not tuned manually. Different optimizers including 'rmsprop' and 'sgd' were also tried but 'adam' yielded faster convergence and lower training and validation loss.

*4. Appropriate training data*

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the sides of the road and left/right camera images with steering angle offsets. The training data was balanced as well to combat bias for center driving on the lake track.  


**Model Architecture and Training Strategy**

*1. Solution Design Approach*

An end-to-end learning system which has driving behavior as input in the form of images and outputs a steering angle prediction was to be designed. My first solution approach was to design a model based around the tried and tested Nvidia architecture which was developed for this task. The Nvidia model was successful in driving the car in straight lanes when trained on unbalanced training data but failed to negotiate curves very well. I realized that the problem may be that the network is biased to drive straight due to the nature of track 1. The histogram of the training data confirmed my hypothesis:

![alt text][image1]

I eventually realized that the network doesn't need more data but richer samples. So I balanced the histogram to a certain degree so that samples associated with high steering angles have more influence on the model and simultaneously the bias for low steering angles is curbed. The balanced histogram of the training data (obtained after data augmentation as well) is shown below. This balancing increased the sensitivity of the model to curves at the cost of slight oscillations about mean centre position. But nevertheless it was crucial for generalization of the model.

![alt text][image2]

Additionally, I experimented with dropout so that the network doesn't rely on all features it has detected to generate a steering output. However, introducing more than one dropout layers gave slower convergence and decreased the sensitivity to curves. Batch normalization was tried too, but somehow its inclusion saturated the decrease of the training loss. I also increased the receptive field of the convolutional layers using 3x3 filters across all layers but no discernible benefits were observed. 

The realization that 'Data is king' dawned upon me and rather than making changes to the architecture, I refocused my efforts to forming a richer training set which tries to map a majority of the road positions that the car may be found driving in so that suitable recovery behaviors could be learned. This included flipping the images and the steering angles, using left and right camera images with appropriate angle offsets and recording recovery behaviors from the sides of the road. These techniques are discussed in detail in later sections. 

Data augmentation and balancing improved driving performance greatly as now the car was able to negotiate the lake track track flawlessly but the model did not generalize to the mountain track from the old simulator which had a lot of shadows which proved troublesome due to their absence in the training data. So the data was further augmented by decreasing the brightness of images to imitate shadows. Now the performance on the test mountain track was decent with a slight room for improvement. So the model generalized to a new track with shadowed, curvier and sloped roads while only being trained on data from the relatively simpler lake track.

Then I tested my model on the the jungle track which is included in the new simulator and it failed miserably. This was expected as this is a very challenging track and even I failed to negotiate it manually the first time. The problem was that it had extremely sharp curves interspersed with steep descents and ascents. Braking was essential for negotiating the track manually itself so a model based on almost constant throttle did not perform very well. The second phase of the project was to incorporate throttle control into the model so that it outputs not only steering angle but also throttle value predictions. This was a harder problem as throttle values had to be judiciously be incorporated while recording the training data. Moreover while regressing on two outputs other problems popped up like the different rate of convergence of steering and throttle values - training the model for a greater number of epochs decreased the error in throttle values at the cost of overfitting steering values. 

Since now there were two outputs, I doubled the number of nodes in all fully connected layers to map higher level correlations between features, throttle and steering values for a particular image frame. Once the model was trained appropriately, the steering and throttle values complemented each other just like in real driving by a human. For instance, before a sharp curve or steep descent there was momentary braking which prevented the car from veering off-course and increased the stability of the drive. Given that human drivers also give only two main control inputs to the car - steering and throttle/brake, the model was successful in mimicking human driving behavior to a great extent.  


*2. Final Model Architecture*

The final model architectures for track 1 (lake track) consisted of a convolution neural networks with the following layers and layer sizes:

**Nvidia Model Architecture**

**Input**: 160x320x3 BGR image

**Preprocessing Layer: Cropping**: Horizontal Crops - (70, 25), Vertical Crops - (0,0)

**Preprocessing Layer: Normalization**: Normalization Limits - (-0.5,0.5)

**Layer 1: Convolutional 1**: Filter Size - 5x5, Number of Filters - 24, Stride - (2,2)

**Activation**: Relu 

**Layer 2: Convolutional 2**: Filter Size - 5x5, Number of Filters - 36, Stride - (2,2)

**Activation**: Relu

**Layer 3: Convolutional 3**: Filter Size - 5x5, Number of Filters - 48, Stride - (2,2)

**Activation**: Relu 

**Layer 4: Convolutional 4**: Filter Size - 3x3, Number of Filters - 64, Stride - (2,2)

**Activation**: Relu

**Layer 5: Convolutional 5**: Filter Size - 3x3, Number of Filters - 64, Stride - (2,2)

**Activation**: Relu

**Flatten**

**Dropout**: Keep Probability = 0.5

**Layer 6: Fully Connected 1**:Output - 100

**Activation**: Relu

**Layer 7: Fully Connected 2**: Output - 50

**Activation**: Relu

**Layer 8: Fully Connected 3**: Output - 10

**Activation**: Relu

**Layer 8: Fully Connected 4**: Output - 1

**Output**: 1 steering value


The final model architectures for track 2 (jungle track) consisted of a convolution neural networks with the following layers and layer sizes. Note that now the number of outputs are increased to two and all fully connected layers contain twice the number of nodes as before. Also the level of vertical cropping has also been increased while horizontal cropping has been decreased.

**Throttle Model Architecture**

**Input**: 160x320x3 BGR image

**Preprocessing Layer: Cropping**: Horizontal Crops - (60, 25), Vertical Crops - (20,20)

**Preprocessing Layer: Normalization**: Normalization Limits - (-0.5,0.5)

**Layer 1: Convolutional 1**: Filter Size - 5x5, Number of Filters - 24, Stride - (2,2)

**Activation**: Relu 

**Layer 2: Convolutional 2**: Filter Size - 5x5, Number of Filters - 36, Stride - (2,2)

**Activation**: Relu

**Layer 3: Convolutional 3**: Filter Size - 5x5, Number of Filters - 48, Stride - (2,2)

**Activation**: Relu 

**Layer 4: Convolutional 4**: Filter Size - 3x3, Number of Filters - 64, Stride - (2,2)

**Activation**: Relu

**Layer 5: Convolutional 5**: Filter Size - 3x3, Number of Filters - 64, Stride - (2,2)

**Activation**: Relu

**Flatten**

**Dropout**: Keep Probability = 0.5

**Layer 6: Fully Connected 1**:Output - 200

**Activation**: Relu

**Layer 7: Fully Connected 2**: Output - 100

**Activation**: Relu

**Layer 8: Fully Connected 3**: Output - 20

**Activation**: Relu

**Layer 8: Fully Connected 4**: Output - 2

**Output**: 1 steering value, 1 throttle value

A visualization of the original Nvidia model has been provided. Note that the actual models used consisted of slight alterations to the original model as have been described above.

![alt text][image3]

*3. Creation of the Training Set & Training Process*

To capture good driving behavior, I first recorded one lap on track 1 using center lane driving. Images from the left, right and center cameras were used. The steering measurements of the left and right images were offset by a value of 0.1 or 0.2 so that the car moves towards the centre of the road. Here is an example of left, center and right camera images and their corresponding steering measurements for an offset of 0.2:

![alt text][image4] <div align="center">-0.03</div>
![alt text][image5] <div align="center">-0.23</div>
![alt text][image6] <div align="center">-0.43</div>

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that recovery behaviors could be learned. The following images show the center camera images of start, middle and stop of a recovery behavior from the right edge of the road.

![alt text][image7] <div align="center">-0.56</div>
![alt text][image8] <div align="center">-0.27</div>
![alt text][image9] <div align="center">0.00</div>

To augment the data sat, I also flipped images and angles as this seemingly offsets the bias for left turn driving for track 1 which has a majority of left turns. Images were darkened to simulate shadows which were rare in track 1 but are prominent in track 2. Cropping and normalization were also done as part of keras layers. All these operations are visualized below.

![alt text][image10] <div align="center">Original Image</div>
![alt text][image11] <div align="center">Flipped Image</div>
![alt text][image12] <div align="center">Darkened Image</div>
![alt text][image13] <div align="center">Cropped Image</div>

Finally after generating all these images I balanced the histogram as shown previously. The balancing scheme was such that if for each 0.1 interval between -1.0 and 1.0 (the steering range), 1500 samples were selected randomly from original and augmented samples. If an interval contained less than 1500 samples then all samples in that interval were selected. This lead to a formation of a representative dataset which was crucial for the driving performance achieved.  

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as poorer steering was observed when the number of epochs were increased beyond this. I used an adam optimizer so that manually training the learning rate wasn't necessary. Once the training and validation datasets were formed, a python generator was used to yield batches of training or validation data to the model. The model was trained multiple times due to random sampling and the best performing model was selected as the final model. In each run the model was trained on about only 10,000 samples but still the dataset captured a rich diversity of driving behavior as represented by the balanced histogram. The loss is visualized for one of the runs below:

![alt text][image14]

The solution for the second phase of the project i.e. devising a model for track 2 (jungle track) proceeded similarly except that now capturing recovery data for the most difficult parts of the track became even more important. Data augmentation by darkening images was not necessary as the track had plenty of shadows. The level of horizontal cropping was reduced to capture uphill or downhill climbs more efficiently and vertical cropping was increased to reduce the effect of extraneous landscape features on the drive. The number of epochs were increased to 10 to learn light braking checkpoints better and the model was trained on about 16,000 samples. Throttle control granted more stability to the car and better steering as well as the car's speed was always reasonable around curves, descents and climbs.


## Results

Please click on the images to see youtube videos of final driving results.

[![Alt text](https://img.youtube.com/vi/MCxvOjnUyCE/0.jpg)](https://www.youtube.com/watch?v=MCxvOjnUyCE)

[![Alt text](https://img.youtube.com/vi/aZBikAzYqvw/0.jpg)](https://www.youtube.com/watch?v=aZBikAzYqvw)

[![Alt text](https://img.youtube.com/vi/6yZS8GH3NDw/0.jpg)](https://www.youtube.com/watch?v=6yZS8GH3NDw)
