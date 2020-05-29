# Music Millenium Classification
![](https://github.com/saihiel/Music_Year_Classification/blob/master/million_song_dataset.jpg)  
I will build Logistic Regression and K - Nearest Neighbour models to predict which century a piece of music was released. I am using the "YearPredictionMSD Data Set" based on the Million Song Dataset. The data is available to download from the UCI Machine Learning Repository:

* https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd

*As this is not as challenging a dataset/project I will make it more interesting by implementing my Logistic Regression models using only numpy and train it using stochastic gradient descent from scratch. I will similarly implement my K-NN model from scratch!*

## K - Nearest Neighbour Implementation Details

I will compare the nearest neighbour model with the logistic regression model.

To make predictions for a new data point using k-nearest neighbour, I will need to:

1. Compute the distance from this new data point to every element in the training set
2. Select the top *k* closest neighbour in the training set
3. Find the most common label among those neighbours

I'll use the validation test to select *k*. 

Since I have a fairly large data set, computing the distance between a point in the validation
set and all points in the training set will require more RAM than Google Colab provides.
To make the comptuations tractable, I will:

1. Use only a subset of the training set (only the first 100,000 elements)
2. Use only a subset of the validation set (only the first 1000 elements)
3. I will use the **cosine similarity** rather than Euclidean distance. I will also pre-scale
   each element in training set and the validation set to be a unit vector, so that computing
   the cosine similarity is equivalent to computing the dot product. ie: 
   ![equation](https://latex.codecogs.com/gif.latex?cos%28%5Ctheta%29%20%3D%20%5Cfrac%7Bv%20%5Ccdot%20w%7D%7B%7C%7Cv%7C%7C%20%7C%7Cw%7C%7C%7D)  
   But if both ||v|| and ||w|| are zero, then
   only the dot product remains.

## Results
The accuracies achieved by my Logistic Regression model:  
* Training:  72.17%  
* Validation:  72.49%  
* **Test:  71.32%**  
  
The K-Nearest Neighbours model does not perform as well:  
* Validation Accuracy for k=100:  70.4%  
* Test Accuracy for k=100: 55.%  
