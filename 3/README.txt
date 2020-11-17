#1 Implementation

The implementation involves a 2 layer neural network with 9 input and 15 hidden layer neurons. The activations used are Tanh and Sigmoid.
The Cost function is Cross Entropy. The Optimization method used is RMS prop. The training involves the use of K-Fold training method.


The final achieved values are -->
Precision : 1.0
Recall : 0.769
F1 SCORE : 0.8696
Testing Accuracy :  79.310
Training Accuracy : 82.089
Confusion Matrix : 
[[3, 0], [6, 20]]



#2 List of Hyper Parameters

- test_train_split : 0.3
- learning rate : 0.07 (initially), it changes with each Fold of training
- Number of layers : 2
- Number of Neurons : 9 in layer 1 and 15 in layer 2
- Activation Functions : Tanh, Sigmoid
- Cost function : Cross entropy
- decay rate for RMS Prop : 0.9
- epochs : 140
- Number of Folds : 3
- Fold coefficient : 7
- Type of Initialization : Henormal, Xavier, Random


#3 Key feature of the Design

The key feature of the design is the use of K-Fold Training coupled with the normal methods. This method involves training 
the model for K training sessions and where each successive session has a different test train split. 
The weights and biases are initialized according to above mentioned methods for the first Training session. The successive sessions
just loads the best weights and biases from the last session. Each successive training session has a reduced learning rate.
Thus it helps to train our model and increase the accuracy of the model even with less data. The number of Folds/Training sessions is 
kept low to avoid over fitting problems.




#4 Implementation beyond the basics

- RMS Prop
- Different Initialization methods : Henormal, Xavier, Random
- K-Fold method for Training
- Dynamic Learning Rate

#5 Steps to run the file

- Ensure that the file named "LBW_Dataset.csv" is present in the same directory as the python file.
- then run the file with the following command
- For Linux : python3 NeuralNetwork.py
- Foe Windows : python NeuralNetwork.py