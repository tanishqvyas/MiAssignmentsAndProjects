#1 Implementation

The implementation involves a 2 layer neural network with 10 input and 15 hidden layer neurons. The activations used are Tanh and Sigmoid.
The Cost function is Cross Entropy. The Optimization method used is RMS prop. The training involves the use of K-Fold training method.


(NOTE : These values may differ slightly sometimes depending on the test train split returned by the test_train_split function.
These values even increase more in case of higher folds, and epoch per fold, addition of Xavier, RMS Prop).
The final achieved values are -->
Precision : 1.0
Recall : 0.9523809523809523
F1 SCORE : 0.975609756097561
Testing Accuracy :  96.551%
Training Accuracy : 92.53%
Confusion Matrix : 
[[8, 0], [1, 20]]


#2 List of Hyper Parameters

- test_train_split : 0.3
- learning rate : 0.07 (initially), it changes with each Fold of training
- Number of layers : 2
- Number of Neurons : 10 in layer 1 and 15 in layer 2
- Activation Functions : Tanh, Sigmoid
- Cost function : Cross entropy
- decay rate for RMS Prop : 0.9
- epochs per fold : 24
- Number of Folds : 3
- Type of Initialization : Henormal, Xavier, Random

- Fold coefficient : 7 (self made parameter for dynamic lr for each fold)


#3 Key feature of the Design : 

The key feature of the design is the use of K-Fold Training coupled with the normal methods. This method involves training 
the model for K training sessions and where each successive session has a different test train split. 
The weights and biases are initialized according to above mentioned methods for the first Training session. The successive sessions
just loads the best weights and biases from the last session. Each successive training session has a reduced learning rate.
Thus it helps to train our model and increase the accuracy of the model even with less data. The number of Folds/Training sessions is 
kept low to avoid over fitting problems.

So, let's say we have normal training session with 99 epochs and the results from this is compared against 3-Fold, 33 Epochs per fold, and
Thus effectively 99 epochs. But the results obtained by the latter are much better than the former. Even a lesser number of epochs per fold
outperforms the former normal trainning method. The learning rate for each fold varies.

This one design choice alone is very effective. This when coupled with other additional methods like Xavier initialization, RMS Prop, etc gives even 
better results than before.




#4 Implementation beyond the basics

- RMS Prop
- Different Initialization methods : Henormal, Xavier, Random
- K-Fold method for Training
- Dynamic Learning Rate

#5 Steps to run the file

- Ensure that the file named "Cleaned_LBW_Dataset.csv" is present in the same directory as the python file.
- then run the file with the following command
- For Linux : python3 NeuralNetwork.py
- Foe Windows : python NeuralNetwork.py