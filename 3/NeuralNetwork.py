'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

test_split_size :

Number of Layers :
Number of Nodes in L1 :
Number of Nodes in L2 :

Number of Epochs :
Learning Rate :
Initialization of Weights and Biases :

Activation Function in L1 :
Activation Function in L2 :



'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNetworkFromScratch:

	# Activation Functions

	def sigmoid(self, X):
		z = 1/(1 + np.exp(-X))
		return z

	def sigmoid_der(self, x):
		y = self.sigmoid(x) * (1-self.sigmoid(x))
		return y

	def relu(self, X):
   		return np.maximum(0, X)

	def softmax(self, X):
		expo = np.exp(X)
		expo_sum = np.sum(np.exp(X))
		return expo/expo_sum

	def leakyrelu(x, theta=0.01):
		return np.where(x > 0, x, x * theta)

	def tanh(self, x):
		t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
		return t

	# End of activation Functions

	# cost function

	# Mean squared error loss
	# out -> label predictions
	# Y -> actual true label
	def mse_loss(self, out, Y):
		s = (np.square(out-Y))
		cost = np.sum(s)/len(Y)
		return cost

	# binary cost entropy cost
	def crossentropy_cost(self, AL, Y):
    	# number of examples
		m = Y.shape[1]
		# Compute loss from AL and y.
		cost = (-1./m) * np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))

		# To make sure our cost's shape is what we expect
		cost = np.squeeze(cost)

		return cost

	# Initialization

	def __init__(self, x_train, y_train, x_test, y_test, size_of_ip_layer, size_of_hidden_layer, size_of_op_layer, ip_layer_activation, hidden_layer_activation, op_layer_activation, num_epoch, learning_rate):
		self.x_train = x_train.to_numpy()
		self.y_train = y_train.to_numpy()
		self.x_test = x_test.to_numpy()
		self.y_test = y_test.to_numpy()

		self.size_of_ip_layer = size_of_ip_layer
		self.size_of_hidden_layer = size_of_hidden_layer
		self.size_of_op_layer = size_of_op_layer

		self.ip_layer_activation = ip_layer_activation
		self.hidden_layer_activation = hidden_layer_activation
		self.op_layer_activation = op_layer_activation
		self.epochs = num_epoch
		self.learning_rate = learning_rate
		self.weights_and_biases = {}

	# Function to initialize weights and biases

	def initialize_weights_and_biases(self, model_weights_biases = None):

		# Initializing untrained model
		if(model_weights_biases == None):

			W1 = np.random.randn(self.size_of_hidden_layer, self.size_of_ip_layer) * 0.01
			b1 = np.zeros((self.size_of_hidden_layer, 1))
			W2 = np.random.randn(self.size_of_op_layer, self.size_of_hidden_layer) * 0.01
			b2 = np.zeros((self.size_of_op_layer, 1))

			weights_and_biases = {
				"w1": W1,
				"b1": b1,
				"w2": W2,
				"b2": b2
			}

			self.weights_and_biases = weights_and_biases

		# Loading a trained model
		else:
			self.weights_and_biases = model_weights_biases

	def summary(self):

		print("---------------Model Summary----------------")
		print("Number of Epochs : ", self.epochs)

		print("\nShapes of Test and Train Data : \n")
		print("X Train Shape : ", self.x_train.shape)
		print("Y Train Shape : ", self.y_train.shape)
		print("X Test Shape : ", self.x_test.shape)
		print("Y Test Shape : ", self.y_test.shape)

		print("Weights and Biases : \n")
		print("W1 Shape : ", self.weights_and_biases["w1"].shape)
		print("W2 Shape : ", self.weights_and_biases["w2"].shape)
		print("b1 Shape : ", self.weights_and_biases["b1"].shape)
		print("b2 Shape : ", self.weights_and_biases["b2"].shape)

		print("\nSize of Input Layer : ", self.size_of_ip_layer)
		print("Size of Hidden Layer : ", self.size_of_hidden_layer)
		print("Size of Output Layer : ", self.size_of_op_layer)

		print("\nActivation for Input Layer : ", self.ip_layer_activation)
		print("Activation for Hidden Layer : ", self.hidden_layer_activation)
		print("Activation for Output Layer : ", self.op_layer_activation)


	# Function to return the current model
	def get_current_model(self):
		# Returning the weights and biases of the model
		return self.weights_and_biases

		


	def forward_pass(self):

		# Input layer
		Z1 = np.dot(self.weights_and_biases["w1"],
		            self.x_train.T) + self.weights_and_biases["b1"]
		A1 = self.relu(Z1)

		# Hidden Layer
		Z2 = np.dot(self.weights_and_biases["w2"],
		            A1) + self.weights_and_biases["b2"]
		A2 = self.relu(Z2)

		# Output Layer
		Y_pred = self.sigmoid(A2)

		params = {
			"Z1" : Z1,
			"A1" : A1.T,
			"Z2" : Z2,
			"A2" : A2.T,
			"W2" : self.weights_and_biases["w2"]
		}

		return Y_pred.T, params


	def backward_pass(self, params):

		# Calculating partials deravatives
		dY_by_dA2 = self.sigmoid_der(params["A2"])

		dA2_by_dZ2 = self.sigmoid_der(params["Z2"])

		dZ2_by_dW2 = params["A1"]

		# Since its 1 hence we never use
		dZ2_by_db2 = np.ones((83, 1), dtype=int)

		dZ2_by_dA1 = params["W2"]

		dA1_by_dZ1 = self.sigmoid_der(params["Z1"])

		dZ1_by_dW1 = self.x_train

		# Since its one we never use
		dZ1_by_db1 = 1



		dY_by_dZ2 =  dY_by_dA2 * dA2_by_dZ2.T

		dY_by_dZ1 = np.dot(dY_by_dZ2 * dZ2_by_dA1, dA1_by_dZ1)


		# Computing deravitives to adjust weights and biases using partial derivatives
		dY_by_dW2 = np.dot(dY_by_dZ2.T, dZ2_by_dW2)

		dY_by_db2 = np.dot( dY_by_dZ2.T, dZ2_by_db2)

		dY_by_dW1 = np.dot(dY_by_dZ1 , dZ1_by_dW1)

		dY_by_db1 = dY_by_dZ1

		print("-------------------------------------------")
		print("dY_by_dZ1 : ", dY_by_dZ1.shape)
		print("dZ1_by_dW1 : ", dZ1_by_dW1.shape)
		print("dY_by_dW1 : ", dY_by_dW1.shape)
		print("-------------------------------------------")
		
		# Saving all the values
		deravatives = {
			"dw1" : dY_by_dW1,
			"dw2" : dY_by_dW2,
			"db1": dY_by_db1,
			"db2": dY_by_db2
		}


		return deravatives

	
	def fit(self):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''

		# Store History of training
		history = {
			"Training Loss" : [],
			"Training Accuracy" : [],
			"Testing Accuracy" : [],
			"Test Loss": []
		}

		# Training 
		for epoch in range(self.epochs):

			# Compute forward Pass
			y_train_pred, params = self.forward_pass()

			# Compute Loss
			loss = self.mse_loss(y_train_pred, self.y_train)
			

			# Back prop
			deravatives = self.backward_pass(params)
			

			# Update Paramaters
			self.weights_and_biases["w2"] = self.weights_and_biases["w2"] - self.learning_rate*deravatives["dw2"]
			self.weights_and_biases["b2"] = self.weights_and_biases["b2"] - self.learning_rate*deravatives["db2"]
			self.weights_and_biases["w1"] = self.weights_and_biases["w1"] - self.learning_rate*deravatives["dw1"]
			self.weights_and_biases["b1"] = self.weights_and_biases["b1"] - self.learning_rate*deravatives["db1"]

			print("Loss for epoch #", epoch, " : ", loss)


	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat



	def CM(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
			


	
if __name__ == "__main__":
	
	# Reading in the Data Set
	dataset = pd.read_csv("LBW_Dataset.csv")
	print(dataset.head())

	# <-------------- Pre processing of DataFrame--------------------->

	dataset.describe()

	print(dataset.isnull().sum())  #sum of all null values in dataset per column

	print(dataset["Age"].mean())  #mean of Age

	dataset['Age'].fillna(value=dataset['Age'].mean(), inplace=True) #replacing all missing values with mean of Age

	print(dataset["Weight"].mean())  #mean of Weight

	dataset['Weight'].fillna(value=dataset['Weight'].mean(), inplace=True)  #replacing all missing values with mean of Weight

	dataset['Delivery phase'].fillna(dataset['Delivery phase'].mode()[0], inplace=True) #replacing all missing values with mode of Delivery phase

	print(dataset["HB"].mean())  #mean of HB

	dataset['HB'].fillna(value=dataset['HB'].mean(), inplace=True)  #replacing all missing values with mean of HB

	print(dataset["BP"].mean())  #mean of BP

	dataset['BP'].fillna(value=dataset['BP'].mean(), inplace=True)  #replacing all missing values with mean of BP

	dataset["Education"].fillna( method ='ffill', inplace = True) #using the ffill method to fill in the missing values in Education

	dataset["Residence"].fillna( method ='ffill', inplace = True)  #using the ffill method to fill in the missing values in Residence

	print(dataset.isnull().sum())  #no null values remaining in dataset, it has been cleaned

	print(dataset)


	# Extracting Features and Labels
	features = dataset.drop(dataset.columns[[-1]], axis=1)  # Remove last Column
	label = dataset[dataset.columns[-1]]  					# Extract Last Column


	# Making a train_test_split
	x_train, x_test, y_train, y_test = train_test_split(features, label, test_size= 0.13, random_state= 42)


	# Creating the Model
	model = NeuralNetworkFromScratch(x_train, y_train, x_test, y_test,
										size_of_ip_layer = 9,
										size_of_hidden_layer = 15,
										size_of_op_layer = 1,
										ip_layer_activation = "relu",
										hidden_layer_activation = "relu",
										op_layer_activation = "relu",
										num_epoch = 10,
										learning_rate = 0.001
											)

	# Initializing weights and biases
	model.initialize_weights_and_biases()

	# Print the model summary
	model.summary()

	# Training the Model
	model.fit()
	



