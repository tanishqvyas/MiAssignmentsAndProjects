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

	def sigmoid (self, X):
		z = 1/(1 + np.exp(-X)) 
		return z

	def relu(self, X):
   		return np.maximum(0,X)

	def softmax(self, X):
		expo = np.exp(X)
		expo_sum = np.sum(np.exp(X))
		return expo/expo_sum

	def leakyrelu(self, x):
		return np.where(x > 0, x, x * 0.01) 

	def tanh(self, x):
		t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
		return t
	
	# End of activation Functions

	# cost function

	# Mean squared error loss
	# out -> label predictions
	# Y -> actual true label
	def mse_loss(self, out, Y): 
		s =(np.square(out-Y)) 
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
	def __init__(self, x_train, y_train, x_test, y_test, size_of_ip_layer, size_of_hidden_layer, size_of_op_layer, ip_layer_activation, hidden_layer_activation, op_layer_activation, num_epoch):
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

		self.weights_and_biases = {}

	

	# Function to initialize weights and biases
	def initialize_weights_and_biases(self):

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
		

	def summary(self):

		print("---------------Model Summary----------------")
		print("Number of Epochs : ", self.epochs)

		print("\nShapes of Test and Train Data : \n")
		print("X Train Shape : ", self.x_train.shape)
		print("Y Train Shape : ", self.y_train.shape)
		print("X Test Shape : ", self.x_test.shape)
		print("Y Test Shape : ", self.y_test.shape)

		# print("\nSize of Input Layer : ", self.size_of_ip_layer)
		# print("Size of Hidden Layer : ", self.size_of_hidden_layer)
		print("Size of Output Layer : ", self.size_of_op_layer)

		print("\nActivation for Input Layer : ", self.ip_layer_activation)
		print("Activation for Hidden Layer : ", self.hidden_layer_activation)
		print("Activation for Output Layer : ", self.op_layer_activation)




	def forward_pass(self):

		Z1 = np.dot(self.weights_and_biases["w1"], self.x_train.T) + self.weights_and_biases["b1"]
		A1 = self.sigmoid(Z1)
		Z2 = np.dot(self.weights_and_biases["w2"], Z1) + self.weights_and_biases["b2"]
		A2 = self.sigmoid(Z2)

		Y_pred = self.sigmoid(A2)


		return Y_pred.T



	''' X and Y are dataframes '''
	
	def fit(self):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''

		for epoch in range(self.epochs):

			# Compute forward Pass
			y_pred = self.forward_pass()

			# Compute the Loss
			loss = self.mse_loss(self.y_train, y_pred)

			# Compute the Backward Pass



			# Update the Parameters


		
		# Save the model


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
										num_epoch = 1
											)

	# Initializing weights and biases
	model.initialize_weights_and_biases()

	# Print the model summary
	model.summary()

	# Training the Model
	model.fit()
	



