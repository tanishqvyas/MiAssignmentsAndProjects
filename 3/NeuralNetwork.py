'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark



'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 


class NN:

	# Activation Functions
	def sigmoid (x):
		z = 1/(1 + np.exp(-x)) 
		return z

	def relu(X):
   		return np.maximum(0,X)

	def softmax(X):
		expo = np.exp(X)
		expo_sum = np.sum(np.exp(X))
		return expo/expo_sum

	def leakyrelu(x):
		return np.where(x > 0, x, x * 0.01) 

	def tanh(x):
		t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
		return t
	
	# End of activation Functions

	# cost function

	# Mean squared error loss
	# out -> label predictions
	# Y -> actual true label
	def mse_loss(out, Y): 
		s =(np.square(out-Y)) 
		cost = np.sum(s)/len(y) 
		return cost

	# binary cost entropy cost
	def crossentropy_cost(AL, Y):
    	# number of examples
		m = Y.shape[1]
		# Compute loss from AL and y.
		cost = -1./m * np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
		# To make sure our cost's shape is what we expect 
		cost = np.squeeze(cost)
		
		return cost

	# end of cost Functions
	# end of cost function


	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

	''' X and Y are dataframes '''
	
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
	
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
	import pandas as pd  #importing pandas and numpy

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
	x_train, x_test, y_train, y_test = train_test_split(features, label, test_size= 0.2, random_state= 42)



	



