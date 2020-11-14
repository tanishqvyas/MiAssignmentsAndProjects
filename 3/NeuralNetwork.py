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
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        m = Y.shape[0]
        # Compute loss from AL and y.
        cost = (-1./m) * np.sum(Y*np.log(AL)+(1-Y)
                                * np.log(1-AL)) + self.epsilon


        # To make sure our cost's shape is what we expect
        cost = np.squeeze(cost)

        return cost

    # Initialization

    def __init__(self, x_train, y_train, x_test, y_test, size_of_ip_layer, size_of_hidden_layer, size_of_op_layer, ip_layer_activation, hidden_layer_activation, op_layer_activation, num_epoch, learning_rate, type_of_initilization="Random", regularization=None):
        self.x_train = x_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.x_test = x_test.to_numpy()
        self.y_test = y_test.to_numpy()

        self.num_of_samples = self.x_train.shape[0]

        self.size_of_ip_layer = size_of_ip_layer
        self.size_of_hidden_layer = size_of_hidden_layer
        self.size_of_op_layer = size_of_op_layer

        self.ip_layer_activation = ip_layer_activation
        self.hidden_layer_activation = hidden_layer_activation
        self.op_layer_activation = op_layer_activation
        self.epochs = num_epoch
        self.learning_rate = learning_rate
        self.weights_and_biases = {}
        self.type_of_initilization = type_of_initilization

        # Special varables
        self.epsilon = 1e-7
        self.regularization = regularization
        self.history = {
            "Training Loss": [],
            "Training Accuracy": [],
            "Testing Accuracy": [],
            "Test Loss": []
        }

    # Function to initialize weights and biases

    def initialize_weights_and_biases(self, model_weights_biases=None):

        # Initializing untrained model
        if(model_weights_biases == None):

            # Layer 1
            W1 = np.random.randn(self.size_of_hidden_layer,
                                 self.size_of_ip_layer) * 0.01

            b1 = np.zeros((self.size_of_hidden_layer, 1))

            # Layer 2
            if(self.type_of_initilization == "Random"):
                W2 = np.random.randn(self.size_of_op_layer,
                                     self.size_of_hidden_layer) * 0.01

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

    def get_history(self):

        # Returning the history of the model
        return self.history

    def forward_pass(self, X):

        # Input layer
        Z1 = np.dot(self.weights_and_biases["w1"],
                    X.T) + self.weights_and_biases["b1"]
        # print(self.weights_and_biases["w1"].shape)
        # print(self.x_train.T.shape)
        # print(self.weights_and_biases["b1"].shape)
        A1 = self.tanh(Z1)

        # Hidden Layer
        Z2 = np.dot(self.weights_and_biases["w2"],
                    A1) + self.weights_and_biases["b2"]
        A2 = self.sigmoid(Z2)

        # Output Layer
        Y_pred = A2

        params = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
            "W2": self.weights_and_biases["w2"]
        }

        return Y_pred, params

    def backward_pass(self, params):

        dJ_by_dZ2 = params["A2"] - self.y_train

        dJ_by_dW2 = (1 / self.num_of_samples) * \
            np.dot(dJ_by_dZ2, params["A1"].T)

        dJ_by_db2 = (1 / self.num_of_samples) * \
            np.sum(dJ_by_dZ2, axis=1, keepdims=True)

        dJ_by_dZ1 = np.multiply(np.dot(
            self.weights_and_biases["w2"].T, dJ_by_dZ2), 1 - np.power(params["A1"], 2))

        dJ_by_dW1 = (1 / self.num_of_samples) * \
            np.dot(dJ_by_dZ1, self.x_train)

        dJ_by_db1 = (1/self.num_of_samples) * \
            np.sum(dJ_by_dZ1, axis=1, keepdims=True)

        # Saving all the values
        deravatives = {
            "dw2": dJ_by_dW2,
            "dw1": dJ_by_dW1,
            "db1": dJ_by_db1,
            "db2": dJ_by_db2
        }

        return deravatives

    def predict(self, X):
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values

        yhat is a list of the predicted value for df X

        """
        yhat, _ = self.forward_pass(X)

        return yhat

    def get_accuracy(self, pred, actual):

        num_of_elements = pred.shape[1]
        correct = 0
        pred = np.round(pred.T)
        actual = actual.reshape((actual.shape[0], 1))

        for i in range(num_of_elements):
            if(pred[i] == actual[i]):
                correct += 1

        return (correct / num_of_elements) * 100

    def fit(self):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''

        # Training
        for epoch in range(self.epochs):

            # Compute forward Pass
            y_train_pred, params = self.forward_pass(self.x_train)

            # Compute Loss
            loss = self.crossentropy_cost(y_train_pred, self.y_train)

            # Back prop
            deravatives = self.backward_pass(params)

            # Update Paramaters
            self.weights_and_biases["w2"] = self.weights_and_biases["w2"] - \
                self.learning_rate*deravatives["dw2"]
            self.weights_and_biases["b2"] = self.weights_and_biases["b2"] - \
                self.learning_rate*deravatives["db2"]

            self.weights_and_biases["w1"] = self.weights_and_biases["w1"] - \
                self.learning_rate*deravatives["dw1"]
            self.weights_and_biases["b1"] = self.weights_and_biases["b1"] - \
                self.learning_rate*deravatives["db1"]

            # Get training accuracy
            train_acc = self.get_accuracy(y_train_pred, self.y_train)

            # Calculate Validation Accuracy and Loss
            val_predictions = self.predict(self.x_test)
            val_acc = self.get_accuracy(val_predictions, self.y_test)
            val_loss = self.mse_loss(val_predictions, self.y_test)

            # Saving the data for plotting purpose
            self.history["Training Loss"].append(loss)
            self.history["Training Accuracy"].append(train_acc)
            self.history["Testing Accuracy"].append(val_acc)
            self.history["Test Loss"].append(val_loss)

            print(
                f"Epoch #{epoch+1} : val_loss={val_loss}, val_acc={val_acc}, training_loss={loss}, training_acc={train_acc}\n")

    def CM(y_test, y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i] > 0.6):
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0

        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(len(y_test)):
            if(y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp+1
            if(y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn+1
            if(y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp+1
            if(y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn+1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)

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

    # sum of all null values in dataset per column
    print(dataset.isnull().sum())

    print(dataset["Age"].mean())  # mean of Age

    # replacing all missing values with mean of Age
    dataset['Age'].fillna(value=dataset['Age'].mean(), inplace=True)

    print(dataset["Weight"].mean())  # mean of Weight

    # replacing all missing values with mean of Weight
    dataset['Weight'].fillna(value=dataset['Weight'].mean(), inplace=True)

    # replacing all missing values with mode of Delivery phase
    dataset['Delivery phase'].fillna(
        dataset['Delivery phase'].mode()[0], inplace=True)

    print(dataset["HB"].mean())  # mean of HB

    # replacing all missing values with mean of HB
    dataset['HB'].fillna(value=dataset['HB'].mean(), inplace=True)

    print(dataset["BP"].mean())  # mean of BP

    # replacing all missing values with mean of BP
    dataset['BP'].fillna(value=dataset['BP'].mean(), inplace=True)

    # using the ffill method to fill in the missing values in Education
    dataset["Education"].fillna(method='ffill', inplace=True)

    # using the ffill method to fill in the missing values in Residence
    dataset["Residence"].fillna(method='ffill', inplace=True)

    # no null values remaining in dataset, it has been cleaned
    print(dataset.isnull().sum())

    print(dataset)

    # Extracting Features and Labels
    features = dataset.drop(
        dataset.columns[[-1]], axis=1)  # Remove last Column
    label = dataset[dataset.columns[-1]]  					# Extract Last Column

    #--------------------- MODEL ----------------------------------#

    Num_of_Folds = 10
    model_learning_rate = 0.07

    # Get the current weights and biases for K-fold Approach
    current_weights_and_biases = None
    Fold_training_history = []

    # Implementing K-fold approach
    for fold in range(Num_of_Folds):

        print("<-------------------------------Beginning Fold Number : ",
              fold+1, "--------------------------------->\n")

        # Making a train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            features, label, test_size=0.4, random_state=42)

        # Initialize
        model = NeuralNetworkFromScratch(x_train, y_train, x_test, y_test,
                                         size_of_ip_layer=9,
                                         size_of_hidden_layer=15,
                                         size_of_op_layer=1,
                                         ip_layer_activation="relu",
                                         hidden_layer_activation="relu",
                                         op_layer_activation="relu",
                                         num_epoch=169,
                                         learning_rate=model_learning_rate,
                                         type_of_initilization="Random",
                                         regularization = "L2"
                                         )

        if(current_weights_and_biases == None):
            model.initialize_weights_and_biases()
        else:
            # Load previously trained model
            model.initialize_weights_and_biases(
                model_weights_biases=current_weights_and_biases)

        # Training
        model.fit()

        # Get the current weights and biases for K-fold Approach
        current_weights_and_biases = model.get_current_model()

        # Saving fold Specific History
        Fold_training_history.append(model.get_history())

        # Reduce lr
        model_learning_rate -= (0.069*model_learning_rate)

    # summarize history for accuracy
    # plt.plot(Fold_training_history[0]["Training Accuracy"])
    # plt.plot(Fold_training_history[0]["Testing Accuracy"])
    plt.plot(Fold_training_history[0]["Training Loss"])
    plt.plot(Fold_training_history[0]["Test Loss"])
    # plt.plot(history.history['lr'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    # plt.legend(['train acc', 'val acc', 'train loss',
    #             'val loss'], loc='upper right')

    # plt.legend(['train acc', 'val acc'], loc='upper right')
    plt.legend(['train loss', 'val loss'], loc='upper right')

    plt.show()
