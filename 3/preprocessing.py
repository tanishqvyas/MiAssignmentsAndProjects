import pandas as pd

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


# Saving the file 
dataset.to_csv("../data/Cleaned_LBW_Dataset.csv")