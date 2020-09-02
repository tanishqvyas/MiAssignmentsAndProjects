"""
Assume df is a pandas dataframe object of the dataset given
"""
import numpy as np
import pandas as pd
import random

"""Calculate the entropy of the enitre dataset"""
# input:pandas_dataframe
# output:int/float/double/large


def get_entropy_of_dataset(df):
    entropy = 0

    # finding the outcomes of the classifier
    diff_output = df.iloc[:, -1].unique()

    # finding the n in n-nary classifier
    num_of_diff_output = len(df.iloc[:, -1].unique())

    # total input instances
    num_of_instances = len(df.iloc[:, -1])

    for i in diff_output:
        count = 0
        for j in df.iloc[:, -1]:
            if i == j:
                count += 1
        p = count / num_of_instances
        entropy = entropy - (p) * (np.log2(p))

    return entropy


"""Return entropy of the attribute provided as parameter"""
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float/double/large
def get_entropy_of_attribute(df, attribute):
    entropy_of_attribute = 0

    # given a column i.e, attribute how many unique alues are present in it
    diff_output = df[attribute].unique()

    # number of input
    num_of_instances = len(df.iloc[:, -1])

    for i in diff_output:
        dp = df.loc[df[attribute] == i]
        num_of_instances_attribute = len(dp.iloc[:, -1])
        entropy_of_attribute += (
            num_of_instances_attribute / num_of_instances
        ) * get_entropy_of_dataset(dp)

    return abs(entropy_of_attribute)


"""Return Information Gain of the attribute provided as parameter"""
# input:int/float/double/large,int/float/double/large
# output:int/float/double/large
def get_information_gain(df, attribute):
    information_gain = 0

    # table entropy
    table_entropy = get_entropy_of_dataset(df)

    # attribute entropy
    attribute_entropy = get_entropy_of_attribute(df, attribute)

    information_gain = table_entropy - attribute_entropy

    return information_gain


""" Returns Attribute with highest info gain"""
# input: pandas_dataframe
# output: ({dict},'str')
def get_selected_attribute(df):

    information_gains = {}
    selected_column = ""

    for i, j in df.iteritems():
        information_gains[i] = get_information_gain(df, i)

    # poping the last key value pair in dict as the last value corresponds to the last attribute which is the result and not one of the input
    information_gains.popitem()

    # i dont know how to find max in dict so copied this from net
    selected_column = max(information_gains, key=information_gains.get)

    """
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	"""

    return (information_gains, selected_column)


"""
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

"""
