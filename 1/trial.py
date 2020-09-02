import numpy as np
import pandas as pd
import random

outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
dataset ={'outlook':outlook,'temp':temp,'humidity':humidity,'windy':windy,'play':play}
df = pd.DataFrame(dataset,columns=['outlook','temp','humidity','windy','play'])

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

    diff_output = df[attribute].unique()

    num_of_instances = len(df.iloc[:, -1])

    for i in diff_output:
        dp = df.loc[df[attribute] == i]
        # print[dp]
        num_of_instances_attribute = len(dp.iloc[:, -1])
        entropy_of_attribute += (
            num_of_instances_attribute / num_of_instances
        ) * get_entropy_of_dataset(dp)
        # print(entropy_of_attribute)

    return abs(entropy_of_attribute)


"""Return Information Gain of the attribute provided as parameter"""
# input:int/float/double/large,int/float/double/large
# output:int/float/double/large
def get_information_gain(df, attribute):
    information_gain = 0

    table_entropy=get_entropy_of_dataset(df)

    attribute_entropy= get_entropy_of_attribute(df,attribute)
    
    information_gain= table_entropy-attribute_entropy
	
    return information_gain


""" Returns Attribute with highest info gain"""
# input: pandas_dataframe
# output: ({dict},'str')
def get_selected_attribute(df):

    information_gains = {}
    selected_column = ""


    
    for i, j in df.iteritems():
        information_gains[i]=get_information_gain(df,i)
    information_gains.popitem()
    print(information_gains)

    selected_column=max(information_gains, key= information_gains.get)

    print(selected_column)
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


get_selected_attribute(df)