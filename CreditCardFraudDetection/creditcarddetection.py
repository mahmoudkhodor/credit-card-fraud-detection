import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

# Load the data from the csv file
data = pd.read_csv('Data/creditcard.csv')

# Just exploring the dataset
# You can see the columns name v1, v2 ... this is the result of PCA dimensionality reduction
# that was used to protect the sensitive the information in this dataset
print(data.columns)

# This will display how many transaction we have and how many columns
print(data.shape)

# This give a useful informatrion about each colum this will give the mean, min and max
print(data.describe)
