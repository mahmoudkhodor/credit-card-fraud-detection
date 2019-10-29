import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Load the data from the csv file
data = pd.read_csv('Data/creditcard.csv')

# Just exploring the dataset
# You can see the columns name v1, v2 ... this is the result of PCA dimensionality reduction
# that was used to protect the sensitive the information in this dataset
print(data.columns)

# This will display how many transaction we have and how many columns
print(data.shape)

# This give a useful information about each colum this will give the mean, min and max
print(data.describe)

# In order to save on time and computational requirement, we are going to take
# only 10% of the dataset
data = data.sample(frac=0.1, random_state=1)
print(data.shape)

# Plot histograms of each parameter
data.hist(figsize=(20, 20))
plt.show()

# Determine the number of fraud cases in the dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))

# Correlation matrix
# This will tell us if there is any strong correlation between different variables
# in our dataset, it will tell us if we need to remove certain variables
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Get all the data from the dataframes
columns = data.columns.tolist()

# Filter the columns to remove data that we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predecting on
target = "Class"

X = data[columns]
Y = data[target]

# Print the shapes of X and Y
print(X.shape)
print(Y.shape)

# To this point we successfully extracted and we preproccessed our data, now the good part begins

# The algorithm that being used

# Local Outlier Factor
# is usupervised outlier detection method, it calculate the anomly score
# of each sample, it measure the local deviation of density of a given sample camparin to it neighbors

# Isolation Forest
# It isolates the observations by randomly a feature and randomly and randomly select a split value
# between the max and min of the select feature

# Define a random state
state = 1

# Define the outlier methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),

    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                               contamination=outlier_fraction)
}

# Fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    # Fit the data and tag the outlier
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_funciton(X)
        y_pred = clf.predict(X)

    # Reshape the prediction valeus to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    # Run classification metrics
    print(f'{clf_name}:{n_errors}')
