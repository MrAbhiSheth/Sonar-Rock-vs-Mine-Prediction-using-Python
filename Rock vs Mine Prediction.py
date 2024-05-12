#!/usr/bin/env python
# coding: utf-8

# Importing the required libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Data Processing

# In[2]:


#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('C:/Users/abhim/Desktop/ML Projects/Submarine/sonar data.csv', header=None)


# In[3]:


sonar_data.head()


# In[4]:


# number of rows and columns
sonar_data.shape


# In[5]:


sonar_data.describe()  #describe --> statistical measures of the data


# In[6]:


sonar_data[60].value_counts()


# In[7]:


sonar_data.groupby(60).mean()


# In[8]:


# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


# In[9]:


print(X)
print(Y)


# Training and test data

# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)


# In[11]:


print(X.shape, X_train.shape, X_test.shape)


# In[12]:


print(X_train)
print(Y_train)


# Model Training --> Logistic Regression 

# In[14]:


model = LogisticRegression()


# In[15]:


#training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# Model evaluation

# In[16]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 


# In[17]:


print('Accuracy on training data : ', training_data_accuracy)


# In[18]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 


# In[19]:


print('Accuracy on test data : ', test_data_accuracy)


# Making a Predictive System

# In[20]:


input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')


# In[27]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, X_test_prediction)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

# Add numerical annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Rock (R)', 'Mine (M)'], rotation=45)
plt.yticks([0, 1], ['Rock (R)', 'Mine (M)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[28]:


# Given values
TP = 9
FP = 2
FN = 3
TN = 7

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Calculate Precision
precision = TP / (TP + FP)

# Calculate Recall (Sensitivity)
recall = TP / (TP + FN)

# Calculate F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# # Analyses :
# 
# Accuracy: 0.76 indicates that the model correctly classified 76% of the data points. This is a moderately good accuracy, but there is room for improvement.
# 
# Precision: 0.82 indicates that out of the data points that the model classified as positive, 82% were actually positive. This is a good precision, which means that the model is good at avoiding false positives.
# 
# Recall: 0.75 indicates that the model captured 75% of the actual positive cases. This is a moderately good recall, but it means that the model missed 25% of the actual positive cases (false negatives).
# 
# F1 Score: 0.78 is the harmonic mean of precision and recall, and it provides a balance between the two. A score of 0.78 indicates a good balance between precision and recall.
# 
# Overall, the model is performing moderately well. There might be room for improvement, especially on recall.

# In[29]:


# ROC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, y_prob, pos_label='M')
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# # Analyses
# 
# An ROC curve is a graph that shows the performance of a classification model at all classification thresholds. It plots the true positive rate (TPR) on the y axis against the false positive rate (FPR) on the x axis.
# 
# The TPR is the proportion of positive cases that were correctly identified by the model.
# The FPR is the proportion of negative cases that were incorrectly identified as positive by the model.
# A perfect classifier would have a TPR of 1 and an FPR of 0, which corresponds to a point in the upper left corner of the ROC curve. A random classifier would perform no better than guessing and would have an ROC curve that follows a diagonal line from (0, 0) to (1, 1).
# 
# The ROC curve in the image shows the performance of a classifier that is better than random but not perfect. The curve starts at (0, 0) and ends at (1, 1), and it bulges up into the upper left corner of the graph, which indicates that the classifier is able to correctly classify a significant number of positive cases while keeping the number of false positive classifications low.

# In[30]:


# Area Under Curve (AUC)
auc = roc_auc_score(Y_test, y_prob)
print('AUC Score:', auc)


# # Analyses  
# An AUC score of 0.7363 is a positive sign for the model's ability to differentiate between positive and negative classes. It performs well across different classification thresholds. However, there's always room for improvement. 
