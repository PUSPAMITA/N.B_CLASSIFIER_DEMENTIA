#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split  ##for splitting training and test data
from sklearn.preprocessing import StandardScaler   ##feature scaling
##for assesing the accuracy of our model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import confusion_matrix
##for plotting figures
import matplotlib.pyplot as plt

 #Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#This code iterates through all files within a directory and its subdirectories, printing the full path of each file.
import os 
for dirname, _, filenames in os.walk(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[61]:


df = pd.read_csv(r"C:\Users\PUSPAMITA\Desktop\Dementia Data Set\dementia_patients_health_data.csv")
df.head()


# In[62]:


df.describe()


# In[63]:


df.info()


# In[64]:


df.isnull().sum()


# In[65]:


###dropping features since there is more than 515 blanks out of 1000.
df=df.drop('Prescription',axis=1)
df=df.drop('Dosage in mg', axis=1)


# In[66]:


### Replace non-numerical values with integers in columns
### Define a function to replace non-numerical values with integers

# replacing values
df['Education_Level'].replace(['Primary School','Secondary School', 'Diploma/Degree','No School'],[1, 2,3,0], inplace=True)

df['Dominant_Hand'].replace(['Right','Left'],[1, 2], inplace=True)

df['Gender'].replace(['Female','Male'],[1, 2], inplace=True)

df['Family_History'].replace(['Yes','No'],[1, 0], inplace=True)

df['Smoking_Status'].replace(['Former Smoker','Current Smoker','Never Smoked'],[1, 2,0], inplace=True)

df['APOE_ε4'].replace(['Positive','Negative'],[1, 0], inplace=True)

df['Physical_Activity'].replace(['Mild Activity','Moderate Activity','Sedentary'],[1, 2,0], inplace=True)

df['Depression_Status'].replace(['Yes','No'],[1, 0], inplace=True)

df['Medication_History'].replace(['Yes','No'],[1, 0], inplace=True)

df['Nutrition_Diet'].replace(['Low-Carb Diet','Mediterranean Diet','Balanced Diet'],[1, 2,0], inplace=True)

df['Sleep_Quality'].replace(['Poor','Good'],[1, 0], inplace=True)

df['Chronic_Health_Conditions'].replace(['Heart Disease','Hypertension','Diabetes','None'],[1, 2,3,0], inplace=True)


# In[67]:


df.head()


# In[68]:


##Feature Scaling


X = df.drop('Dementia', axis=1)
y = df['Dementia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert Pandas Series to NumPy array
y_train_array = np.array(y_train)

# Reshape the array
y_train_reshaped = y_train_array.reshape((800, 1))

# Convert Pandas Series to NumPy array
y_test_array = np.array(y_test)

# Reshape the array
y_test_reshaped = y_test_array.reshape((200, 1))


# In[80]:


###Naive Bayes Classification Algorithm...
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = None
        self.feature_probabilities = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Ensure y has the correct shape
        n_classes = len(self.classes)

        # Compute class probabilities
        self.class_probabilities = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_probabilities[i] = np.sum(y == c) / float(n_samples)
            

        # Compute feature probabilities
        self.feature_probabilities = []
        for i, c in enumerate(self.classes):
            X_c = X[y.flatten() == c]  # Flatten y to ensure correct shape
          
            probabilities = []
            for j in range(n_features):
                feature_values = np.unique(X[:, j])
                
                counts = np.zeros(len(feature_values))
                
                for k, v in enumerate(feature_values):
                    counts[k] = np.sum(X_c[:, j] == v)
                    
                # Laplace smoothing to handle unseen features
                probabilities.append((counts + 1) / (len(X_c) + len(feature_values)))
            self.feature_probabilities.append(probabilities)

    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = []
            for i, c in enumerate(self.classes):
                #class_probability = np.log(self.class_probabilities[i])
                class_probability = (self.class_probabilities[i])
                
                #print(self.class_probabilities[i])
                for j, value in enumerate(x):
                    # If feature value is unseen, set probability to a small value
                    if value not in np.unique(X[:, j]):
                        feature_probability = np.log(1 / len(self.feature_probabilities[i][j]))
                        #feature_probability=(1 / len(self.feature_probabilities[i][j]))
                    else:
                        feature_probability = np.log(self.feature_probabilities[i][j][np.where(np.unique(X[:, j]) == value)[0][0]])
                        #feature_probability = (self.feature_probabilities[i][j][np.where(np.unique(X[:, j]) == value)[0][0]])
                    
                    class_probability += feature_probability
                probabilities.append(class_probability)
            predictions.append(self.classes[np.argmax(probabilities)])
        return predictions


# In[81]:


# Create and train Naive Bayes Classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train_scaled, y_train_reshaped)


# In[82]:


# PredictION with test Data 
predictions = nb_classifier.predict(X_test_scaled)
print("Predictions:", predictions)


# In[83]:


###evaluation of the model
y_predicted_cls=predictions

accuracy = accuracy_score(y_test_reshaped, y_predicted_cls)
precision = precision_score(y_test_reshaped, y_predicted_cls)
recall = recall_score(y_test_reshaped, y_predicted_cls)
f1 = f1_score(y_test_reshaped, y_predicted_cls)
roc_auc = roc_auc_score(y_test_reshaped, y_predicted_cls)
conf_matrix = confusion_matrix(y_test_reshaped, y_predicted_cls)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.metrics import roc_curve, roc_auc_score
# Assuming you have true labels and predicted class probabilities
fpr, tpr, thresholds = roc_curve(y_test_reshaped, y_predicted_cls)


# In[84]:


#PLOTTING THE STATISTICAL MEASURES.
# Create subplots for plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16, 4))
# Plot accuracy, precision, recall, F1-score
ax1.bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'], [accuracy, precision, recall, f1])
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Metrics')
# Plot confusion matrix as heatmap
ax2.imshow(conf_matrix, cmap='Blues')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
ax2.set_title('Confusion Matrix')
ax2.text(0, 0, str(conf_matrix[0, 0]), ha='center', va='center', fontsize=12, color='white')
ax2.text(0, 1, str(conf_matrix[0, 1]), ha='center', va='center', fontsize=12, color='black')
ax2.text(1, 0, str(conf_matrix[1, 0]), ha='center', va='center', fontsize=12, color='black')
ax2.text(1, 1, str(conf_matrix[1, 1]), ha='center', va='center', fontsize=12, color='white')
# Placeholder for ROC AUC 
ax3.text(0.5, 0.5,  f'ROC AUC\n{roc_auc:.4f}', ha='center', va='center', fontsize=12)
ax3.set_title('ROC AUC')
ax3.axis('off')
#Plot ROC Curve
plt.figure(figsize=(6,6 ))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid(True)
# Add diagonal line for random classification
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
# Adjust layout
plt.tight_layout()
plt.legend()
plt.show()


# In[ ]:




