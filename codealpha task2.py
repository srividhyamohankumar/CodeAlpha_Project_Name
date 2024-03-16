#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install ucimlrepo


# In[3]:


from ucimlrepo import fetch_ucirepo 

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

print(breast_cancer_wisconsin_diagnostic.metadata)
print(breast_cancer_wisconsin_diagnostic.variables)


# In[7]:


import pandas as pd

# Load the dataset using the ucimlrepo library
from ucimlrepo import fetch_ucirepo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Create a pandas dataframe from the dataset
df = pd.DataFrame(X, columns=breast_cancer_wisconsin_diagnostic.variables['name'])
df['Diagnosis'] = y

# Save the dataframe to an Excel file
df.to_excel('breast_cancer_wisconsin_diagnostic.xlsx', index=False)


# In[8]:


import os

print(os.getcwd())


# In[9]:


df.to_excel(os.path.join(os.path.expanduser("~"), "Downloads", "breast_cancer_wisconsin_diagnostic.xlsx"), index=False)


# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[36]:


# Load the dataset
url = "C:\\Users\\srivi\\OneDrive\\Desktop\\CODEALPHA\\archive (2)\\breast_cancer_wisconsin_diagnostic.xlsx"
column_names = ['ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']
df = pd.read_excel(url, header=None, names=column_names)
df = df.drop('ID', axis=1)  # Drop the ID column

# Preprocess the dataset
X = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
y = df.Diagnosis.astype(str)  # Convert Diagnosis to string type
# Encode the categorical values
le = LabelEncoder()
y = le.fit_transform(y)


# In[37]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the data into features (X) and target variable (y)
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[38]:


# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




