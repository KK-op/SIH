#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("C:\\Users\DELL\Downloads\WQuality_River-Data-2021 GANGA1 edited.csv")
df.head()


# In[2]:


df.isnull().sum()


# In[3]:


df.dtypes


# In[ ]:





# In[ ]:





# In[4]:


import numpy as np
df['MIN - TEMPERATURE ( C)'].fillna(df['MIN - TEMPERATURE ( C)'].mean(),inplace=True)


# In[5]:


df['MAX - TEMPERATURE ( C)'].fillna(df['MAX - TEMPERATURE ( C)'].mean(),inplace=True)


# In[6]:


df['MIN - NITRATE\n(mg/L)'].fillna(df['MIN - NITRATE\n(mg/L)'].mean(),inplace=True)


# In[7]:


df['MAX - NITRATE\n(mg/L)'].fillna(df['MAX - NITRATE\n(mg/L)'].mean(),inplace=True)


# In[8]:


df['MIN - FECAL\nCOLIFORM\n(MPN/100ML)'].fillna(df['MIN - FECAL\nCOLIFORM\n(MPN/100ML)'].mean(),inplace=True)


# In[9]:


df['MAX - FECAL\nCOLIFORM\n(MPN/100ML)'].fillna(df['MAX - FECAL\nCOLIFORM\n(MPN/100ML)'].mean(),inplace=True)


# In[10]:


df['MIN - TOTAL\nCOLIFORM\n(MPN/100ML)'].fillna(df['MIN - TOTAL\nCOLIFORM\n(MPN/100ML)'].mean(),inplace=True)


# In[11]:


df['MAX - TOTAL\nCOLIFORM\n(MPN/100ML)'].fillna(df['MAX - TOTAL\nCOLIFORM\n(MPN/100ML)'].mean(),inplace=True)


# In[12]:


df['MIN - FECAL\nSTREPTOCOCCI\n(MPN/100ML)'].fillna(df['MIN - FECAL\nSTREPTOCOCCI\n(MPN/100ML)'].mean(),inplace=True)


# In[13]:


df['MAX - FECAL\nSTREPTOCOCCI\n(MPN/100ML)'].fillna(df['MAX - FECAL\nSTREPTOCOCCI\n(MPN/100ML)'].mean(),inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


df.head()


# In[16]:


col_2=pd.get_dummies(df['NAME OF\nMONITORING\nLOCATION PRIMARY WATER QUALITY CRITERIA NOTIFIED\nUNDER E(P) RULES, 1986'],dtype=int)
col_3=pd.get_dummies(df['STATE NAME'],dtype=int)
df=pd.concat([df,col_2,col_3],axis='columns')


# In[17]:


df.head()


# In[18]:


df.drop(columns=['NAME OF\nMONITORING\nLOCATION PRIMARY WATER QUALITY CRITERIA NOTIFIED\nUNDER E(P) RULES, 1986','STATE NAME'],axis=1,inplace=True)


# In[24]:


df.fillna(0,inplace=True)


# In[28]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


# In[31]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_scaled)
cluster_assignments = kmeans.labels_

# Use the cluster assignments to make predictions
def predict_pollution(cluster_assignment):
    # Use the cluster assignment to determine the predicted temperature
    # For example, you can use the mean temperature of the cluster
    predicted_pollution = df_scaled[cluster_assignment].mean()
    return predicted_pollution

# Test the model
predicted_pollution_1 = []
for i in range(len(cluster_assignments)):
    predicted_pollution = predict_pollution(cluster_assignments[i])
    predicted_pollution_1.append(predicted_pollution)
print(predicted_pollution_1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[0:train_size], df_scaled[train_size:len(df_scaled)]

# Create the training and testing datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   # 'i' is the starting point, 'i+time_step' is the ending point
        dataX.append(a)
        dataY.append(dataset[(i+time_step), 0])
    return np.array(dataX), np.array(dataY)

time_step = 24  # assuming 24 hourly readings per day
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Make predictions for the next 5 days
predictions = []
for i in range(5):
    last_24_hours = df_scaled[-24:]  # get the last 24 hours of data
    last_24_hours = np.reshape(last_24_hours, (1, 24, 1))
    prediction = model.predict(last_24_hours)
    predictions.append(prediction)

print(predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


from sklearn.model_selection import train_test_split


# In[25]:


X=df.drop('RIVER POLLUTION',axis=1)
y=df['RIVER POLLUTION']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[26]:


from sklearn.ensemble import RandomForestClassifier as rnd


# In[27]:


model=rnd()
model.fit(X_train,y_train)
model.predict(X_test)


# In[ ]:




