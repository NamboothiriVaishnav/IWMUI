# Databricks notebook source
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# COMMAND ----------

train_m102 = pd.read_csv(r'C:\Users\Rajesh.Mandal\Downloads\initiative\phm_data_challenge_2018\train\01_M02_DC_train.csv')
fault_m102 = pd.read_csv(r'C:\Users\Rajesh.Mandal\Downloads\initiative\phm_data_challenge_2018\train\train_faults\01_M02_train_fault_data.csv')

# COMMAND ----------

merge_train_fault_m102 = pd.merge(train_m102,
                     fault_m102,
                     how='left',
                    on=['time','Tool'])
merge_train_fault_m102.head()

# COMMAND ----------

df = merge_train_fault_m102
df.shape

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # fill null value

# COMMAND ----------

df['fault_name'] = df['fault_name'].replace(np.nan,'No_fault')
df['FIXTURESHUTTERPOSITION'] =  df['FIXTURESHUTTERPOSITION'].ffill(axis = 0)
df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # check the fault category

# COMMAND ----------

df['fault_name'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # Replace the categorical values witn number
# MAGIC No_fault  = 0    
# MAGIC FlowCool Pressure Dropped Below Limit = 1  
# MAGIC Flowcool leak                         = 2   
# MAGIC Flowcool Pressure Too High Check Flowcool Pump = 3   

# COMMAND ----------

df['fault_name'].replace('No_fault',0, inplace = True)
df['fault_name'].replace('FlowCool Pressure Dropped Below Limit',1, inplace = True)
df['fault_name'].replace('Flowcool leak',2, inplace = True)
df['fault_name'].replace('Flowcool Pressure Too High Check Flowcool Pump',3, inplace = True)

#check again
df['fault_name'].value_counts()

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # convert time column into date

# COMMAND ----------

# df['time'] = pd.to_datetime(df['time'])
# df.head()

# COMMAND ----------

df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC # Drop Tool Column

# COMMAND ----------

df.drop(['Tool'], axis = 1, inplace = True)
df.columns

# COMMAND ----------

df.shape

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Normalize the datasets

# COMMAND ----------

df['fault_name'] = df['fault_name'].astype('int')

# COMMAND ----------

df_train = df.iloc[:,:-1]
df_trainY = df.iloc[:,-1]
df_trainY = df_trainY.values.reshape(-1,1)

# COMMAND ----------

df_scaler = MinMaxScaler(feature_range = (0,1))
df_label = MinMaxScaler(feature_range = (0,1))
df_trainX = df_scaler.fit_transform(df_train)
df_trainY = df_label.fit_transform(df_trainY)

# COMMAND ----------

df_trainX.shape, df_trainY.shape

# COMMAND ----------

sequence_length = 10

# COMMAND ----------

def generate_data(X, y, sequence_length = 20, step = 1):
    X_local = []
    y_local = []
    for start in range(0, len(df_trainX) - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(y[end-1])
    return np.array(X_local), np.array(y_local)

X_sequence, y = generate_data(df_trainX, df_trainY)

# COMMAND ----------

X_sequence.shape, y.shape

# COMMAND ----------

def Error(y_pred, y_real):
    y_pred = np.nan_to_num(y_pred, copy = True)
    y_real = np.nan_to_num(y_real, copy = True)
    temp = np.exp(-0.001 * y_real) * np.abs(y_real - y_pred)
    error = np.sum(temp)
    return error

# COMMAND ----------

def customLoss(y_pred, y_real):
    return K.sum(K.exp(-0.001 * y_real) * K.abs(y_real - y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC # split the datasets

# COMMAND ----------

training_size = int(len(X_sequence) * 0.8)

# COMMAND ----------

X_train, y_train = X_sequence[:training_size], y[:training_size]
X_test, y_test = X_sequence[training_size:], y[training_size:]

# COMMAND ----------

X_train.shape

# COMMAND ----------

model = tf.keras.Sequential()
model.add(LSTM(100, input_shape = (20, 23)))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))

model.compile(loss="SparseCategoricalCrossentropy",
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
              , optimizer="adam")

model.summary()

# COMMAND ----------

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
model.fit(X_train, y_train, batch_size=64, epochs=10, callbacks = [callback])

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

y_test_prob = model.predict(X_test, verbose=1)

# COMMAND ----------

y_test_prob

# COMMAND ----------

plt.figure()
plt.plot(X_test[:,0],label = 'Real')
plt.plot(y_test_prob[:,0],label = 'Prediction')
plt.legend()
plt.show()

# COMMAND ----------


