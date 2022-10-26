# Databricks notebook source
pip install keras

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import random
random.seed(1234)
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


import math
 

from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras.callbacks
from keras import backend as K

# COMMAND ----------

from google.colab import drive
drive.mount('/content/drive')

# COMMAND ----------

import pandas as pd
path ="/content/drive/MyDrive/01_M01_DC_train.csv"
sensor_data =pd.read_csv(path)
sensor_data=sensor_data.iloc[0:50000]

# COMMAND ----------

print(sensor_data)

# COMMAND ----------

sensor_data=sensor_data.drop(["Tool","stage","Lot","runnum","recipe","recipe_step"],axis =1)

# COMMAND ----------

train = sensor_data.iloc[0:35000]
test = sensor_data.iloc[35000:]

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# COMMAND ----------

 from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# COMMAND ----------

length = 144 # Length of the output sequences (in number of timesteps)
batch_size = 1 #Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)

# COMMAND ----------

X,y = generator[0]

# COMMAND ----------

print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
import keras

# COMMAND ----------

 scaled_train.shape

# COMMAND ----------

# # define model
# model = Sequential()

# # Simple RNN layer
# model.add(LSTM(100,input_shape=(length,scaled_train.shape[1])))

# # Final Prediction (one neuron per feature)
# model.add(Dense(scaled_train.shape[1]))

# model.compile(optimizer='adam', loss='mse')

# COMMAND ----------

import tensorflow as tf
lstm_multi = tf.keras.models.Sequential()
lstm_multi.add(tf.keras.layers.LSTM(150,input_shape=(length,scaled_train.shape[1]),return_sequences=True))
lstm_multi.add(tf.keras.layers.Dropout(0.2)),
lstm_multi.add(tf.keras.layers.LSTM(units=100,return_sequences=False)),
lstm_multi.add(tf.keras.layers.Dropout(0.2)),
lstm_multi.add(tf.keras.layers.Dense(scaled_train.shape[1])),
lstm_multi.compile(optimizer='adam', loss='mse')

# COMMAND ----------

 lstm_multi.summary()

# COMMAND ----------

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=1)
validation_generator = TimeseriesGenerator(scaled_test,scaled_test, 
                                           length=length, batch_size=batch_size)

# COMMAND ----------

 lstm_multi.fit_generator(generator,epochs=10,
                    validation_data=validation_generator,
                   callbacks=[early_stop])

# COMMAND ----------

 lstm_multi.history.history.keys()

# COMMAND ----------

losses = pd.DataFrame(lstm_multi.history.history)
losses.plot()

# COMMAND ----------

 first_eval_batch = scaled_train[-length:]

# COMMAND ----------

 first_eval_batch

# COMMAND ----------

first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))

# COMMAND ----------

lstm_multi.predict(first_eval_batch)

# COMMAND ----------

scaled_test[0]

# COMMAND ----------

n_features = scaled_train.shape[1]
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = lstm_multi.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

# COMMAND ----------

test_predictions

# COMMAND ----------


true_predictions = scaler.inverse_transform(test_predictions)

# COMMAND ----------

true_predictions

# COMMAND ----------

test

# COMMAND ----------

true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)

# COMMAND ----------

true_predictions 

# COMMAND ----------

test_predictions=pd.DataFrame(data=test_predictions,columns=test.columns)

# COMMAND ----------

test_predictions

# COMMAND ----------

sensors_data = sensor_data .iloc[0:14999]
sensors_data = sensor_data.drop(["IONGAUGEPRESSURE","ETCHBEAMVOLTAGE","FLOWCOOLFLOWRATE","ETCHBEAMCURRENT","ETCHSOURCEUSAGE","ETCHSUPPRESSORVOLTAGE","ETCHSUPPRESSORCURRENT","FLOWCOOLPRESSURE","ETCHGASCHANNEL1READBACK",	"ETCHPBNGASREADBACK",	"FIXTURETILTANGLE",	"ROTATIONSPEED",	"ACTUALROTATIONANGLE",	"FIXTURESHUTTERPOSITION",	"ETCHSOURCEUSAGE",	"ETCHAUXSOURCETIMER",	"ETCHAUX2SOURCETIMER",	"ACTUALSTEPDURATION"],axis = 1 )
sensors_data

# COMMAND ----------

test_predictions =test_predictions.drop(["time"],axis = 1)

# COMMAND ----------

df_mergeds = pd.concat([test_predictions,sensors_data], axis=1)

# COMMAND ----------

df = df_mergeds.iloc[0:15000]
df

# COMMAND ----------

# shift column 'Name' to first position
first_column = df.pop('time')
  
# insert column using insert(position,column_name,
# first_column) function
df.insert(0, 'time', first_column)
df 


# COMMAND ----------

merge_train_fault_m102 = pd.merge(train_m102,
                     fault_m102,
                     how='left',
                    on=['time'])
merge_train_fault_m102.head()

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["IONGAUGEPRESSURE"],test["IONGAUGEPRESSURE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHBEAMVOLTAGE"],test["ETCHBEAMVOLTAGE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHBEAMCURRENT"],test["ETCHBEAMCURRENT"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHSUPPRESSORVOLTAGE"],test["ETCHSUPPRESSORVOLTAGE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHSUPPRESSORCURRENT"],test["ETCHSUPPRESSORCURRENT"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["FLOWCOOLFLOWRATE"],test["FLOWCOOLFLOWRATE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["FLOWCOOLPRESSURE"],test["FLOWCOOLPRESSURE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHGASCHANNEL1READBACK"],test["ETCHGASCHANNEL1READBACK"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHPBNGASREADBACK"],test["ETCHPBNGASREADBACK"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["FIXTURETILTANGLE"],test["FIXTURETILTANGLE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ROTATIONSPEED"],test["ROTATIONSPEED"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ACTUALROTATIONANGLE"],test["ACTUALROTATIONANGLE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["FIXTURESHUTTERPOSITION"],test["FIXTURESHUTTERPOSITION"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHSOURCEUSAGE"],test["ETCHSOURCEUSAGE"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHAUXSOURCETIMER"],test["ETCHAUXSOURCETIMER"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ETCHAUX2SOURCETIMER"],test["ETCHAUX2SOURCETIMER"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df["ACTUALSTEPDURATION"],test["ACTUALSTEPDURATION"])
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(true_predictions,test)
rmse = np.sqrt(mse)
print(mse)
print(rmse)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


