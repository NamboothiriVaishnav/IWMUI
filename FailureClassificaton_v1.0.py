# Databricks notebook source
import pyspark 
from pyspark.sql import SparkSession

import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import time
import numpy as np 
import pandas as pd
import os
from collections import Counter

# COMMAND ----------

# data1=spark.read.csv("abfss://data@iwmstgacc.dfs.core.windows.net/TrainFaults/export (13).csv",header=True,inferSchema=True)
data1 = pd.read_csv("C:/Users/Rajesh.Mandal/Downloads/initiative/classification/only_sensors_columns/Train_1M02.csv")

# COMMAND ----------

#  test_df=spark.read.csv("abfss://train@iwmstgacc.dfs.core.windows.net/testingm02.csv",header=True,inferSchema=True)  
test_df = pd.read_csv("C:/Users/Rajesh.Mandal/Downloads/initiative/classification/only_sensors_columns/testingm02.csv")

# COMMAND ----------

test_independent = test_df.iloc[:,:-1]
test_independent

# COMMAND ----------

test_dependent_y=test_df.iloc[:,-1]
test_dependent_y

# COMMAND ----------

#sensor_data = mdata.toPandas()
faults_data = data1
faults_data
#ttf_data = ttfdata.toPandas()


# COMMAND ----------

x1 = faults_data.iloc[:, :5]
Y1 = faults_data.iloc[:, 5]

# COMMAND ----------

x1.head()

# COMMAND ----------

Y1.head()

# COMMAND ----------

Y1.value_counts()

# COMMAND ----------

# from imblearn.over_sampling import SMOTE
# oversample = SMOTE()
# X,Y = oversample.fit_resample(x1,Y1)
# counter = Counter(Y)
# print(counter)

# COMMAND ----------

x1.shape, Y1.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x1, Y1, test_size = 0.2, random_state = 0)

# COMMAND ----------

X_train.shape, X_test.shape

# COMMAND ----------

y_train.shape, y_test.shape

# COMMAND ----------

y_train.value_counts()

# COMMAND ----------


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # XGboost

# COMMAND ----------

start = time.time()
xgb = XGBClassifier(random_state=42)
xgb.fit(x_train,y_train)
xgbpreds = xgb.predict(x_test)
print("Time", time.time()-start)
print("Accuracy",accuracy_score(y_test,xgbpreds))
print(classification_report(y_test,xgbpreds))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction on unseen data

# COMMAND ----------

#Predicted value
xgbpreds=xgb.predict(test_independent)
xgbpreds

# COMMAND ----------

#Real value 
np.array(test_dependent_y)

# COMMAND ----------

print("Accuracy",accuracy_score(test_dependent_y,xgbpreds))
print(classification_report(test_dependent_y,xgbpreds))

# COMMAND ----------

# MAGIC %md
# MAGIC # Support Vector Machine

# COMMAND ----------

# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score

# instantiate classifier with default hyperparameters
svc=SVC() 
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred=svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction on unseen data

# COMMAND ----------

svc_test=svc.predict(test_independent)
svc_test

# COMMAND ----------

np.array(test_dependent_y)

# COMMAND ----------

print("Accuracy",accuracy_score(test_dependent_y,svc_test))
print(classification_report(test_dependent_y,svc_test))

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Tree classifier

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # save the model

# COMMAND ----------

import pickle

with open("./models_onSampleData/DT_Model.pkl","wb") as f:
    pickle.dump(clf_gini,f)


# COMMAND ----------

# MAGIC %md
# MAGIC # Load the model

# COMMAND ----------

with open("./models_onSampleData/DT_Model.pkl","rb") as files:
    model = pickle.load(files)

y_pred_gini = model.predict(X_test) 
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

# COMMAND ----------

y_pred_gini = clf_gini.predict(X_test)
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

# COMMAND ----------

y_pred_train_gini = clf_gini.predict(X_train)
# y_pred_train_gini
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction on unseen data

# COMMAND ----------

#testing on onseen data
save_model_result = model.predict(test_independent)
save_model_result

# COMMAND ----------

#testing on onseen data
clf_gini_test = clf_gini.predict(test_independent)
clf_gini_test

# COMMAND ----------

np.array(test_dependent_y)

# COMMAND ----------

#clf_gini.evaluate(clf_gini_test,test_dependent_y)
print("Accuracy",accuracy_score(test_dependent_y,clf_gini_test))
print(classification_report(test_dependent_y,clf_gini_test))

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction on original test data

# COMMAND ----------

test_org = pd.read_csv(r'C:\Users\Rajesh.Mandal\Downloads\initiative\classification\only_sensors_columns\Tests.csv')
test_org_X = test_org.iloc[:,1:-1]
test_org_y = test_org.iloc[:,-1]
test_org_y

# COMMAND ----------

test_org_X

# COMMAND ----------

test_org_X.shape, test_org_y.shape

# COMMAND ----------

# testing on onseen data

DT_original_test = clf_gini.predict(test_org_X)
DT_original_test

# COMMAND ----------

print("Accuracy",accuracy_score(test_org_y,DT_original_test))

# COMMAND ----------

from sklearn.metrics import f1_score

f1_score(test_org_y, DT_original_test,average='weighted')

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction on Generated Test Datasets from lstm model using decision tree

# COMMAND ----------

test_Gen = pd.read_csv(r'C:\Users\Rajesh.Mandal\Downloads\initiative\classification\only_sensors_columns\True_predictions (1).csv')
test_Gen_X = test_Gen.iloc[:,1:-1]
test_Gen_y = test_Gen.iloc[:,-1]
test_Gen_y

# COMMAND ----------

# testing on onseen data

DT_Generated_test = clf_gini.predict(test_Gen_X)
DT_Generated_test

# COMMAND ----------

from sklearn.metrics import f1_score
f1_score(test_org_y,DT_Generated_test,average='weighted')

# COMMAND ----------

print("Accuracy",accuracy_score(test_org_y,DT_Generated_test))

# COMMAND ----------

# MAGIC %md
# MAGIC # Naive Bayes

# COMMAND ----------

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)

# COMMAND ----------

y_pred = gnb.predict(X_test)

# y_pred
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# COMMAND ----------

y_pred_train = gnb.predict(X_train)
# y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction on unseen data

# COMMAND ----------

gnb_test = gnb.predict(test_independent)
gnb_test

# COMMAND ----------

np.array(test_dependent_y)

# COMMAND ----------

print("Accuracy",accuracy_score(test_dependent_y,gnb_test))
print(classification_report(test_dependent_y,gnb_test))

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0, criterion='entropy')
classifier.fit(X_train, y_train)


# COMMAND ----------

# MAGIC %md
# MAGIC # save the model

# COMMAND ----------

import pickle

with open("./models_onSampleData/RF_Model.pkl","wb") as f:
    pickle.dump(classifier,f)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the model

# COMMAND ----------

with open("./models_onSampleData/RF_Model.pkl","rb") as files:
    model = pickle.load(files)

y_pred_RF = model.predict(X_test) 
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_RF)))

# COMMAND ----------

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying K-fold Cross Validation APPLY TO ALL THE DATA, NOT JUST TRAINING DATA THIS WILL BIAS IT
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x1, y=Y1, cv=10)
# if working with a lot of data, you can set n_jobs to -1
accuracies.mean()
accuracies.std()

print("Test set classification rate: {}".format(np.mean(y_pred == y_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction on unseen data

# COMMAND ----------

RF_test = model.predict(test_independent)
RF_test

# COMMAND ----------

classifier.predict(test_independent)

# COMMAND ----------

np.array(test_dependent_y)

# COMMAND ----------

print("Accuracy",accuracy_score(test_dependent_y,RF_test))
print(classification_report(test_dependent_y,RF_test))

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction on original test data

# COMMAND ----------

test_org = pd.read_csv(r'C:\Users\Rajesh.Mandal\Downloads\initiative\classification\only_sensors_columns\Tests.csv')
test_org_X = test_org.iloc[:,1:-1]
test_org_y = test_org.iloc[:,-1]
test_org_y

# COMMAND ----------

RF_gtest = classifier.predict(test_org_X)
RF_gtest

# COMMAND ----------

from sklearn.metrics import f1_score

f1_score(test_org_y, RF_gtest,average='weighted')

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction on Generated Test Datasets using Random Forest

# COMMAND ----------

test_Gen = pd.read_csv(r'C:\Users\Rajesh.Mandal\Downloads\initiative\classification\only_sensors_columns\True_predictions (1).csv')
test_Gen_X = test_Gen.iloc[:,1:-1]
test_Gen_y = test_Gen.iloc[:,-1]
test_Gen_y

# COMMAND ----------

# testing on onseen data

DT_Generated_test = classifier.predict(test_Gen_X)
DT_Generated_test

# COMMAND ----------

from sklearn.metrics import f1_score
f1_score(test_org_y, DT_Generated_test,average='weighted')

# COMMAND ----------

# MAGIC %md
# MAGIC # Artificial Neural Network

# COMMAND ----------

#ANN
# sequential model to initialise our ann and dense module to build the layers
from keras.models import Sequential
from keras.layers import Dense

# COMMAND ----------

classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN | means applying SGD on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'SparseCategoricalCrossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 16, epochs = 300,verbose = 2)



# COMMAND ----------

score, acc = classifier.evaluate(X_train, y_train)
print('Train score:', score)
print('Train accuracy:', acc)
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

score, acc = classifier.evaluate(X_test, y_test,batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)


# COMMAND ----------

RF_test = classifier.predict(test_independent)
RF_test
# np.argmax(RF_test[[7]]) #returns an index of an higesht probability value

# COMMAND ----------

for i in RF_test:
    print(np.argmax(i), "=>", np.amax(i))

# COMMAND ----------

np.amax(RF_test[2])

# COMMAND ----------

r1 = np.argmax(classifier.predict([[-0.022398,0.532598402,0.221735768,-0.122454958,0.803526088]]))
r2 = np.argmax(classifier.predict([[0.011107734,0.581418603,0.207579792,-0.122454958,0.803526088]]))
r3 = np.argmax(classifier.predict([[-0.079523281,-0.939042679,-2.332806896,4.530130587,0.803526088]]))
r4 = np.argmax(classifier.predict([[-0.079514245,-0.939042679,-2.332806896,2.239236279,0.803526088]]))
r5 = np.argmax(classifier.predict([[0.032083252,1.462681135,-0.216699227,-0.122454958,-0.683189559]]))
r6 = np.argmax(classifier.predict([[0.031457236,1.64251418,-0.184553366,-0.122454958,0.021044168]]))
r7 = np.argmax(classifier.predict([[-0.079428347,-0.939042679,-2.332806896,-0.122437071,4.037784688]]))
r8 = np.argmax(classifier.predict([[-0.079418743,-0.939042679,-2.332806896,-0.122437071,4.037784688]]))
print(r1,r2,r3,r4,r5,r6,r7,r8)

# COMMAND ----------

label=['FlowCool Pressure Dropped Below Limit','Flowcool leak ','NaN','Flowcool Pressure Too High Check Flowcool Pump']

# COMMAND ----------

# creating initial dataframe
Fault_types = ('FlowCool Pressure Dropped Below Limit','Flowcool leak ','NaN','Flowcool Pressure Too High Check Flowcool Pump')
Fault_df = pd.DataFrame(Fault_types, columns=['Fault_Types'])
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
Fault_df['Fault_df_category'] = labelencoder.fit_transform(Fault_df['Fault_Types'])
Fault_df

# COMMAND ----------

from keras import regularizers
from keras.layers import Dropout
model_3 = Sequential([
    Dense(6, activation='relu',  input_shape=(5,)),
    
    Dense(6, activation='relu'),
    
    Dense(6, activation='relu'),
   
    #Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    #Dropout(0.3),
    Dense(4, activation='softmax'),
])
model_3.compile(optimizer='adam',
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])
              
hist_3 = model_3.fit(X_train, y_train,
          batch_size=16, epochs=50)

# COMMAND ----------


