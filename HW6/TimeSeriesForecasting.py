#import modules and load the csv files given 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import string
import nltk
import tensorflow as tf
import re

d = pd.read_csv('/Users/ananyasharma/Downloads/usc-dsci552-section-32415d-spring-2021-ps6/ps6_trainvalid.csv')
d


#For data visualisation of each column
titles = [
    "Temperature",
    "Humidity",
    "Pressure",
    "Wind direction",
    "Wind speed",
]

feature_keys = [
    "temperature",
    "humidity",
    "pressure",
    "wind_direction",
    "wind_speed",
    
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
]

date_time_key = "datetime"


def show_raw_visualization(data):
    time_data = d[date_time_key]
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = d[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()


show_raw_visualization(d)

#To show correlation between features
def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

show_heatmap(d)


#New dataframe to hold only temperature
df = d[['datetime', 'temperature']]
#first row is NaN
df= df.drop(df.index[0])
#setting index to datetime
df.set_index("datetime",inplace=True)
#removing empty values
df = df.dropna(subset=["temperature"])
df

#To reduce noise, a 30 days temperature data rolled together
df_mean = df.temperature.rolling(window=30).mean()
df_mean.plot(figsize=(20,15))

#To reduce noise, a 120 days temperature data rolled together
df_mean = df.temperature.rolling(window=120).mean()
df_mean.plot(figsize=(20,15))


#remove NaN value row
d = d.drop(d.index[0])
#set index to datetime
d.set_index("datetime",inplace=True)
#drop the weather feature
d = d.drop('weather', 1)
#drop all empty values
d = d.dropna(axis = 0, how ='any')
d

from sklearn.preprocessing import MinMaxScaler


dataset = d.temperature.values #numpy.ndarray
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


#for model 1

#split the data into test and train
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
    
# reshape into X=t and Y=t+1
look_back = 30
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#model 1
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout,Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

model.summary()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
#Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
#Y_test = scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#for model 2
#splitting the dataset
from sklearn.preprocessing import MinMaxScaler


dataset1 = d.values #numpy.ndarray
dataset1 = dataset1.astype('float32')
dataset1 = np.reshape(dataset1, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1)
train_size_1 = int(len(dataset1) * 0.80)
test_size_1 = len(dataset1) - train_size
train_1, test_1 = dataset[0:train_size_1,:], dataset[train_size_1:len(dataset),:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
    
# reshape into X=t and Y=t+1
look_back = 30
X_train_1, Y_train_1 = create_dataset(train_1, look_back)
X_test_1, Y_test_1 = create_dataset(test_1, look_back)

# reshape input to be [samples, time steps, features]
X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], 1, X_train_1.shape[1]))
X_test_1 = np.reshape(X_test_1, (X_test.shape_1[0], 1, X_test.shape_1[1]))

#model2
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout,Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

model1 = Sequential()
model1.add(LSTM(100, input_shape=(X_train_1.shape[1], X_train_1.shape[2])))
model1.add(Dropout(0.2))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')

history1 = model1.fit(X_train_1, Y_train_1, epochs=20, batch_size=70, validation_data=(X_test_1, Y_test_1), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

model1.summary()

plt.figure(figsize=(8,4))
plt.plot(history1.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()


#splitting data for models 3 and 4
train_d,test_d=d[1:35735],d[35735:]
train_d
test_d

from sklearn.preprocessing import MinMaxScaler

train_2 = train_d
scalers={}
for i in train_d.columns:
    #normalizing the data between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train_2[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train_2[i]=s_s
test_2 = test_d
for i in train_d.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test_2[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test_2[i]=s_s
    
#See the data now    
train_2
test_2

#splitting data into series for model predictions
def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations 
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

#past 10 days observed to predict 
n_past = 10
#data for next 5 days
n_future = 5 
#5 columns to be considered
n_features = 5 

#splitting into x and y of training and test
X_train_2, y_train_2 = split_series(train_2.values,n_past, n_future)
X_train_2 = X_train_2.reshape((X_train_2.shape[0], X_train_2.shape[1],n_features))
y_train_2 = y_train_2.reshape((y_train_2.shape[0], y_train_2.shape[1], n_features))
X_test_2, y_test_2 = split_series(test_2.values,n_past, n_future)
X_test_2 = X_test_2.reshape((X_test_2.shape[0], X_test_2.shape[1],n_features))
y_test_2 = y_test_2.reshape((y_test_2.shape[0], y_test_2.shape[1], n_features))

#view the training data
X_train_2
y_train_2


#model 3
# E1D1
# n_features ==> no of features at each timestep in the data.
#
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)

encoder_states1 = encoder_outputs1[1:]

#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

#
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

#
model_e1d1.summary()


#model 4
# E2D2
# n_features ==> no of features at each timestep in the data.
#
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
#
model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
#
model_e2d2.summary()

#metrics for the 2 models and fitting and evaluating them
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(),metrics=['acc',f1_m,precision_m, recall_m])
history_e1d1=model_e1d1.fit(X_train_2,y_train_2,epochs=25,validation_data=(X_test_2,y_test_2),batch_size=32,verbose=0,callbacks=[reduce_lr])
model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(),metrics=['acc',f1_m,precision_m, recall_m])
history_e2d2=model_e2d2.fit(X_train_2,y_train_2,epochs=25,validation_data=(X_test_2,y_test_2),batch_size=32,verbose=0,callbacks=[reduce_lr])

loss_1, accuracy_1, f1_score_1, precision_1, recall_1 = model_e1d1.evaluate(X_test_2,y_test_2, verbose=0)
loss_2, accuracy_2, f1_score_2, precision_2, recall_2 = model_e2d2.evaluate(X_test_2,y_test_2, verbose=0)

#Plotting graphs for model 3
# Plot training & validation loss values
plt.plot(history_e1d1.history['loss'])
plt.plot(history_e1d1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation precision values
plt.plot(history_e1d1.history['precision_m'])
plt.plot(history_e1d1.history['val_precision_m'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history_e1d1.history['acc'])
plt.plot(history_e1d1.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation F1 values
plt.plot(history_e1d1.history['f1_m'])
plt.plot(history_e1d1.history['val_f1_m'])
plt.title('Model F1 Score')
plt.ylabel('F1 score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot training & validation recall values
plt.plot(history_e1d1.history['recall_m'])
plt.plot(history_e1d1.history['val_recall_m'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#print the metrics
recall_1
f1_score_1
accuracy_1
precision_1
loss_1

#Plotting graphs for model 4
# Plot training & validation loss values
plt.plot(history_e2d2.history['loss'])
plt.plot(history_e2d2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation precision values
plt.plot(history_e2d2.history['precision_m'])
plt.plot(history_e2d2.history['val_precision_m'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history_e2d2.history['acc'])
plt.plot(history_e2d2.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation F1 values
plt.plot(history_e2d2.history['f1_m'])
plt.plot(history_e2d2.history['val_f1_m'])
plt.title('Model F1 Score')
plt.ylabel('F1 score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot training & validation recall values
plt.plot(history_e2d2.history['recall_m'])
plt.plot(history_e2d2.history['val_recall_m'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#print the metrics
recall_2
f1_score_2
accuracy_2
precision_2
loss_2


#Prediction by model 3
aa=[x for x in range(200)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:200], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Weather', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

