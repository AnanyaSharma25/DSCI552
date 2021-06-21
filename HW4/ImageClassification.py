#import libraries and modules

import numpy as np
import pandas as pd
from numpy import genfromtxt
from keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from keras import backend as K


#load data - images & labels
data = np.load('/Users/ananyasharma/Downloads/ps4_trainvalid_images.npy')
a = np.genfromtxt('/Users/ananyasharma/Downloads/ps4_trainvalid_labels.csv', delimiter=",", skip_header=1)

#know dimensions of the data 
data.shape, a.shape

#dropping the first column which is basically an index for the csv file
#this will help in one hot encoding
a = np.delete(a, 0, axis=1)
a.shape

#one hot encoding the label
a = to_categorical(a)
a.shape

#To shuffle data while keeping the correlation between data and labels the same, create an index
idx = list(range(len(a)))
np.random.shuffle(idx)
idx

#Split the indices using this index 
N = len(idx)
train_indices = idx[:int(N*0.8)]
valid_indices = idx[int(N*0.8):]


#Splitting data into training and validation sets 
train_x = data[train_indices]
valid_x = data[valid_indices]
train_y = a[train_indices]
valid_y = a[valid_indices]
train_x.shape , valid_x.shape, train_y.shape, valid_y.shape

#create a model for the image classifier with 7 layers 

num_output_classes = 3
input_img_size = (64, 64, 1)  # 64x64 image with 1 color channel
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_output_classes, activation="softmax"))
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)


#model summary and the layers 
model.summary()
model.layers

#fitting the model by training it on the training dataset 
batch_size = 128
epochs = 20
history  = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_x,valid_y))
          
loss, accuracy, f1_score, precision, recall

Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation Precision values
plt.plot(history.history['precision_m'])
plt.plot(history.history['val_precision_m'])
plt.title('Model Precision')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#Plot training & validation Recall values
plt.plot(history.history['recall_m'])
plt.plot(history.history['val_recall_m'])
plt.title('Model Recall')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plot training & validation F1 values
plt.plot(history.history['f1_m'])
plt.plot(history.history['val_f1_m'])
plt.title('Model F1')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#finding out parameters to measure model 
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

# compile the model
model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy, metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history = model.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(valid_x, valid_y, verbose=0)

#print the values
loss, accuracy, f1_score, precision, recall

#a classification report in order to see these metrics individually for each of the three classes. 
rounded_labels=np.argmax(valid_y, axis=1)
rounded_labels

x = classification_report(rounded_labels, predictions, target_names = ['Healthy (Class 0)','Pre-existing condition (Class 1)','Effusion/Mass (Class 2)'],output_dict=True)
x

#Tabulating the classification report
df = pd.DataFrame(x).transpose()
df


#model 2 with 6 layers
#create a model for the image classifier 

num_output_classes = 3
input_img_size = (64, 64, 1)  # 64x64 image with 1 color channel
model1 = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
#model1.add(Conv2D(64, (3, 3), activation="relu"))
model1.add(MaxPooling2D(pool_size=(2, 2)))
#model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(64, activation="relu"))
model1.add(Dropout(0.5))
model.add(Dense(num_output_classes, activation="softmax"))
model1.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)


#model summary and the layers 
model1.summary()
model1.layers

#fitting the model by training it on the training dataset 

batch_size = 128
epochs = 20

history1  = model1.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_x,valid_y))
          


#Plot training & validation accuracy values
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot training & validation Precision values
plt.plot(history1.history['precision_m'])
plt.plot(history1.history['val_precision_m'])
plt.title('Model Precision')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#Plot training & validation Recall values
plt.plot(history1.history['recall_m'])
plt.plot(history1.history['val_recall_m'])
plt.title('Model Recall')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plot training & validation F1 values
plt.plot(history1.history['f1_m'])
plt.plot(history1.history['val_f1_m'])
plt.title('Model F1')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
          


#classification report 
# compile the model
model1.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy, metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history1 = model1.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model1.evaluate(valid_x, valid_y, verbose=0)


#now getting a report stating the testing parameters for the model
predictions = model1.predict_classes(valid_x)

#as the labels are still hot encoded, need to transform it into a 1D array first
rounded_labels=np.argmax(valid_y, axis=1)
rounded_labels

x = classification_report(rounded_labels, predictions, target_names = ['Healthy (Class 0)','Pre-existing condition (Class 1)','Effusion/Mass (Class 2)'],output_dict=True)
#in tabular form 
df1 = pd.DataFrame(x).transpose()
df1



#Model 3 with 4 layers
#create a model for the image classifier 

num_output_classes = 3
input_img_size = (64, 64, 1)  # 64x64 image with 1 color channel
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
#model2.add(Conv2D(64, (3, 3), activation="relu"))
model2.add(MaxPooling2D(pool_size=(2, 2)))
#model2.add(Dropout(0.25))
model2.add(Flatten())
#model2.add(Dense(64, activation="relu"))
#model2.add(Dropout(0.5))
model2.add(Dense(num_output_classes, activation="softmax"))
model2.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)


#model summary and the layers 
model2.summary()
model2.layers

#fitting the model by training it on the training dataset 

batch_size = 128
epochs = 20

history2  = model2.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_x,valid_y))

#Plot training & validation accuracy values
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Recall
pyplot.subplot(211)
pyplot.title('Recall')
pyplot.plot(history2.history['recall_m'], label='train')
pyplot.plot(history2.history['val_recall_m'], label='test')
pyplot.legend()

#Precision
pyplot.subplot(211)
pyplot.title('Precision')
pyplot.plot(history2.history['precision_m'], label='train')
pyplot.plot(history2.history['val_precision_m'], label='test')
pyplot.legend()

#F1 score
pyplot.subplot(211)
pyplot.title('F1')
pyplot.plot(history2.history['f1_m'], label='train')
pyplot.plot(history2.history['val_f1_m'], label='test')
pyplot.legend()

# compile the model
model2.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy, metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history2 = model2.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(valid_x, valid_y, verbose=0)
loss, accuracy, f1_score, precision, recall

#now getting a report stating the testing parameters for the model
predictions = model2.predict_classes(valid_x)

#as the labels are still hot encoded, need to transform it into a 1D array first
rounded_labels=np.argmax(valid_y, axis=1)
rounded_labels

x2 = classification_report(rounded_labels, predictions, target_names = ['Healthy (Class 0)','Pre-existing condition (Class 1)','Effusion/Mass (Class 2)'],output_dict=True)
#in tabular form 
df2 = pd.DataFrame(x).transpose()
df2

#testing the kaggle dataset on the first model
b = np.load('/Users/ananyasharma/Downloads/ps4_kaggle_images.npy')
prediction = model.predict(b)
prediction.shape

#converting into 1D array
labels = np.argmax(prediction, axis=-1)    
print(labels)

#saving labels as CSV file
np.savetxt('/Users/ananyasharma/Desktop/t2.csv', labels, delimiter=',', fmt='%d')



