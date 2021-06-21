#import modules and load the csv files given 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import string
import nltk
import re

data = pd.read_csv('/Users/ananyasharma/Downloads/usc-dsci552-section-32415d-spring-2021-ps5/ps5_tweets_text.csv')
data1 = pd.read_csv('/Users/ananyasharma/Downloads/usc-dsci552-section-32415d-spring-2021-ps5/ps5_tweets_labels.csv')
data2 = pd.read_csv('/Users/ananyasharma/Downloads/usc-dsci552-section-32415d-spring-2021-ps5/ps5_tweets_labels_as_numbers.csv')

#view the data in all 3 files 
print(data)
print(data1)
print(data2)

#view the dimensionality of this data
print(data.shape)
print(data1.shape)
print(data2.shape)


#visualise how the tweets are versus their labels
sns.countplot(x='Label', data=data2)


#merging the files into 1 dataframe
r1 = pd.merge(data,data1, on="Id")
r1
result = pd.merge(r1,data2, on="Id")
result


#Preprocessing the data
def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text


result['Tidy_Tweets'] = np.vectorize(remove_pattern)(result['Tweet'], "@[\w]*")
result.head()

# data cleaning: remove URL's from data
result['Tidy_Tweets'] = result['Tidy_Tweets'].apply(lambda x: re.sub(r'http\S+', '', x))

#replacing special characters, white spaces, etc.
result['Tidy_Tweets'] = result['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

#removing words of length less than 3
result['Tidy_Tweets'] = result['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
result.head(10)

# convert text to lowercase
result['Tidy_Tweets'] = result['Tidy_Tweets'].str.lower()
result


#Tokenize the data
tokenized_tweet = result['Tidy_Tweets'].apply(lambda x: x.split())

#stemming the data
from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
tokenized_tweet.head()

#putting it all back in the dataframe
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

result['Tidy_Tweets'] = tokenized_tweet
result.head()

#splitting the data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result['Tidy_Tweets'],result['Label'],random_state = 0)
                                    
X_train


#Implementation of Naive Bayes Classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
#Train and evaluate the model
vect = CountVectorizer().fit(X_train) #vectorizing the text i.e. text to numbers
X_train_vectorized = vect.transform(X_train)
clfrNB = MultinomialNB(alpha = 0.1)
clfrNB.fit(X_train_vectorized, y_train)
preds = clfrNB.predict(vect.transform(X_test))

#classification report that will give all metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,preds))
print('Accuracy on training set :{:.2f}'.format(clfrNB.score(X_train_vectorized,y_train)))
print('Accuracy on training set :{:.2f}'.format(clfrNB.score((vect.transform(X_test)),y_test)))


#before oversampling see the data
X_train.shape, y_train.shape


#Oversampling the data using SMOTE 
import imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train.values.ravel())
X_train=vectorizer.transform(X_train.values.ravel())
X_test=vectorizer.transform(X_test.values.ravel())
X_train=X_train.toarray()
X_test=X_test.toarray()
X_sm, y_sm = smote.fit_resample(X_train,y_train)

print(X_sm.shape,y_sm.shape)

y1 = y_sm.values


#Neural network model 1
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,GRU
from keras.layers.embeddings import Embedding

EMBEDDING_DIM=100
vocab_size = 38900
max_length = 25
num_labels=5

model=Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(GRU(units=16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
                       
#training the model 
history = model.fit(X_sm,y1,batch_size=128, epochs=10, verbose=1, validation_data=(X_test,y_test)) 

loss, accuracy, f1_score, precision, recall

#Plot training & validation accuracy values
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
    
    
#Neural Network model 2
from keras.layers import SpatialDropout1D

EMBEDDING_DIM = 100
MAX_NB_WORDS = 5000

model1 = Sequential()
model1.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=250))
model1.add(SpatialDropout1D(0.2))
model1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(13, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
batch_size = 64

print(model.summary())

history1 = model1.fit(X_sm, y1, epochs=epochs, batch_size=batch_size,validation_data=(X_test,y_test))

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
          





