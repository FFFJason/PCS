import logging
import gensim
from gensim.models import word2vec
import pandas as pd
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import models,layers
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Input,Dense
from keras.models import Model,Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D,BatchNormalization,TimeDistributed,Bidirectional
from keras.layers import concatenate,Flatten,Dropout,LSTM
import keras
from keras import metrics
from keras.layers import  Lambda, TimeDistributed, Bidirectional
from keras import backend as K
import sklearn

'''def recall0(y, y1):
    a = [1.,0.,0.]
    a = np.array(a)
    y_true = y[:]
    y_pred = y1[:]
#    y_true = np.array(y_true)
    y_true = y_true*a
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision0(y, y1):
    a = [1.,0.,0.]
    a = np.array(a)
    y_true = y[:]
    y_pred = y1[:]
#    y_true = np.array(y_true)
    y_true = y_true*a
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall1(y, y1):
    a = [0.,1.,0.]
    a = np.array(a)
    y_true = y[:]
    y_pred = y1[:]
#    y_true = np.array(y_true)
    y_true = y_true*a
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision1(y, y1):
    a = [0.,1.,0.]
    a = np.array(a)
    y_true = y[:]
    y_pred = y1[:]
#    y_true = np.array(y_true)
    y_true = y_true*a
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall2(y, y1):
    a = [0.,0.,1.]
    a = np.array(a)
    y_true = y[:]
    y_pred = y1[:]
#    y_true = np.array(y_true)
    y_true = y_true*a
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision2(y, y1):
    a = [0.,0.,1.]
    a = np.array(a)
    y_true = y[:]
    y_pred = y1[:]
#    y_true = np.array(y_true)
    y_true = y_true*a
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision'''
def recall(y_true, y_pred):
    print(y_true,y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_score(recall,precision):
    f1_score = (recall*precision*2)/(recall+precision)
    return f1_score

data = np.load('data_123.npy')


# rib_a = np.load('Rib_a_vectors.npy')
# rib_b = np.load('Rib_b_vectors.npy')

# for i in range(len(rib_a)):
# num = 20 - len(rib_a[i])
# zeros = []
# for j in range(num):
#     zeros.append(np.zeros(768))
# zeros.extend(rib_a[i])
# rib_a[i] = zeros
# rib_a_all = np.zeros((len(rib_a),20,768))
# for i in range(len(rib_a)):
# for j in range(len(rib_a[i])):
#     rib_a_all[i][j] = rib_a[i][j]
# rib_a = None
# for i in range(len(rib_b)):
# num = 20 - len(rib_b[i])
# zeros = []
# for j in range(num):
#     zeros.append(np.zeros(768))
# zeros.extend(rib_b[i])
# rib_b[i] = zeros

# rib_b_all = np.zeros((len(rib_b),20,768))

# for i in range(len(rib_b)):
# for j in range(len(rib_b[i])):
#     rib_b_all[i][j] = rib_b[i][j]
# rib_b = None

# rib_data = rib_a_all - rib_b_all
# rib_a_all = None
# rib_b_all = None


labels = [1]*104594+[2]*61703+[0]*133703
labels = to_categorical(np.asarray(labels))
# rib_labels = [1]*346 + [2]*651 + [0]*812
# rib_labels = to_categorical(np.asarray(rib_labels))

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.1)

# x_train = np.concatenate((x_train,rib_data),axis = 0)
# y_train = np.concatenate((y_train,rib_labels),axis = 0)


model = Sequential()
#model.add(embedding_layer)
model.add(Bidirectional(LSTM(32,return_sequences=True),merge_mode='concat'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=1,batch_size=512,validation_data=(x_test,y_test))
model.save('asdfasel_1.h5')
y_pred = model.predict(x_test)
y_pred = list(map(lambda x: x==max(x),y_pred)) * np.ones(shape=y_pred.shape)
y_pred1 = []
y_test1 = []
for i in range(len(y_pred)):    
    if y_pred[i].reshape(3)[0] == 1.:
        y_pred1.append(0)
    elif y_pred[i].reshape(3)[1] == 1.:
        y_pred1.append(1)
    elif y_pred[i].reshape(3)[2] == 1:
        y_pred1.append(2)
for i in range(len(y_test)):
    if y_test[i].reshape(3)[0] == 1.:
        y_test1.append(0)
    elif y_test[i].reshape(3)[1] == 1.:
        y_test1.append(1)
    elif y_test[i].reshape(3)[2] == 1:
        y_test1.append(2)

print(sklearn.metrics.confusion_matrix(y_test1,y_pred1))
print(sklearn.metrics.classification_report(y_test1, y_pred1))

fp = 0
tp = 0
fn = 0

for i in range(len(y_pred)):
    if (y_pred1[i]== 1 and y_test1[i]==1 ) or (y_pred1[i]==2 and y_test1[i]==2):
        tp+=1
    elif (y_pred1[i]== 1 or y_pred1[i]==2) and y_test1[i]==0 :
        fp+=1
    elif y_pred1[i] == 0 and (y_test1[i]==1 or y_test1[i]==2):
        fn+=1

recall_s = tp/(tp+fn)
precision_s = tp/(tp+fp)
f1 = (recall_s*precision_s*2)/(recall_s+precision_s)
print('recall',recall_s)
print('precision',precision_s)
print('f1',f1)
