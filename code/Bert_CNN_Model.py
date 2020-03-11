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

data_a = np.load('FMA1AllDataUniqueBert.npy',allow_pickle=True)

def load_file(path,A,B):
    dataFrame = pd.read_csv(path)
    text_A = dataFrame[A]
    text_B = dataFrame[B]
    return text_A,text_B

child,parent = load_file('FMA1AllData.csv','Child','Parent')
n_child,n_parent = load_file('All_NDR_3Times1.csv','Child','Parent')
test_all = list(np.load('FMA1AllDataUnique.npy',allow_pickle=True))

pos_vects_a = []
pos_vects_b = []
for i in range(len(child)):
    num_a = 20 - len(child[i])
    num_b = 20 - len(parent[i])
    zeros = []
    for j in range(num_a):
        zeros.append(np.zeros(768))
    zeros.extend(data_a[test_all.index(child[i])])
    pos_vects_a.append(zeros)
    zeros = []
    for j in range(num_b):
        zeros.append(np.zeros(768))
    zeros.extend(data_a[test_all.index(parent[i])])
    pos_vects_b.append(zeros)

data_a_all = np.zeros((len(pos_vects_a),20,768),dtype='float32')
for i in range(len(pos_vects_a)):
    for j in range(len(pos_vects_a[i])):
        data_a_all[i][j] = pos_vects_a[i][j]
data_b_all = np.zeros((len(pos_vects_b),20,768),dtype='float32')
for i in range(len(pos_vects_b)):
    for j in range(len(pos_vects_b[i])):
        data_b_all[i][j] = pos_vects_b[i][j]

neg_vects_a = []
neg_vects_b = []
for i in range(len(n_child)):
    num_a = 20 - len(n_child[i])
    num_b = 20 - len(n_parent[i])
    zeros = []
    for j in range(num_a):
        zeros.append(np.zeros(768))
    zeros.extend(data_a[test_all.index(n_child[i])])
    neg_vects_a.append(zeros)
    zeros = []
    for j in range(num_b):
        zeros.append(np.zeros(768))
    zeros.extend(data_a[test_all.index(n_parent[i])])
    neg_vects_b.append(zeros)
data_a_neg = np.zeros((len(neg_vects_a),20,768),dtype='float32')
for i in range(len(neg_vects_a)):
    for j in range(len(neg_vects_a[i])):
        data_a_neg[i][j] = neg_vects_a[i][j]
data_b_neg = np.zeros((len(neg_vects_b),20,768),dtype='float32')
for i in range(len(neg_vects_b)):
    for j in range(len(neg_vects_b[i])):
        data_b_neg[i][j] = neg_vects_b[i][j]
pos_vects = data_a_all - data_b_all
neg_vects = data_a_neg - data_b_neg
del data_a_all,data_b_all,data_a_neg,data_b_neg
data = []
data.extend(pos_vects)
data.extend(neg_vects)
data = np.array(data)
del pos_vects,neg_vects
# for i in range(len(data_a)):
#     num = 20 - len(data_a[i])
#     zeros = []
#     for j in range(num):
#         zeros.append(np.zeros(768))
#     zeros.extend(data_a[i])
#     data_a[i] = zeros
# data_a_all = np.zeros((len(data_a),20,768),dtype='float32')
# for i in range(len(data_a)):
#     for j in range(len(data_a[i])):
#         data_a_all[i][j] = data_a[i][j]
# print('a_done!')
# del data_a
# for i in range(len(data_b)):
#     num = 20 - len(data_b[i])
#     zeros = []
#     for j in range(num):
#         zeros.append(np.zeros(768))
#     zeros.extend(data_b[i])
#     data_b[i] = zeros

# data_b_all = np.zeros((len(data_b),20,768),dtype='float32')
# for i in range(len(data_b)):
#     for j in range(len(data_b[i])):
#         data_b_all[i][j] = data_b[i][j]
# del data_b
# print('b_done!')
# data = data_a_all - data_b_all
# del data_a_all 
# del data_b_all 
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

print(data.shape)
labels = [1]*104488+[2]*61375+[0]*497006
# labels = [1]*104488+[2]*104488
labels = to_categorical(np.asarray(labels))
# rib_labels = [1]*346 + [2]*651 + [0]*812
# rib_labels = to_categorical(np.asarray(rib_labels))

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.1)

# x_train = np.concatenate((x_train,rib_data),axis = 0)
# y_train = np.concatenate((y_train,rib_labels),axis = 0)

model = Sequential()
model.add(Conv1D(256, 5, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(128, 5, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(64, 3, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(32, 3, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(16, 3, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(8, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(BatchNormalization())  
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=50,batch_size=512,validation_data=(x_test,y_test))
model.save('FMA1CNN_AllWithRandomNdr_Model.h5')
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
# for i in range(len(y_pred)):
#     if y_pred1[i]== 1 and y_test1[i]==1:
#         tp+=1
#     elif y_pred1[i]== 1 and y_test1[i]==0 :
#         fp+=1
#     elif y_pred1[i] == 0 and y_test1[i]==1:
#         fn+=1
recall_s = tp/(tp+fn)
precision_s = tp/(tp+fp)
f1 = (recall_s*precision_s*2)/(recall_s+precision_s)
print('recall',recall_s)
print('precision',precision_s)
print('f1',f1)
