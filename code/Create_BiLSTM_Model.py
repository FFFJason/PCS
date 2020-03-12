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

data_bert = np.load(bert_data,allow_pickle=True)#.npy file

def load_file(path,A,B):
    dataFrame = pd.read_csv(path)
    text_A = dataFrame[A]
    text_B = dataFrame[B]
    return text_A,text_B

child,parent = load_file(path,A,B)
n_child,n_parent = load_file(ndrdata_path,A,B)
test_all = list(np.load(data_path,allow_pickle=True))

pos_vects_a = []
pos_vects_b = []
for i in range(len(child)):
    num_a = 20 - len(child[i])
    num_b = 20 - len(parent[i])
    zeros = []
    for j in range(num_a):
        zeros.append(np.zeros(768))
    zeros.extend(data_bert[test_all.index(child[i])])
    pos_vects_a.append(zeros)
    zeros = []
    for j in range(num_b):
        zeros.append(np.zeros(768))
    zeros.extend(data_bert[test_all.index(parent[i])])
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
    zeros.extend(data_bert[test_all.index(n_child[i])])
    neg_vects_a.append(zeros)
    zeros = []
    for j in range(num_b):
        zeros.append(np.zeros(768))
    zeros.extend(data_bert[test_all.index(n_parent[i])])
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


labels = [1]*number+[0]*number
labels = to_categorical(np.asarray(labels))


x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.1)


model = Sequential()
model.add(Bidirectional(LSTM(32,return_sequences=True),merge_mode='concat'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=50,batch_size=512,validation_data=(x_test,y_test))
model.save(model_name)
y_pred = model.predict(x_test)
y_pred = list(map(lambda x: x==max(x),y_pred)) * np.ones(shape=y_pred.shape)
y_pred1 = []
y_test1 = []
for i in range(len(y_pred)):    
    if y_pred[i].reshape(2)[0] == 1.:
        y_pred1.append(0)
    elif y_pred[i].reshape(2)[1] == 1.:
        y_pred1.append(1)
    # elif y_pred[i].reshape(3)[2] == 1:
    #     y_pred1.append(2)
for i in range(len(y_test)):
    if y_test[i].reshape(2)[0] == 1.:
        y_test1.append(0)
    elif y_test[i].reshape(2)[1] == 1.:
        y_test1.append(1)
    # elif y_test[i].reshape(3)[2] == 1:
    #     y_test1.append(2)

print(sklearn.metrics.confusion_matrix(y_test1,y_pred1))
print(sklearn.metrics.classification_report(y_test1, y_pred1))

fp = 0
tp = 0
fn = 0

for i in range(len(y_pred)):
    if y_pred1[i]== 1 and y_test1[i]==1:
        tp+=1
    elif y_pred1[i]== 1 and y_test1[i]==0 :
        fp+=1
    elif y_pred1[i] == 0 and y_test1[i]==1:
        fn+=1

recall_s = tp/(tp+fn)
precision_s = tp/(tp+fp)
f1 = (recall_s*precision_s*2)/(recall_s+precision_s)
print('recall',recall_s)
print('precision',precision_s)
print('f1',f1)
