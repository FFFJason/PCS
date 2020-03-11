import pandas as pd
import numpy as np 

def load_file(path,A,B):
    dataFrame = pd.read_csv(path)
    text_A = dataFrame[A]
    text_B = dataFrame[B]
    return text_A,text_B

child,parent = load_file('FMA1AllData.csv','Child','Parent')

test_a = np.array(child)
test_b = np.array(parent)
test_all = np.append(test_a,test_b)
test_all = np.unique(test_all)
np.save('FMA1AllDataUnique.npy',test_all)

split_data_a = []

for i in test_all:
    split_data_a.append(i.split(' '))

from bert_serving.client import BertClient

bc = BertClient()

data_a_vectors = []
num = len(split_data_a)
i = 0
for j in split_data_a:
    if i%1000 == 0:
        print(i/num)

    try:
        data_a_vectors.append(bc.encode(j))
    except ValueError:
        j.remove('')
        data_a_vectors.append(bc.encode(j))
    i += 1

np.save('FMA1AllDataUniqueBert_pre.npy',data_a_vectors)





