from stanfordcorenlp import StanfordCoreNLP
import re
import pandas as pd
import numpy as np
import functools
import random

pcs_pairs = np.load('GOPCSMax.npy',allow_pickle=True)

#获得pcs_pairs的全部切片，放入concepts中，保证concepts改变的情况下pcs_pairs不变
concepts = pcs_pairs[:]


concepts_copy =[]
for i in range(len(concepts)):
    if type(concepts[i]) == np.ndarray:
        concepts_copy.append(list(concepts[i]))
    else:
        concepts_copy.append(concepts[i])

concepts = concepts_copy
pcs_slice = concepts[:]
pcs_pairs = concepts[:]
#pairs存储有父子串关系的关系对，pairs_child存储关系中的子，pairs_parent存储父
pairs_child = []
pairs_parent = []
pairs = []

#此模块的作用是把pcs_pairs中所有有父子串关系的元素找出来存储到pairs中
for i in range(len(concepts)):
    for j in range(len(concepts)):
        if i == j:
            continue
        if type(concepts[i]) == list and type(concepts[j]) == list :
            label = 0
            for x in range(len(concepts[i])):
                for y in range(len(concepts[j])):
                    a = concepts[i][x]
                    b = concepts[j][y]
                    a_split = a.split(' ')
                    b_split = b.split(' ')
                    if  bool(re.search(b,a,re.IGNORECASE)):
                        pairs_child.append(concepts[i])
                        pairs_parent.append(concepts[j])
                        pairs.append([concepts[i],concepts[j]])
                        label = 1
                        break
                    elif len(a_split) == len(b_split)+1:#b不是a的子串，但是a只比b多一个单词，且除了那个单词，其余单词都相同
                        for z in range(len(b_split)):
                            if a_split[z] != b_split[z]:
                                a_split.pop(z)
                                break
                        if  a_split==b_split:
                            pairs_child.append(concepts[i])
                            pairs_parent.append(concepts[j])
                            pairs.append([concepts[i],concepts[j]])
                            label = 1
                            break
                if label == 1:
                    break
        elif type(concepts[i]) == list and type(concepts[j]) != list:
            b = concepts[j]
            for x in range(len(concepts[i])):
                a = concepts[i][x]
                a_split = a.split(' ')
                b_split = b.split(' ')
                if bool(re.search(b,a,re.IGNORECASE)):
                    pairs_child.append(concepts[i])
                    pairs_parent.append(concepts[j])
                    pairs.append([concepts[i],concepts[j]])
                    break
                elif len(a_split) == len(b_split)+1:#b不是a的子串，但是a只比b多一个单词，且除了那个单词，其余单词都相同
                    for z in range(len(b_split)):
                        if a_split[z] != b_split[z]:
                            a_split.pop(z)
                            break
                    if  a_split==b_split:
                        pairs_child.append(concepts[i])
                        pairs_parent.append(concepts[j])
                        pairs.append([concepts[i],concepts[j]])
                        label = 1
                        break
        elif type(concepts[i]) != list and type(concepts[j]) == list:
            a = concepts[i]
            for x in range(len(concepts[j])):
                b = concepts[j][x]
                a_split = a.split(' ')
                b_split = b.split(' ')
                if bool(re.search(b,a,re.IGNORECASE)):
                    pairs_child.append(concepts[i])
                    pairs_parent.append(concepts[j])
                    pairs.append([concepts[i],concepts[j]])
                    break
                elif len(a_split) == len(b_split)+1:#b不是a的子串，但是a只比b多一个单词，且除了那个单词，其余单词都相同
                    for z in range(len(b_split)):
                        if a_split[z] != b_split[z]:
                            a_split.pop(z)
                            break
                    if  a_split==b_split:
                        pairs_child.append(concepts[i])
                        pairs_parent.append(concepts[j])
                        pairs.append([concepts[i],concepts[j]])
                        break
        elif type(concepts[i]) != list and type(concepts[j]) != list:
            a = concepts[i]
            b = concepts[j]
            a_split = a.split(' ')
            b_spilt = b.split(' ')
            if bool(re.search(b,a,re.IGNORECASE)):
                pairs_child.append(concepts[i])
                pairs_parent.append(concepts[j])
                pairs.append([concepts[i],concepts[j]])
            elif len(a_split) == len(b_split)+1:#b不是a的子串，但是a只比b多一个单词，且除了那个单词，其余单词都相同
                for z in range(len(b_split)):
                    if a_split[z] != b_split[z]:
                        a_split.pop(z)
                        break
                if  a_split==b_split:
                    pairs_child.append(concepts[i])
                    pairs_parent.append(concepts[j])
                    pairs.append([concepts[i],concepts[j]])    

pairs_parent_copy = pairs_parent[:]
pairs_child_copy = pairs_child[:]
pairs_copy = pairs[:]

for s in range(len(pairs)-1):
    if pairs[s] not in pairs_copy:
        continue
    f = s+1

    while f <= len(pairs)-1:
        if pairs[f] not in pairs_copy:
            f += 1
            continue

        if pairs[s][0] == pairs[f][0]:
            if type(pairs[s][1]) == list and type(pairs[f][1]) == list:
                label = 0
                for x in range(len(pairs[s][1])):
                    for y in range(len(pairs[f][1])):
                        a = pairs[s][1][x]
                        b = pairs[f][1][y]
                        a_split = a.lower().split(' ')
                        b_split = b.lower().split(' ')
                        if bool(re.search(a,b,re.IGNORECASE)):
                            label = 1
                            if pairs[s] in pairs_copy:                                
                                pairs_copy.remove(pairs[s])
                            break
                        elif bool(re.search(b,a,re.IGNORECASE)):
                            label = 1
                            if pairs[f] in pairs_copy:
                                pairs_copy.remove(pairs[f])
                            break
                        elif len(a_split) == len(b_split)+1:
                            for z in range(len(b_split)):
                                if a_split[z] != b_split[z]:
                                    a_split.pop(z)
                                    break
                            if a_split==b_split:
                                label = 1
                                if pairs[f] in pairs_copy:
                                    pairs_copy.remove(pairs[f])    
                                break
                        elif len(b_split) == len(a_split)+1:
                            for z in range(len(a_split)):
                                if b_split[z] != a_split[z]:
                                    b_split.pop(z)
                                    break
                            if a_split==b_split:
                                label = 1
                                if pairs[s] in pairs_copy:
                                    pairs_copy.remove(pairs[s])
                                break
                        if label == 1:
                            break
            elif type(pairs[s][1]) == list and type(pairs[f][1]) != list:
                b = pairs[f][1]
                for x in range(len(pairs[s][1])):
                    a = pairs[s][1][x]
                    a_split = a.lower().split(' ')
                    b_split = b.lower().split(' ')
                    if bool(re.search(a,b,re.IGNORECASE)):
                        if pairs[s] in pairs_copy:                                
                            pairs_copy.remove(pairs[s])
                        break
                    elif bool(re.search(b,a,re.IGNORECASE)):
                        if pairs[f] in pairs_copy:
                            pairs_copy.remove(pairs[f])
                        break
                    elif len(a_split) == len(b_split)+1:
                        for x in range(len(b_split)):
                            if a_split[x] != b_split[x]:
                                a_split.pop(x)
                                break
                        if a_split==b_split:
                            if pairs[f] in pairs_copy:
                                pairs_copy.remove(pairs[f])    
                            break
                    elif len(b_split) == len(a_split)+1:
                        for z in range(len(a_split)):
                            if b_split[z] != a_split[z]:
                                b_split.pop(z)
                                break
                        if a_split==b_split:
                            if pairs[s] in pairs_copy:
                                pairs_copy.remove(pairs[s]) 
                            break
            elif type(pairs[s][1]) != list and type(pairs[f][1]) == list:
                a = pairs[s][1]
                for y in range(len(pairs[f][1])):
                    b = pairs[f][1][y]
                    a_split = a.lower().split(' ')
                    b_split = b.lower().split(' ')
                    if bool(re.search(a,b,re.IGNORECASE)):
                        if pairs[s] in pairs_copy:                                
                            pairs_copy.remove(pairs[s])
                        break
                    elif bool(re.search(b,a,re.IGNORECASE)):
                        if pairs[f] in pairs_copy:
                            pairs_copy.remove(pairs[f])
                        break
                    elif len(a_split) == len(b_split)+1:
                        for x in range(len(b_split)):
                            if a_split[x] != b_split[x]:
                                a_split.pop(x)
                                break
                        if a_split==b_split:
                            if pairs[f] in pairs_copy:
                                pairs_copy.remove(pairs[f])    
                            break
                    elif len(b_split) == len(a_split)+1:
                        for x in range(len(a_split)):
                            if b_split[x] != a_split[x]:
                                b_split.pop(x)
                                break
                        if a_split==b_split:
                            if pairs[s] in pairs_copy:
                                pairs_copy.remove(pairs[s]) 
                            break
            elif type(pairs[s][0]) != list and type(pairs[f][0]) != list:
                a = pairs[s][0]
                b = pairs[f][0]
                a_split = a.lower().split(' ')
                b_split = b.lower().split(' ')
                if bool(re.search(a,b,re.IGNORECASE)):
                    if pairs[s] in pairs_copy:                                
                        pairs_copy.remove(pairs[s])
                    break
                elif bool(re.search(b,a,re.IGNORECASE)):
                    if pairs[f] in pairs_copy:
                        pairs_copy.remove(pairs[f])
                    break
                elif len(a_split) == len(b_split)+1:
                    for x in range(len(b_split)):
                        if a_split[x] != b_split[x]:
                            a_split.pop(x)
                            break
                    if a_split==b_split:
                        if pairs[f] in pairs_copy:
                            pairs_copy.remove(pairs[f])    

                elif len(b_split) == len(a_split)+1:
                    for x in range(len(a_split)):
                        if b_split[x] != a_split[x]:
                            b_split.pop(x)
                            break
                    if a_split==b_split:
                        if pairs[s] in pairs_copy:
                            pairs_copy.remove(pairs[s]) 
                        break     
        f += 1

            


child = []
parent = []
for i in range(len(pairs_copy)):
    child.append(pairs_copy[i][0])
    parent.append(pairs_copy[i][1])            

#找出所有的根节点
def FindAllRoot(child_array,parent_array):#形参分别为孩子节点列表，父母节点列表
    root_list = []#存储根节点的列表
#     parent_array = np.array(parent_array)
#     parent_array = np.unique(parent_array)#将父节点列表去重
    for i in range(len(parent_array)):
        if parent_array[i] not in child_array:#如果父节点列表中的某元素不在子节点中，则表示此元素为根节点
            root_list.append(parent_array[i])
    return root_list

root_list = FindAllRoot(child,parent)

level1 = []
for i in root_list:
    if i not in level1:
        level1.append(i)

levels = []
levels.append(level1)
levels_link = []
while True:
    level = levels[-1]
    temp1 = []
    temp2 = []
    count = 0
    for i in range(len(level)):
        for j in range(len(parent)):
#             if i == j:
#                 continue
            if level[i] == parent[j]:
                count += 1
                temp1.append(child[j])
                temp2.append([child[j],level[i]])
    if count == 0:
        break
    else:
        levels.append(temp1)
        levels_link.append(temp2)
        print(count)

levels_uni = []
for i in levels:
    temp = []
    for j in i:
        if j not in temp:
            temp.append(j)
    levels_uni.append(temp)

levels_link_uni = []
for i in levels_link:
    temp = []
    for j in i:
        if j not in temp:
            temp.append(j)
    levels_link_uni.append(temp)

threads = []
for i in levels_link_uni[0]:
    threads.append(i)

for i in range(len(levels_uni[1:-1])):
    temp = []
    threads_copy = threads[:]
    for j in levels_uni[i+1]:
        print(j)
        temp1 = []
        temp_up = []
        temp_down = []
        for x in range(len(threads)):
            if  threads[x][0] == j:
                temp_up.append(threads[x])
                threads_copy.remove(threads[x])
        for y in levels_link_uni[i+1]:
            if y[1] == j:
                temp_down.append(y)
        p = len(temp_up)
        q = len(temp_down)
        print(p,q)
        print('temp_up:',temp_up)
        print('temp_down',temp_down)
        if q == 0:
            temp1.extend(temp_up)
        elif p > q:
            dis = p-q
            for s in range(dis):
                temp_down.append(temp_down[s%q])
        elif q > p:
            dis = q-p
            for s in range(dis):
                temp_up.append(temp_up[s%p])
#         print('temp_up:',temp_up)
#         print('temp_down:',temp_down)
        if p > 0 and q > 0:
            for s in range(len(temp_up)):
                a = [temp_down[s][0],]
                a.extend(temp_up[s])
                temp1.append(a)
        temp.extend(temp1)
    threads = temp
    threads.extend(threads_copy)
    
threads_set = threads[:]
threads_set_all = []
for i in range(len(threads)):
    for j in range(len(threads[i])):
        threads_set_all.append(threads[i][j])
for i in range(len(concepts)):
    if concepts[i] not in threads_set_all:
        threads.append(concepts[i])


from bert_serving.client import BertClient
bc = BertClient()

def create_embeding(text):
    text =  text.split(' ') 
    if '' in text:
        text.remove('')
    length = 30 - len(text)
    zeros = []
    text_vectors = []
    for i in range(length):
        zeros.append(np.zeros(768))
    zeros.extend(bc.encode(text))   
    vectors = np.array(zeros)
    return vectors

from keras.models import load_model
model = load_model('GO_bert_lstm_pcs_max5.h5')

is_a = []
part_of = []
no_rela = []
has_sub = []

a_list = []
b_list = []
set_i = []
noass_is_a = []
noass_part_of = []
noass_norela = []

def PredictRelation(a,b):
    global part_of,is_a,no_rela
    text_vectors_a = create_embeding(a)
    text_vectors_b = create_embeding(b)
    text_vectors = text_vectors_a - text_vectors_b
    predict = model.predict(text_vectors.reshape((1,30,768)))
    predict = list(map(lambda x: x==max(x),predict)) * np.ones(shape=predict.shape)
    predict = predict.reshape(3)
    if predict[2] == 1. and [a,b] not in part_of:
        part_of.append([a,b])
    elif predict[1] == 1. and [a,b] not in is_a:
        is_a.append([a,b]) 
    elif predict[0] == 1. and [a,b] not in no_rela:
        no_rela.append([a,b])

def PredictNoRela(a,i):
    global noass_is_a,noass_part_of,noass_norela,is_a,part_of,set_i,has_sub,threads
    if a in set_i or a in has_sub:
        pass
    else:
        set_i.append(a)
        for j in range(len(threads)):
            if i == j:
                continue
            if type(threads[j]) == list:
                if threads[j] in pcs_pairs:
                    for w in range(len(threads[j])):
                        b = threads[j][w]
                        text_vectors_a = create_embeding(a)
                        text_vectors_b = create_embeding(b)
                        text_vectors = text_vectors_a - text_vectors_b
                        predict = model.predict(text_vectors.reshape((1, 30, 768)))
                        predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                        predict = predict.reshape(3)
                        if predict[2] == 1. and [a,b] not in part_of and [a,b] not in noass_part_of:
                            noass_part_of.append([a,b])
                        elif predict[1] == 1. and [a,b] not in is_a and [a,b] not in noass_is_a:
                            noass_is_a.append([a,b])
                        elif predict[0] == 1. and [a,b] not in no_rela:
                            noass_norela.append([a,b])  
                else:
                    if threads[i][-1] == threads[j][-1]:                        
                        print(threads[i][-1],threads[j][-1])
                        continue
                    label = 0
                    for x in range(len(threads[j])):
                        if type(threads[j][x]) == list:
                            for w in range(len(threads[j][x])):
                                b = threads[j][x][w]
                                text_vectors_a = create_embeding(a)
                                text_vectors_b = create_embeding(b)
                                text_vectors = text_vectors_a - text_vectors_b
                                predict = model.predict(text_vectors.reshape((1, 30, 768)))
                                predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                                predict = predict.reshape(3)
                                if predict[2] == 1. and [a,b] not in part_of and [a,b] not in noass_part_of:
                                    noass_part_of.append([a,b])
                                    label = 1
                                elif predict[1] == 1. and [a,b] not in is_a and [a,b] not in noass_is_a:
                                    noass_is_a.append([a,b])
                                    label = 1
                                elif predict[0] == 1. and [a,b] not in no_rela:
                                    noass_norela.append([a,b])
                            if label == 1:
                                break
                        else:
                            b = threads[j][x]
                            text_vectors_a = create_embeding(a)
                            text_vectors_b = create_embeding(b)
                            text_vectors = text_vectors_a - text_vectors_b
                            predict = model.predict(text_vectors.reshape((1, 30, 768)))
                            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                            predict = predict.reshape(3)
                            if predict[2] == 1. and [a,b] not in part_of and [a,b] not in noass_part_of:
                                noass_part_of.append([a,b])
                                break
                            elif predict[1] == 1. and [a,b] not in is_a and [a,b] not in noass_is_a:
                                noass_is_a.append([a,b])
                                break
                            elif predict[0] == 1. and [a,b] not in no_rela:
                                noass_norela.append([a,b])   
            else:
                b = threads[j]
                text_vectors_a = create_embeding(a)
                text_vectors_b = create_embeding(b)
                text_vectors = text_vectors_a - text_vectors_b
                predict = model.predict(text_vectors.reshape((1, 30, 768)))
                predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                predict = predict.reshape(3)
                if predict[2] == 1. and [a,b] not in part_of and [a,b] not in noass_part_of:
                    noass_part_of.append([a,b])
                elif predict[1] == 1. and [a,b] not in is_a and [a,b] not in noass_is_a:
                    noass_is_a.append([a,b])
                elif predict[0] == 1. and [a,b] not in no_rela:
                    noass_norela.append([a,b])  

done =[]
for i in range(len(threads_set)):
    j = 0
    while j < len(threads_set[i])-1:        
        if type(threads_set[i][j]) == list and type(threads_set[i][j+1]) == list:
            copy_j = threads_set[i][j][:]
            for x in range(len(threads_set[i][j])):
                for y in range(len(threads_set[i][j+1])):
                    a = threads_set[i][j][x]
                    b = threads_set[i][j+1][y]
                    print(x,y)
                    print(a,b)
                    a_split = a.lower().split(' ')
                    b_split = b.lower().split(' ')
                    if bool(re.search(b,a,re.IGNORECASE)):
                        has_sub.append(a)
                        if a in copy_j:
                            copy_j.remove(a)
                        PredictRelation(a,b)

                    elif len(a_split) == len(b_split)+1:
                        for z in range(len(b_split)):
                            if a_split[z] != b_split[z]:
                                a_split.pop(z)
                                break
                        if a_split==b_split:
                            print(a,b)
                            has_sub.append(a)
                            if a in copy_j:
                                copy_j.remove(a)
                            PredictRelation(a,b)
            if len(copy_j) != 0:
                for x in range(len(copy_j)):
                    PredictNoRela(copy_j[x],i)                    
                        
        elif type(threads_set[i][j]) == list and type(threads_set[i][j+1]) != list:
            b = threads_set[i][j+1]
            copy_j = threads_set[i][j][:]
            for x in range(len(threads_set[i][j])):
                a = threads_set[i][j][x]               
                a_split = a.lower().split(' ')
                b_split = b.lower().split(' ')
                if bool(re.search(b,a,re.IGNORECASE)):
                    print(a,b)
                    has_sub.append(a)
                    if a in copy_j:
                        copy_j.remove(a)
                    PredictRelation(a,b)
                elif len(a_split) == len(b_split)+1:
                    for z in range(len(b_split)):
                        if a_split[z] != b_split[z]:
                            a_split.pop(z)
                            break
                    if a_split==b_split:
                        print(a,b)
                        has_sub.append(a)
                        if a in copy_j:
                            copy_j.remove(a)
                        PredictRelation(a,b)  
            if len(copy_j) != 0:
                for x in range(len(copy_j)):
                    PredictNoRela(copy_j[x],i)
        elif type(threads_set[i][j]) != list and type(threads_set[i][j+1]) == list:
            a = threads_set[i][j]
            for y in range(len(threads_set[i][j+1])):
                b = threads_set[i][j+1][y]  
                a_split = a.lower().split(' ')
                b_split = b.lower().split(' ')
                if bool(re.search(b,a,re.IGNORECASE)):
                    print(a,b)
                    has_sub.append(a)
                    PredictRelation(a,b)
                elif len(a_split) == len(b_split)+1:
                    for z in range(len(b_split)):
                        if a_split[z] != b_split[z]:
                            a_split.pop(z)
                            break
                    if a_split==b_split:
                        print(a,b)
                        has_sub.append(a)
                        PredictRelation(a,b) 
        else:
            a = threads_set[i][j]
            b = threads_set[i][j+1]           
            PredictRelation(a,b) 
        j += 1
                        
            
threads_set_all = []
for i in range(len(threads_set)):
    for j in range(len(threads_set[i])):
        threads_set_all.append(threads_set[i][j])
for i in range(len(concepts)):
    if concepts[i] not in threads_set_all:
        threads_set.append(concepts[i])

for i in range(len(threads_set)):
    print(i/len(threads_set)) 
    
    if type(threads_set[i]) == list:
        if threads_set[i] in pcs_pairs:
            if threads_set[i] in set_i:
                continue
            else:
                set_i.append(threads_set[i])
        else:
            if threads_set[i][-1] in set_i:
                continue
            else:
                set_i.append(threads_set[i][-1])
    else:
        if threads_set[i] in set_i:
            continue
        else:
            set_i.append(threads_set[i][-1])
    for j in range(len(threads_set)):     
        if i == j:
            continue
        if type(threads_set[i]) == list and threads_set[i] not in pcs_pairs and type(threads_set[j]) ==list and threads_set[j] not in pcs_pairs:
            if threads_set[i][-1] == threads_set[j][-1]:
                continue
        if type(threads_set[i]) == list and type(threads_set[j]) == list:
            if threads_set[i] not in pcs_pairs and threads_set[j] not in pcs_pairs:
                label = 0
                for x in range(len(threads_set[j])):
                    if type(threads_set[i][-1]) == list and type(threads_set[j][x]) == list:
                        for q in range(len(threads_set[i][-1])):
                            for w in range(len(threads_set[j][x])):
                                a = threads_set[i][-1][q]
                                b = threads_set[j][x][w]
                                print(1,a,b)
                                text_vectors_a = create_embeding(a)
                                text_vectors_b = create_embeding(b)
                                text_vectors = text_vectors_a - text_vectors_b
                                predict = model.predict(text_vectors.reshape((1, 30, 768)))
                                predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                                predict = predict.reshape(3)
                                if predict[2] == 1.:
                                    if [a,b] not in part_of and [a,b] not in noass_part_of:
                                        noass_part_of.append([a,b])
                                    label = 1
                                elif predict[1] == 1.:
                                    if [a,b] not in is_a and [a,b] not in noass_is_a:
                                        noass_is_a.append([a,b])
                                    label = 1  
                                elif predict[0] == 1.:
                                    noass_norela.append([a,b])
                        if label == 1:
                            break
                    elif type(threads_set[i][-1]) == list and type(threads_set[j][x]) != list:
                        b = threads_set[j][x]
                        for q in range(len(threads_set[i][-1])):
                            a = threads_set[i][-1][q]
                            print(2,a,b)
                            text_vectors_a = create_embeding(a)
                            text_vectors_b = create_embeding(b)
                            text_vectors = text_vectors_a - text_vectors_b
                            predict = model.predict(text_vectors.reshape((1, 30, 768)))
                            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                            predict = predict.reshape(3)
                            if predict[2] == 1.:
                                if [a,b] not in part_of and [a,b] not in noass_part_of:
                                    noass_part_of.append([a,b])
                                label = 1
                            elif predict[1] == 1.:
                                if [a,b] not in is_a and [a,b] not in noass_is_a:
                                    noass_is_a.append([a,b])
                                label = 1  
                            elif predict[0] == 1. and [a,b] not in no_rela:
                                noass_norela.append([a,b]) 
                        if label == 1:
                            break
                    elif type(threads_set[i][-1]) != list and type(threads_set[j][x]) == list:
                        a = threads_set[i][-1]
                        for w in range(len(threads_set[j][x])):
                            b = threads_set[j][x][w]
                            print(3,a,b)
                            text_vectors_a = create_embeding(a)
                            text_vectors_b = create_embeding(b)
                            text_vectors = text_vectors_a - text_vectors_b
                            predict = model.predict(text_vectors.reshape((1, 30, 768)))
                            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                            predict = predict.reshape(3)
                            if predict[2] == 1.:
                                if [a,b] not in part_of and [a,b] not in noass_part_of:
                                    noass_part_of.append([a,b])
                                label = 1
                            elif predict[1] == 1.:
                                if [a,b] not in is_a and [a,b] not in noass_is_a:
                                    noass_is_a.append([a,b])
                                label = 1  
                            elif predict[0] == 1. and [a,b] not in no_rela:
                                noass_norela.append([a,b])  
                        if label == 1:
                            break
                    else:
                        a = threads_set[i][-1]
                        b = threads_set[j][x]
                        print(4,a,b)
                        text_vectors_a = create_embeding(a)
                        text_vectors_b = create_embeding(b)
                        text_vectors = text_vectors_a - text_vectors_b
                        predict = model.predict(text_vectors.reshape((1, 30, 768)))
                        predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                        predict = predict.reshape(3)
                        if predict[2] == 1.:
                            if [a,b] not in part_of and [a,b] not in noass_part_of:
                                noass_part_of.append([a,b])
                            label = 1
                        elif predict[1] == 1.:
                            if [a,b] not in is_a and [a,b] not in noass_is_a:
                                noass_is_a.append([a,b])
                            label = 1  
                        elif predict[0] == 1. and [a,b] not in no_rela:
                            noass_norela.append([a,b])                     
            elif threads_set[i] not in pcs_pairs and threads_set[j] in pcs_pairs:
                if type(threads_set[i][-1]) == list:
                    for q in range(len(threads_set[i][-1])):
                        for w in range(len(threads_set[j])):
                            a = threads_set[i][-1][q]
                            b = threads_set[j][x][w]
                            print(5,a,b)
                            text_vectors_a = create_embeding(a)
                            text_vectors_b = create_embeding(b)
                            text_vectors = text_vectors_a - text_vectors_b
                            predict = model.predict(text_vectors.reshape((1, 30, 768)))
                            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                            predict = predict.reshape(3)
                            if predict[2] == 1.:
                                if [a,b] not in part_of and [a,b] not in noass_part_of:
                                    noass_part_of.append([a,b])
                                label = 1
                            elif predict[1] == 1.:
                                if [a,b] not in is_a and [a,b] not in noass_is_a:
                                    noass_is_a.append([a,b])
                                label = 1  
                            elif predict[0] == 1. and [a,b] not in no_rela:
                                noass_norela.append([a,b])      
                elif type(threads_set[i][-1]) != list:
                    a = threads_set[i][-1]
                    for w in range(len(threads_set[j])):
                        b = threads_set[j][w]
                        print(6,a,b)
                        text_vectors_a = create_embeding(a)
                        text_vectors_b = create_embeding(b)
                        text_vectors = text_vectors_a - text_vectors_b
                        predict = model.predict(text_vectors.reshape((1, 30, 768)))
                        predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                        predict = predict.reshape(3)
                        if predict[2] == 1.:
                            if [a,b] not in part_of and [a,b] not in noass_part_of:
                                noass_part_of.append([a,b])
                            label = 1
                        elif predict[1] == 1.:
                            if [a,b] not in is_a and [a,b] not in noass_is_a:
                                noass_is_a.append([a,b])
                            label = 1  
                        elif predict[0] == 1. and [a,b] not in no_rela:
                            noass_norela.append([a,b])  
            elif threads_set[i] in pcs_pairs and threads_set[j] not in pcs_pairs:
                label = 0
                for x in range(len(threads_set[j])):
                    if type(threads_set[j][x]) == list:
                        for q in range(len(threads_set[i])):
                            for w in range(len(threads_set[j][x])):
                                a = threads_set[i][q]
                                b = threads_set[j][x][w]
                                print(7,a,b)
                                text_vectors_a = create_embeding(a)
                                text_vectors_b = create_embeding(b)
                                text_vectors = text_vectors_a - text_vectors_b
                                predict = model.predict(text_vectors.reshape((1, 30, 768)))
                                predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                                predict = predict.reshape(3)
                                if predict[2] == 1.:
                                    if [a,b] not in part_of and [a,b] not in noass_part_of:
                                        noass_part_of.append([a,b])
                                    label = 1
                                elif predict[1] == 1.:
                                    if [a,b] not in is_a and [a,b] not in noass_is_a:
                                        noass_is_a.append([a,b])
                                    label = 1  
                                elif predict[0] == 1. and [a,b] not in no_rela:
                                    noass_norela.append([a,b])  
                        if label ==  1:
                            break
                    elif type(threads_set[j][x]) != list:
                        b = threads_set[j][x]
                        for q in range(len(threads_set[i])):
                            a = threads_set[i][q]
                            print(8,a,b)
                            text_vectors_a = create_embeding(a)
                            text_vectors_b = create_embeding(b)
                            text_vectors = text_vectors_a - text_vectors_b
                            predict = model.predict(text_vectors.reshape((1, 30, 768)))
                            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                            predict = predict.reshape(3)
                            if predict[2] == 1.:
                                if [a,b] not in part_of and [a,b] not in noass_part_of:
                                    noass_part_of.append([a,b])
                                label = 1
                            elif predict[1] == 1.:
                                if [a,b] not in is_a and [a,b] not in noass_is_a:
                                    noass_is_a.append([a,b])
                                label = 1           
                            elif predict[0] == 1. and [a,b] not in no_rela:
                                noass_norela.append([a,b]) 
                        if label == 1:
                            break
            elif threads_set[i] in pcs_pairs and threads_set[j] in pcs_pairs:
                for q in range(len(threads_set[i])):
                    for w in range(len(threads_set[j])):
                        a = threads[i][q]
                        b = threads[j][w]
                        print(9,a,b)
                        text_vectors_a = create_embeding(a)
                        text_vectors_b = create_embeding(b)
                        text_vectors = text_vectors_a - text_vectors_b
                        predict = model.predict(text_vectors.reshape((1, 30, 768)))
                        predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                        predict = predict.reshape(3)
                        if predict[2] == 1.:
                            if [a,b] not in part_of and [a,b] not in noass_part_of:
                                noass_part_of.append([a,b])
                            label = 1
                        elif predict[1] == 1.:
                            if [a,b] not in is_a and [a,b] not in noass_is_a:
                                noass_is_a.append([a,b])
                            label = 1  
                        elif predict[0] == 1. and [a,b] not in no_rela:
                            noass_norela.append([a,b]) 
        elif type(threads_set[i]) == list and type(threads_set[j])!= list:
            if threads_set[i] in pcs_pairs:
                for q in range(len(threads_set[i])):
                    a = threads_set[i][q]
                    b = threads_set[j]
                    print(10,a,b)
                    text_vectors_a = create_embeding(a)
                    text_vectors_b = create_embeding(b)
                    text_vectors = text_vectors_a - text_vectors_b
                    predict = model.predict(text_vectors.reshape((1, 30, 768)))
                    predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                    predict = predict.reshape(3)
                    if predict[2] == 1.:
                        if [a,b] not in part_of and [a,b] not in noass_part_of:
                            noass_part_of.append([a,b])
                        label = 1
                    elif predict[1] == 1.:
                        if [a,b] not in is_a and [a,b] not in noass_is_a:
                            noass_is_a.append([a,b])
                        label = 1  
                    elif predict[0] == 1. and [a,b] not in no_rela:
                        noass_norela.append([a,b]) 
            else:
                if type(threads_set[i][-1]) == list:
                    for q in range(len(threads_set[i][-1])):
                        a = threads_set[i][-1][q]
                        b = threads_set[j]
                        text_vectors_a = create_embeding(a)
                        text_vectors_b = create_embeding(b)
                        text_vectors = text_vectors_a - text_vectors_b
                        predict = model.predict(text_vectors.reshape((1, 30, 768)))
                        predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                        predict = predict.reshape(3)
                        if predict[2] == 1.:
                            if [a,b] not in part_of and [a,b] not in noass_part_of:
                                noass_part_of.append([a,b])
                            label = 1
                        elif predict[1] == 1.:
                            if [a,b] not in is_a and [a,b] not in noass_is_a:
                                noass_is_a.append([a,b])
                            label = 1  
                        elif predict[0] == 1. and [a,b] not in no_rela:
                            noass_norela.append([a,b]) 
                else:
                    a = threads_set[i][-1]
                    b = threads_set[j]
                    print(12,a,b)
                    text_vectors_a = create_embeding(a)
                    text_vectors_b = create_embeding(b)
                    text_vectors = text_vectors_a - text_vectors_b
                    predict = model.predict(text_vectors.reshape((1, 30, 768)))
                    predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                    predict = predict.reshape(3)
                    if predict[2] == 1.:
                        if [a,b] not in part_of and [a,b] not in noass_part_of:
                            noass_part_of.append([a,b])
                        label = 1
                    elif predict[1] == 1.:
                        if [a,b] not in is_a and [a,b] not in noass_is_a:
                            noass_is_a.append([a,b])
                        label = 1  
                    elif predict[0] == 1. and [a,b] not in no_rela:
                        noass_norela.append([a,b])  
        elif type(threads_set[i]) != list and type(threads_set[j]) == list:
            a = threads_set[i]
            if threads_set[j] in pcs_pairs:
                for w in range(len(threads_set[j])):
                    b = threads_set[j][w]
                    print(13,a,b)
                    text_vectors_a = create_embeding(a)
                    text_vectors_b = create_embeding(b)
                    text_vectors = text_vectors_a - text_vectors_b
                    predict = model.predict(text_vectors.reshape((1, 30, 768)))
                    predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                    predict = predict.reshape(3)
                    if predict[2] == 1.:
                        if [a,b] not in part_of and [a,b] not in noass_part_of:
                            noass_part_of.append([a,b])
                        label = 1
                    elif predict[1] == 1.:
                        if [a,b] not in is_a and [a,b] not in noass_is_a:
                            noass_is_a.append([a,b])
                        label = 1  
                    elif predict[0] == 1. and [a,b] not in no_rela:
                        noass_norela.append([a,b])  
            else:
                label = 0
                for x in range(len(threads_set[j])):
                    if type(threads_set[j][x]) == list:
                        for w in range(len(threads_set[j][x])):
                            b = threads_set[j][x][w]
                            print(14,a,b)
                            text_vectors_a = create_embeding(a)
                            text_vectors_b = create_embeding(b)
                            text_vectors = text_vectors_a - text_vectors_b
                            predict = model.predict(text_vectors.reshape((1, 30, 768)))
                            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                            predict = predict.reshape(3)
                            if predict[2] == 1.:
                                if [a,b] not in part_of and [a,b] not in noass_part_of:
                                    noass_part_of.append([a,b])
                                label = 1
                            elif predict[1] == 1.:
                                if [a,b] not in is_a and [a,b] not in noass_is_a:
                                    noass_is_a.append([a,b])
                                label = 1                        
                            elif predict[0] == 1. and [a,b] not in no_rela:
                                noass_norela.append([a,b])
                        if label == 1:
                            break
                    else:
                        b = threads_set[j][x]
                        print(15,a,b)
                        text_vectors_a = create_embeding(a)
                        text_vectors_b = create_embeding(b)
                        text_vectors = text_vectors_a - text_vectors_b
                        predict = model.predict(text_vectors.reshape((1, 30, 768)))
                        predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
                        predict = predict.reshape(3)
                        if predict[2] == 1.:
                            if [a,b] not in part_of and [a,b] not in noass_part_of:
                                noass_part_of.append([a,b])
                            label = 1
                        elif predict[1] == 1.:
                            if [a,b] not in is_a and [a,b] not in noass_is_a:
                                noass_is_a.append([a,b])
                            label = 1  
                        elif predict[0] == 1. and [a,b] not in no_rela:
                            noass_norela.append([a,b])   
        else:
            a = threads_set[i]
            b = threads_set[j]
            print(16,a,b)
            text_vectors_a = create_embeding(a)
            text_vectors_b = create_embeding(b)
            text_vectors = text_vectors_a - text_vectors_b
            predict = model.predict(text_vectors.reshape((1, 30, 768)))
            predict = list(map(lambda x: x == max(x), predict)) * np.ones(shape=predict.shape)
            predict = predict.reshape(3)
            if predict[2] == 1.:
                if [a,b] not in part_of and [a,b] not in noass_part_of:
                    noass_part_of.append([a,b])
                label = 1
            elif predict[1] == 1.:
                if [a,b] not in is_a and [a,b] not in noass_is_a:
                    noass_is_a.append([a,b])
                label = 1  
            elif predict[0] == 1. and [a,b] not in no_rela:
                noass_norela.append([a,b])              

def write_result(data,name):
	with open(name,'w') as file:
		for i in data:
			file.write(i[0]+'    '+i[1]+'\n')

write_result(is_a,r'zuixin/GOPCSMaxNoRandomNDR_isa_lstm.txt')
write_result(part_of,r'zuixin/GOPCSMaxNoRandomNDR_partof_lstm.txt')
write_result(noass_is_a,r'zuixin/GOPCSMaxNoRandomNDR_noassisa_lstm.txt')
write_result(noass_part_of,r'zuixin/GOPCSMaxNoRandomNDR_noasspartof_lstm.txt')






