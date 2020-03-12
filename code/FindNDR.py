from stanfordcorenlp import StanfordCoreNLP
import re
import pandas as pd
import numpy as np
import functools
import random


def load_file(path,A,B):
    dataFrame = pd.read_csv(path)
    text_A = dataFrame[A]
    text_B = dataFrame[B]
    return text_A,text_B
child,parent = load_file(path,A,B)

child = []
parent = []
for i in range(len(data)):
    child.append(data[i][0])
    parent.append(data[i][1])

#找出所有的根节点
def FindAllRoot(child_array,parent_array):#形参分别为孩子节点列表，父母节点列表
    root_list = []#存储根节点的列表
    parent_array = np.array(parent_array)
    parent_array = np.unique(parent_array)#将父节点列表去重
    for i in range(len(parent_array)):
        if parent_array[i] not in child_array:#如果父节点列表中的某元素不在子节点中，则表示此元素为根节点
            root_list.append(parent_array[i])
    return root_list
    
root_list = FindAllRoot(child,parent)
def FindOneTreeNodes(root,child_array,parent_array):
    TreeNodes = []
#     TreeLabels = [])
    LevelLabels = []

    TreeNodes.append(root)
    LevelLabels.append(0)
    label = 0
    tail = 1
    while label != tail:
        print(label,tail)
        for j in range(len(parent_array)):
            if parent_array[j] == TreeNodes[label] and child_array[j] not in TreeNodes:
                TreeNodes.append(child_array[j])
                LevelLabels.append(LevelLabels[label]+1)
                    
#                     TreeLabels_temp.append([child_array[j],TreeNodes_temp[label],all_label[j]])
                tail += 1
        label += 1
        if LevelLabels[label]+1 >= 5:
            break
    return TreeNodes
level4 = FindOneTreeNodes(root_list[0],child,parent)

level5 = FindOneTreeNodes(root_list[0],child,parent)

for i in level4:
    if i in level5:
        level5.remove(i)
def FindAllTreeNodes(root_list,child_array,parent_array):
    TreeNodes = []
#     TreeLabels = []
    for i in range(len(root_list)):
        print(i)
        TreeNodes_temp = []
#         TreeLabels_temp = []
        TreeNodes_temp.append(root_list[i])
        label = 0
        tail = 1
        while label != tail:        
            for j in range(len(parent_array)):
                if parent_array[j] == TreeNodes_temp[label] and child_array[j] not in TreeNodes_temp:
                    TreeNodes_temp.append(child_array[j])
#                     TreeLabels_temp.append([child_array[j],TreeNodes_temp[label],all_label[j]])
                    tail += 1
            label += 1
#         TreeNodes_temp = np.unique(np.array(TreeNodes_temp))
        TreeNodes.append(TreeNodes_temp)
#         TreeLabels.append(TreeLabels_temp)
    return TreeNodes


all_TreeNodes = FindAllTreeNodes(level5,child,parent)
allTreeNodes100 = []

for i in range(len(all_TreeNodes)):
    if len(all_TreeNodes[i]) > 100 and len(all_TreeNodes[i]) < 120 :
        print(i,len(all_TreeNodes[i]))
        allTreeNodes100.append(all_TreeNodes[i])
hasrelation_ndr = []
for i in range(len(allTreeNodes100)):
    print(i)
    for x in range(len(allTreeNodes100[i])):
        for y in range(len(allTreeNodes100[i])):
            if x == y :
                continue
            a = allTreeNodes100[i][x]
            b = allTreeNodes100[i][y]
            print(a,b)
            if [a,b] not in celluar_copy:
                hasrelation_ndr.append([a,b])
celluar_all = []
for i in range(len(celluar_copy)):
    for j in range(len(celluar_copy[i])):
        if celluar_copy[i][j] not in celluar_all:
            celluar_all.append(celluar_copy[i][j])

norelation_ndr = []
count = 0
while count <= 300000:
    print(count)
    a = np.random.randint(0,len(celluar_all))
    b = np.random.randint(0,len(celluar_all))
    if a != b and [celluar_all[a],celluar_all[b]] not in celluar_copy and [celluar_all[a],celluar_all[b]] not in norelation_ndr:
        norelation_ndr.append([celluar_all[a],celluar_all[b]])
        count += 1
