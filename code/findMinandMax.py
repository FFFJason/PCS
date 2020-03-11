def CreatePcsPairs(test_all):
    global nlp
    noun_parse = []#初次提取出来的的名词短语
    test_all_copy = test_all[:]
    for s in test_all:
        word_tree = nlp.parse(s)
        if re.search('IN',word_tree):#'IN'表示的是介词，如果没有'IN'说明整句只有一个短语，如果有'IN'说明不止一个短语，需要分开
            b = word_tree.split('\r\n')#按照规律一般是一个换行符一个名词短语，b存储分隔后的tree

            pattern_1_1 = 'NP'
            pattern_1_2 = 'VP'#'NP'是名词短语，设为匹配模式1
            result1 = [] #result存储用模式匹配1匹配出来的结果
            for i in b:#这个循环的作用是，遍历分隔后的树，将有'NP'字符的串存储
                if re.search(pattern_1_1,i):# or re.search(pattern_1_2,i):
                    result1.append(i)

            result2 = []
            pattern_2_1 = 'JJ'
            pattern_2_2 = 'RB'
            pattern_2_3 = 'VBN'#'JJ'是形容词，介于此实验的特殊性，无形容词的短语是不符合条件的，因此舍去，将'JJ'匹配的字符串存储到result2中
            for i in result1:
                if re.search(pattern_2_1,i) or re.search(pattern_2_2,i) or re.search(pattern_2_3,i):
                    result2.append(i)
            for i in range(len(result2)):
                result2[i] = result2[i].replace('(','').replace(')','')
                result2[i] = re.sub(r'[A-Z]{1,10} ','',result2[i])
                result2[i] = re.split(r" +",result2[i])
                result2[i].pop(0)
            noun_parse.extend(result2)
        else:
            result = word_tree.replace('\r\n','').replace('(','').replace(')','')
            result = re.sub(r'[A-Z]{1,10} ','',result)
            result = re.split(r" +",result)
            result.pop(0)
            noun_parse.append(result)
    noun_set = []
    for i in noun_parse:
        if i not in noun_set:
            noun_set.append(i)
    np_pairs = []

    for i in range(len(noun_set)):
        j = i+1
        while j <= len(noun_set)-1:        
            if len(noun_set[i]) == len(noun_set[j]):
                label = True
                for x in range(len(noun_set[i])-1):
                    if noun_set[i][x+1] != noun_set[j][x+1]:
                        label = False
                if label and (re.match(r'[A-Z]{1}',noun_set[i][0]) and re.match(r'[A-Z]{1}',noun_set[j][0])
                              or re.match(r'[a-z]{1}',noun_set[i][0]) and re.match(r'[a-z]{1}',noun_set[j][0])):
                    np_pairs.append([noun_set[i],noun_set[j]])

            j += 1
    np_pairs1 = []
    for i in range(len(np_pairs)):
        for x in range(len(test_all)-1):
            y = x+1
            while y <= len(test_all)-1:
                a = test_all[x].split(' ')
                b = test_all[y].split(' ')
                len_a = len(a)
                len_b = len(b)            
                if len_a != len_b:
                    y += 1
                    continue
                else:
                    np1 = functools.reduce(lambda x,y:x+' '+y,np_pairs[i][0])
                    np2 = functools.reduce(lambda x,y:x+' '+y,np_pairs[i][1])
                    if re.search(np1,test_all[x]) and re.search(np2,test_all[y]) or re.search(np1,test_all[y]) and re.search(np2,test_all[x]):
                        if np_pairs[i][0] == a and np_pairs[i][1] == b or np_pairs[i][1] == a and np_pairs[i][0] == b:
                            y += 1
                            continue
                        else:
                            for z in range(len_a):
                                if a[z] != b[z] and (a[z] == np_pairs[i][0][0] and b[z] == np_pairs[i][1][0] or a[z] == np_pairs[i][1][0] and b[z] == np_pairs[i][0][0] ) :
                                    a.remove(a[z])
                                    b.remove(b[z])
                                    if a == b:
                                        if np_pairs[i] not in np_pairs1:
                                            np_pairs1.append(np_pairs[i])
                                    break
                    y += 1               
    adj_pairs=[]
    for i in range(len(np_pairs1)):
        adj_pairs.append(np_pairs1[i][0][0]+','+np_pairs1[i][1][0])   
    np_pairs = None
    np_pairs1 = None
    noun_set = None
    noun_parse = None
    
    
    pcs_pairs  = []
    pcs_concepts = []
    test_all_copy = list(test_all[:])
    for i in range(len(test_all)-1):
        j = i+1
        while j<= len(test_all)-1:
            a_split = test_all[i].split(' ')
            b_split = test_all[j].split(' ')
            a_split_copy = a_split[:]
            b_split_copy = b_split[:]
            if len(a_split) != len(b_split):
                j += 1
                continue
            else:
                length = len(a_split)
                for x in range(len(a_split)):
                    if a_split[x] != b_split[x]:
                        a = a_split[x]
                        b = b_split[x]
                        a_split_copy.remove(a)
                        b_split_copy.remove(b)
                        if a_split_copy == b_split_copy and ((a+','+b).lower() in adj_pairs or (b+','+a).lower() in adj_pairs):
                            pcs_pairs.append([test_all[i],test_all[j]])
                            if test_all[i] in test_all_copy:
                                test_all_copy.remove(test_all[i])
                            if test_all[j] in test_all_copy:
                                test_all_copy.remove(test_all[j])
                            break
                        elif a_split_copy != b_split_copy and  ((a+','+b).lower() in adj_pairs or (b+','+a).lower() in adj_pairs):
                            continue
                        else:
                            break

            j += 1     
    pcs_pairs_copy = pcs_pairs[:]
    pcs_pairs_add = []
    pcs_pairs_all = []
    for i in range(len(pcs_pairs)):
        if pcs_pairs[i] not in pcs_pairs_copy:
            continue
        temp = [] 
        temp.extend(pcs_pairs[i])
        if pcs_pairs[i] in pcs_pairs_copy:
            pcs_pairs_copy.remove(pcs_pairs[i])
        for j in range(len(pcs_pairs)):
            if i == j:
                continue
            if pcs_pairs[j] not in pcs_pairs_copy:
                continue
            if pcs_pairs[j][0] in temp or pcs_pairs[j][1] in temp:
                temp.extend(pcs_pairs[j])
                if pcs_pairs[j] in pcs_pairs_copy:
                    pcs_pairs_copy.remove(pcs_pairs[j])
            for l in range(len(pcs_pairs)):
                if i == l or j == l:
                    continue
                if pcs_pairs[l] not in pcs_pairs_copy:
                    continue
                if pcs_pairs[l][0] in temp or pcs_pairs[l][1] in temp:
                    temp.extend(pcs_pairs[l])
                    if pcs_pairs[l] in pcs_pairs_copy:
                        pcs_pairs_copy.remove(pcs_pairs[l])
        temp = np.array(temp)
        temp = np.unique(temp)
        pcs_pairs_all.append(temp)
    length_pcs = len(pcs_pairs_all)
    pcs_pairs_all.extend(test_all_copy)
    length_all = len(pcs_pairs_all)
    percent = length_pcs/length_all
    return pcs_pairs_all,percent,adj_pairs
   
   
   def findMinandMax(text_set):
    pcs_set = []
    percent_set = []
    count = 0
    for i in text_set:
        print(count)
        temp1,temp2 = CreatePcsPairs(i)
        pcs_set.append(temp1)
        percent_set.append(temp2)
        count += 1
    min_l = 1
    max_l = 0
    max_label = 0
    min_label = 0
    for i in range(len(percent_set)):
        if percent_set[i] > max_l:
            max_l = percent_set[i]
            max_label = i
        if percent_set[i] < min_l:
            min_l = percent_set[i]
            min_label = i
    return pcs_set,percent_set,pcs_set[min_label],pcs_set[max_label],percent_set[min_label],percent_set[max_label]
