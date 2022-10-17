# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 10:09 PM
# @Author  : He Xingwei

import sys
sys.path.append('../')
from utils.readability import WordRank, Readability

word2rank = {}
with open('../data/oxford/word2rank_dict.txt','r') as fr:
    for line in fr:
        line = line.strip()
        word, rank ,_ = line.split('\t\t')
        word2rank[word] = int(rank)

a =set()
train_set = set()
test_set = set()
validation_set = set()
d = {}
max_len = 0
for filename in ['../data/oxford/training.txt','../data/oxford/validation.txt','../data/oxford/test.txt']:

    with open(filename, 'r') as fr:
        i = 0
        for line in fr:
            line = line.strip()
            word_seg, lemma_seg, pos_seg, definition_seg, example_seg = line.split('\t\t')
            word = word_seg.split("word::: ")[1]
            lemma = lemma_seg.split('lemma::: ')[1]
            pos = pos_seg.split('pos::: ')[1]
            definition = definition_seg.split('definition::: ')[1]
            example = example_seg.split('example::: ')[1]
            word = lemma
            # if lemma.isalpha() and len(lemma)>1 and len(lemma)<14:
            lemma = lemma.replace('-','')
            # if pos != 'Verb':
            #     continue
            if lemma.encode('UTF-8').isalnum() and len(lemma)>1 and len(word)<=20:
                if word not in d:
                    d[word] = set([definition])
                else:
                    d[word].add(definition)
                if len(word)>max_len:
                    max_len = len(word)
                if 'train' in filename:
                    train_set.add(word)
                elif 'test' in filename:
                    test_set.add(word)
                else:
                    validation_set.add(word)
            else:
                a.add(word)
print(len(test_set))
print(len(test_set&train_set))
freq = {}
for e in test_set&train_set:
    rank = word2rank.get(e,len(word2rank))
    index= int(rank/1000)
    index = min(100, index)
    if index not in freq:
        freq[index] = 1
    else:
        freq[index] += 1
l = sorted(freq.items(), key=lambda x: x[0])
print(l)
print(a, len(a))

for filename in ['../data/oxford/test_inference.txt']:
    test_set = set()
    with open(filename, 'r') as fr:
        ans = 0
        for i, line in enumerate(fr):
            if i%2==1:
                continue
            line = line.strip()
            word_seg, lemma_seg, pos_seg, definition_seg = line.split('\t\t')
            word = word_seg.split("word::: ")[1]
            lemma = lemma_seg.split('lemma::: ')[1]
            pos = pos_seg.split('pos::: ')[1]
            definition = definition_seg.split('definition::: ')[1]
            words = word.split('\t')
            d = {}
            for word in words:
                d[word] = d.get(word,0)+1
            max_num = max(d.values())
            if max_num>=2:
                ans+=1
                test_set.add(lemma)
        print(ans)

    freq = {}
    print(len(test_set-train_set))
    for e in test_set - train_set:
        rank = word2rank.get(e, len(word2rank))
        index = int(rank / 1000)
        index = min(100, index)
        if index not in freq:
            freq[index] = 1
        else:
            freq[index] += 1
    l = sorted(freq.items(), key=lambda x: x[0])
    print(l)

