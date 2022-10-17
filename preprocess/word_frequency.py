# -*- coding: utf-8 -*-
# @Time    : 2021/3/4 5:14 PM
# @Author  : He Xingwei
"""
this script is used to count the number of words in One-Billion-Word corpus.
"""

import glob
from nltk.tokenize import word_tokenize

test_files = glob.glob('../../corpora/one-billion-words/'
                           '1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/*')
training_files = glob.glob('../../corpora/one-billion-words'
                       '/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*')
print(f"Count the number of words from {len(training_files)} training files and {len(test_files)} test files.")

num_lines = 0
word2frequency_dict = {}
word2rank_dict_file = '../data/one-billion-word_word2rank_dict3.txt'
for i, filename in enumerate( training_files + test_files):
    print(f'\rProcess the {i+1}-th/{len(training_files)+len(test_files)} file.         ',end='')
    with open(filename, 'r') as fr:
        for line in fr:
            num_lines +=1
            # words = line.strip().split()
            words = word_tokenize(line.strip())
            for w in words:
                word2frequency_dict[w] = word2frequency_dict.get(w,0)+1
word2frequency_dict = word2frequency_dict.items()
word2frequency = sorted(word2frequency_dict, key=lambda x: x[1], reverse=True)
print(f"Count the number of words from {num_lines} lines.")
print(f"Output the word and rank into {word2rank_dict_file}.")

with open(word2rank_dict_file,'w') as fw:
    rank = 1
    for k, v in word2frequency:
        fw.write(f'{k}\t\t{rank}\t\t{v}\n')
        rank += 1


