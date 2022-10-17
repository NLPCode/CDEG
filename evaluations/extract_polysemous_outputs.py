# -*- coding: utf-8 -*-
# @Time    : 2021/11/3
# @Author  : He Xingwei
import sys,os
sys.path.append('../')
from evaluations.automatic_evaluation import load_gold_data
import glob

"""
this script is used to extract the outputs for polysemous
"""

def extract_generated_data(filename, selected_index_list, output_filename, every_line = 3):
    """
    this function aims to load the generated data and extract outputs for polysemous words
    :param filename:
    :return:
    """
    selected_index_set = set(selected_index_list)
    with open(filename, 'r') as fr, open(output_filename, 'w') as fw:
        for i, line in enumerate(fr):
            index = int(i/every_line)
            if index in selected_index_set:
                fw.write(line)
        fw.flush()

test_data = load_gold_data(f"../data/oxford/test_inference.txt")
test_word_set = set([words[0] for words in test_data['word_list']])

training_data = load_gold_data(f"../data/oxford/training_inference.txt")
training_word_set = set([word for words in training_data['word_list'] for word in words])

validation_data = load_gold_data(f"../data/oxford/validation_inference.txt")

# case 1: separate the test set based on the number of definitions
lemma_definition_dict = {}
gold_data = test_data
for lemma, words, pos, definition in zip(gold_data['lemma_list'], gold_data['word_list'],
                                         gold_data['pos_list'], gold_data['definition_list']):
    if lemma not in lemma_definition_dict:
        lemma_definition_dict[lemma] = [(words, pos, definition), ]
    else:
        lemma_definition_dict[lemma].append((words, pos, definition))

gold_data = training_data
for lemma, words, pos, definition in zip(gold_data['lemma_list'], gold_data['word_list'],
                                         gold_data['pos_list'], gold_data['definition_list']):
    if lemma not in lemma_definition_dict:
        lemma_definition_dict[lemma] = [(words, pos, definition), ]
    else:
        lemma_definition_dict[lemma].append((words, pos, definition))

gold_data = validation_data
for lemma, words, pos, definition in zip(gold_data['lemma_list'], gold_data['word_list'],
                                         gold_data['pos_list'], gold_data['definition_list']):
    if lemma not in lemma_definition_dict:
        lemma_definition_dict[lemma] = [(words, pos, definition), ]
    else:
        lemma_definition_dict[lemma].append((words, pos, definition))
lemma_num_definition_list = [(lemma, len(definitions)) for lemma, definitions in lemma_definition_dict.items()]
lemma_num_definition_dict = dict(lemma_num_definition_list)

num = 0
selected_index_list = []
for i, lemma in enumerate(test_data["lemma_list"]):
    _num_definition = lemma_num_definition_dict[lemma]
    if _num_definition >=2:
        num +=1
        selected_index_list.append(i)
print(num)
print(len(test_data["lemma_list"]))
# num = 0
# for i, lemma in enumerate(validation_data["lemma_list"]):
#     _num_definition = lemma_num_definition_dict[lemma]
#     if _num_definition >=2:
#         num +=1
# print(num)
# print(len(test_data["lemma_list"]))

path_dir = f'../outputs/*/*.txt'
filenames = glob.glob(path_dir)

for i, filename in enumerate(filenames):
    output_filename = filename.replace('../outputs', '../polysemous_outputs')
    _path_dir = os.path.dirname(output_filename)
    if not os.path.exists(_path_dir):
        os.makedirs(_path_dir)
    extract_generated_data(filename, selected_index_list, output_filename, every_line=3)


