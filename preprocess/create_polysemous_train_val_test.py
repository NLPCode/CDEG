# -*- coding: utf-8 -*-
# @Time    : 2021/2/25 8:16 PM
# @Author  : He Xingwei
"""
This file is used to create the training, validation, and test sets, which are created based on definitions.
Therefore, the word definitions are mutually exclusive in the training, validation and test sets.
We should use (lemma, pos, definition) to represent a unique definition.
"""
from collections import Counter
from nltk.tokenize import word_tokenize

definitions_dict = {}

fw_log = open('../data/oxford/statistics_polysemous_train_val_test.txt', 'a')


lemma_definition_dict = {}
for mode in ['training', 'validation', 'test']:
    filename = f'../data/oxford/{mode}_inference.txt'
    with open(filename, 'r') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                word_seg, lemma_seg, pos_seg, definition_seg = line.strip().split('\t\t')
                words = word_seg.split("word::: ")[1].split('\t')
                lemma = lemma_seg.split('lemma::: ')[1]
                pos = pos_seg.split('pos::: ')[1]
                definition = definition_seg.split('definition::: ')[1]

                if lemma not in lemma_definition_dict:
                    lemma_definition_dict[lemma] = [(words, pos, definition), ]
                else:
                    lemma_definition_dict[lemma].append((words, pos, definition))

lemma_num_definition_list = [(lemma, len(definitions)) for lemma, definitions in lemma_definition_dict.items()]
lemma_num_definition_dict = dict(lemma_num_definition_list)

for mode in ['training', 'validation', 'test']:
    filename = f'../data/oxford/{mode}_inference.txt'
    filename2 = f'../data/oxford/polysemous_{mode}_inference.txt'
    flag = False
    with open(filename, 'r') as fr, open(filename2, 'w') as fw:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                word_seg, lemma_seg, pos_seg, definition_seg = line.strip().split('\t\t')
                words = word_seg.split("word::: ")[1].split('\t')
                lemma = lemma_seg.split('lemma::: ')[1]
                pos = pos_seg.split('pos::: ')[1]
                definition = definition_seg.split('definition::: ')[1]

                if lemma_num_definition_dict[lemma]>=2:
                    fw.write(line)
                    flag = True
                else:
                    flag = False
            else:
                if flag:
                    fw.write(line)

    filename = f'../data/oxford/{mode}.txt'
    filename2 = f'../data/oxford/polysemous_{mode}.txt'
    with open(filename, 'r') as fr, open(filename2, 'w') as fw:
        for i, line in enumerate(fr):
            word_seg, lemma_seg, pos_seg, definition_seg, example_seg = line.strip().split('\t\t')
            lemma = lemma_seg.split('lemma::: ')[1]
            if lemma_num_definition_dict[lemma]>=2:
                fw.write(line)


# ======================================
# compute the statistics of training, polysemous validation and polysemous test sets.
# ======================================
extracted_pos = ['Noun','Adjective', 'Verb', 'Adverb', 'Preposition','Interjection','Numeral', 'Pronoun','Determiner','Conjunction']
lemma_definition_dict = {}
train_pos_num = {}
valid_pos_num = {}
test_pos_num = {}

train_lemma_set = set()
valid_lemma_set = set()
test_lemma_set = set()
num_examples_list = []
num_definitions_list = []
num_tokens_example_list = []
num_tokens_definition_list = []
num_same_word_examples_list = []
num_same_word_tokens_example_list = []
for mode, num_dict, s in zip(['training', 'validation', 'test'], [train_pos_num, valid_pos_num, test_pos_num], [train_lemma_set, valid_lemma_set,  test_lemma_set]):
    if mode == 'training':
        filename = f'../data/oxford/{mode}_inference.txt'
    else:
        filename = f'../data/oxford/polysemous_{mode}_inference.txt'
    num_examples = 0
    num_definitions = 0
    num_tokens_example = 0
    num_tokens_definition = 0
    num_same_word_examples = 0
    num_same_word_tokens_example = 0

    with open(filename, 'r') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                word_seg, lemma_seg, pos_seg, definition_seg = line.strip().split('\t\t')
                words = word_seg.split("word::: ")[1].split('\t')
                lemma = lemma_seg.split('lemma::: ')[1]
                pos = pos_seg.split('pos::: ')[1]
                definition = definition_seg.split('definition::: ')[1]
                num_tokens_definition += len(word_tokenize(definition))
                num_definitions += 1
                num_dict[pos] = num_dict.get(pos, 0) + 1
                s.add(lemma)

                if lemma not in lemma_definition_dict:
                    lemma_definition_dict[lemma] = [(words, pos, definition), ]
                else:
                    lemma_definition_dict[lemma].append((words, pos, definition))
            else:
                example_list = line.strip().split('Reference examples::: ')[1].split('\t\t')
                assert len(words) == len(example_list)
                for word, example in zip(words, example_list):
                    num_examples += 1
                    num_tokens_example += len(word_tokenize(example))
                    if word == words[0]:
                        num_same_word_examples += 1
                        num_same_word_tokens_example += len(word_tokenize(example))
    num_examples_list.append(num_examples)
    num_definitions_list.append(num_definitions)
    num_tokens_example_list.append(num_tokens_example)
    num_tokens_definition_list.append(num_tokens_definition)
    num_same_word_examples_list.append(num_same_word_examples)
    num_same_word_tokens_example_list.append(num_same_word_tokens_example)
for l in [num_examples_list, num_definitions_list, num_tokens_example_list, num_tokens_definition_list,
          num_same_word_examples_list, num_same_word_tokens_example_list]:
    l.append(sum(l))

fw_log.write('Statistics for Training,  polysemous Validation, polysemous Test, Total\n')
fw_log.write('Partition, Training, Validation, Test, Total\n')
fw_log.write(f"The number of words, {len(train_lemma_set)}, {len(valid_lemma_set)}, {len(test_lemma_set)}, {len(train_lemma_set| valid_lemma_set|  test_lemma_set)}\n")
fw_log.write(f"The number of definitions, {num_definitions_list[0]}, {num_definitions_list[1]}, {num_definitions_list[2]}, {num_definitions_list[3]}\n")
fw_log.write(f"The number of examples, {num_examples_list[0]}, {num_examples_list[1]}, {num_examples_list[2]}, {num_examples_list[3]}\n")
fw_log.write(f"The number of same word examples, {num_same_word_examples_list[0]}, {num_same_word_examples_list[1]}, {num_same_word_examples_list[2]}, {num_same_word_examples_list[3]}\n")
fw_log.write(f"The number of tokens in examples, {num_tokens_example_list[0]}, {num_tokens_example_list[1]}, {num_tokens_example_list[2]}, {num_tokens_example_list[3]}\n")
fw_log.write(f"The number of tokens in same word examples, {num_same_word_tokens_example_list[0]}, {num_same_word_tokens_example_list[1]}, "
             f"{num_same_word_tokens_example_list[2]}, {num_same_word_tokens_example_list[3]}\n")

fw_log.write(f"The average example len, {1.0*num_tokens_example_list[0]/num_examples_list[0]:.3f}, {1.0*num_tokens_example_list[1]/num_examples_list[1]:.3f}, "
             f"{1.0*num_tokens_example_list[2]/num_examples_list[2]:.3f}, {1.0*num_tokens_example_list[3]/num_examples_list[3]:.3f}\n")
fw_log.write(f"The average same word example len, {1.0*num_same_word_tokens_example_list[0]/num_same_word_examples_list[0]:.3f}, "
             f"{1.0*num_same_word_tokens_example_list[1]/num_same_word_examples_list[1]:.3f}, "
             f"{1.0*num_same_word_tokens_example_list[2]/num_same_word_examples_list[2]:.3f}, "
             f"{1.0*num_same_word_tokens_example_list[3]/num_same_word_examples_list[3]:.3f}\n")

fw_log.write(f"The average definition len, {1.0*num_tokens_definition_list[0]/num_definitions_list[0]:.3f}, "
             f"{1.0*num_tokens_definition_list[1]/num_definitions_list[1]:.3f}, "
             f"{1.0*num_tokens_definition_list[2]/num_definitions_list[2]:.3f}, "
             f"{1.0*num_tokens_definition_list[3]/num_definitions_list[3]:.3f}\n")


training_num_def = sum([len(lemma_definition_dict[lemma]) for lemma in train_lemma_set])
valid_num_def = sum([len(lemma_definition_dict[lemma]) for lemma in valid_lemma_set])
test_num_def = sum([len(lemma_definition_dict[lemma]) for lemma in test_lemma_set])

fw_log.write(f"The average number of definitions for each lemma, {1.0*training_num_def/len(train_lemma_set):.3f}, "
             f"{1.0*valid_num_def/len(valid_lemma_set):.3f}, "
             f"{1.0*test_num_def/len(test_lemma_set):.3f}, "
             f"{1.0*(training_num_def+valid_num_def+test_num_def)/(len(train_lemma_set)+len(valid_lemma_set)+len(test_lemma_set)):.3f}\n")

fw_log.write(f"\nthe dataset contains {len(train_lemma_set| valid_lemma_set|  test_lemma_set)} unique lemmas\n")
fw_log.write(f"the test set contains {len(test_lemma_set - train_lemma_set)} unique lemmas, which are not in the train set.\n")
fw_log.write(f"the validation set contains {len(valid_lemma_set - train_lemma_set)} unique lemmas, which are not in the train set.\n")


fw_log.write("\nThe definition distribution over pos.\n")
fw_log.write('Partition, Training, Validation, Test, Total\n')
for e in extracted_pos:
    total = train_pos_num[e] + valid_pos_num[e] + test_pos_num[e]
    fw_log.write(f'{e}, {train_pos_num[e]}, {valid_pos_num[e]}, {test_pos_num[e]}, {total}\n')

# statistics of lemmas over the number of definitions
lemma_num_definition_list = [len(definitions) for lemma, definitions in lemma_definition_dict.items()]
training_lemma_num_definition_list = [len(lemma_definition_dict[lemma]) for lemma in train_lemma_set]
valid_lemma_num_definition_list = [len(lemma_definition_dict[lemma]) for lemma in valid_lemma_set]
test_lemma_num_definition_list = [len(lemma_definition_dict[lemma]) for lemma in test_lemma_set]

counter = Counter(lemma_num_definition_list)
training_counter = Counter(training_lemma_num_definition_list)
valid_counter = Counter(valid_lemma_num_definition_list)
test_counter = Counter(test_lemma_num_definition_list)
max_num  = max(counter.keys())
print(max_num)
# counter_sorted = sorted(counter.items(),key=lambda x:x[0],reverse=False)
fw_log.write(f'\nthe statistics of lemmas over the number of definition.\n')
fw_log.write('Partition, Training, Validation, Test, Total\n')
# print(counter_sorted)
for k in range(1, max_num+1):
    fw_log.write(f'{k}, {training_counter[k]}, {valid_counter[k]}, {test_counter[k]}, {counter[k]}\n')

# statistics of lemmas over pos
lemma_num_pos_list = [len(set([definition[1] for definition in definitions])) for lemma, definitions in lemma_definition_dict.items()]
training_lemma_num_pos_list = [len(set([definition[1] for definition in lemma_definition_dict[lemma]])) for lemma in train_lemma_set]
valid_lemma_num_pos_list = [len(set([definition[1] for definition in lemma_definition_dict[lemma]])) for lemma in valid_lemma_set]
test_lemma_num_pos_list = [len(set([definition[1] for definition in lemma_definition_dict[lemma]])) for lemma in test_lemma_set]

counter = Counter(lemma_num_pos_list)
training_counter = Counter(training_lemma_num_pos_list)
valid_counter = Counter(valid_lemma_num_pos_list)
test_counter = Counter(test_lemma_num_pos_list)
max_num  = max(counter.keys())
print(max_num)

# counter_sorted = sorted(counter.items(), key=lambda x: x[0], reverse=False)

fw_log.write(f'\nthe statistics of lemmas over the number of pos.\n')
fw_log.write('Partition, Training, Validation, Test, Total\n')
for k in range(1, max_num+1):
    fw_log.write(f'{k}, {training_counter[k]}, {valid_counter[k]}, {test_counter[k]}, {counter[k]}\n')

# ======================================
# compute the statistics of polysemous training, polysemous validation and polysemous test sets.
# ======================================
extracted_pos = ['Noun','Adjective', 'Verb', 'Adverb', 'Preposition','Interjection','Numeral', 'Pronoun','Determiner','Conjunction']
lemma_definition_dict = {}
train_pos_num = {}
valid_pos_num = {}
test_pos_num = {}

train_lemma_set = set()
valid_lemma_set = set()
test_lemma_set = set()
num_examples_list = []
num_definitions_list = []
num_tokens_example_list = []
num_tokens_definition_list = []
num_same_word_examples_list = []
num_same_word_tokens_example_list = []
for mode, num_dict, s in zip(['training', 'validation', 'test'], [train_pos_num, valid_pos_num, test_pos_num], [train_lemma_set, valid_lemma_set,  test_lemma_set]):
    filename = f'../data/oxford/polysemous_{mode}_inference.txt'
    num_examples = 0
    num_definitions = 0
    num_tokens_example = 0
    num_tokens_definition = 0
    num_same_word_examples = 0
    num_same_word_tokens_example = 0

    with open(filename, 'r') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                word_seg, lemma_seg, pos_seg, definition_seg = line.strip().split('\t\t')
                words = word_seg.split("word::: ")[1].split('\t')
                lemma = lemma_seg.split('lemma::: ')[1]
                pos = pos_seg.split('pos::: ')[1]
                definition = definition_seg.split('definition::: ')[1]
                num_tokens_definition += len(word_tokenize(definition))
                num_definitions += 1
                num_dict[pos] = num_dict.get(pos, 0) + 1
                s.add(lemma)

                if lemma not in lemma_definition_dict:
                    lemma_definition_dict[lemma] = [(words, pos, definition), ]
                else:
                    lemma_definition_dict[lemma].append((words, pos, definition))
            else:
                example_list = line.strip().split('Reference examples::: ')[1].split('\t\t')
                assert len(words) == len(example_list)
                for word, example in zip(words, example_list):
                    num_examples += 1
                    num_tokens_example += len(word_tokenize(example))
                    if word == words[0]:
                        num_same_word_examples += 1
                        num_same_word_tokens_example += len(word_tokenize(example))
    num_examples_list.append(num_examples)
    num_definitions_list.append(num_definitions)
    num_tokens_example_list.append(num_tokens_example)
    num_tokens_definition_list.append(num_tokens_definition)
    num_same_word_examples_list.append(num_same_word_examples)
    num_same_word_tokens_example_list.append(num_same_word_tokens_example)
for l in [num_examples_list, num_definitions_list, num_tokens_example_list, num_tokens_definition_list,
          num_same_word_examples_list, num_same_word_tokens_example_list]:
    l.append(sum(l))

fw_log.write('Statistics for Training,  polysemous Validation, polysemous Test, Total\n')
fw_log.write('Partition, Training, Validation, Test, Total\n')
fw_log.write(f"The number of words, {len(train_lemma_set)}, {len(valid_lemma_set)}, {len(test_lemma_set)}, {len(train_lemma_set| valid_lemma_set|  test_lemma_set)}\n")
fw_log.write(f"The number of definitions, {num_definitions_list[0]}, {num_definitions_list[1]}, {num_definitions_list[2]}, {num_definitions_list[3]}\n")
fw_log.write(f"The number of examples, {num_examples_list[0]}, {num_examples_list[1]}, {num_examples_list[2]}, {num_examples_list[3]}\n")
fw_log.write(f"The number of same word examples, {num_same_word_examples_list[0]}, {num_same_word_examples_list[1]}, {num_same_word_examples_list[2]}, {num_same_word_examples_list[3]}\n")
fw_log.write(f"The number of tokens in examples, {num_tokens_example_list[0]}, {num_tokens_example_list[1]}, {num_tokens_example_list[2]}, {num_tokens_example_list[3]}\n")
fw_log.write(f"The number of tokens in same word examples, {num_same_word_tokens_example_list[0]}, {num_same_word_tokens_example_list[1]}, "
             f"{num_same_word_tokens_example_list[2]}, {num_same_word_tokens_example_list[3]}\n")

fw_log.write(f"The average example len, {1.0*num_tokens_example_list[0]/num_examples_list[0]:.3f}, {1.0*num_tokens_example_list[1]/num_examples_list[1]:.3f}, "
             f"{1.0*num_tokens_example_list[2]/num_examples_list[2]:.3f}, {1.0*num_tokens_example_list[3]/num_examples_list[3]:.3f}\n")
fw_log.write(f"The average same word example len, {1.0*num_same_word_tokens_example_list[0]/num_same_word_examples_list[0]:.3f}, "
             f"{1.0*num_same_word_tokens_example_list[1]/num_same_word_examples_list[1]:.3f}, "
             f"{1.0*num_same_word_tokens_example_list[2]/num_same_word_examples_list[2]:.3f}, "
             f"{1.0*num_same_word_tokens_example_list[3]/num_same_word_examples_list[3]:.3f}\n")

fw_log.write(f"The average definition len, {1.0*num_tokens_definition_list[0]/num_definitions_list[0]:.3f}, "
             f"{1.0*num_tokens_definition_list[1]/num_definitions_list[1]:.3f}, "
             f"{1.0*num_tokens_definition_list[2]/num_definitions_list[2]:.3f}, "
             f"{1.0*num_tokens_definition_list[3]/num_definitions_list[3]:.3f}\n")


training_num_def = sum([len(lemma_definition_dict[lemma]) for lemma in train_lemma_set])
valid_num_def = sum([len(lemma_definition_dict[lemma]) for lemma in valid_lemma_set])
test_num_def = sum([len(lemma_definition_dict[lemma]) for lemma in test_lemma_set])

fw_log.write(f"The average number of definitions for each lemma, {1.0*training_num_def/len(train_lemma_set):.3f}, "
             f"{1.0*valid_num_def/len(valid_lemma_set):.3f}, "
             f"{1.0*test_num_def/len(test_lemma_set):.3f}, "
             f"{1.0*(training_num_def+valid_num_def+test_num_def)/(len(train_lemma_set)+len(valid_lemma_set)+len(test_lemma_set)):.3f}\n")

fw_log.write(f"\nthe dataset contains {len(train_lemma_set| valid_lemma_set|  test_lemma_set)} unique lemmas\n")
fw_log.write(f"the test set contains {len(test_lemma_set - train_lemma_set)} unique lemmas, which are not in the train set.\n")
fw_log.write(f"the validation set contains {len(valid_lemma_set - train_lemma_set)} unique lemmas, which are not in the train set.\n")


fw_log.write("\nThe definition distribution over pos.\n")
fw_log.write('Partition, Training, Validation, Test, Total\n')
for e in extracted_pos:
    total = train_pos_num[e] + valid_pos_num[e] + test_pos_num[e]
    fw_log.write(f'{e}, {train_pos_num[e]}, {valid_pos_num[e]}, {test_pos_num[e]}, {total}\n')

# statistics of lemmas over the number of definitions
lemma_num_definition_list = [len(definitions) for lemma, definitions in lemma_definition_dict.items()]
training_lemma_num_definition_list = [len(lemma_definition_dict[lemma]) for lemma in train_lemma_set]
valid_lemma_num_definition_list = [len(lemma_definition_dict[lemma]) for lemma in valid_lemma_set]
test_lemma_num_definition_list = [len(lemma_definition_dict[lemma]) for lemma in test_lemma_set]

counter = Counter(lemma_num_definition_list)
training_counter = Counter(training_lemma_num_definition_list)
valid_counter = Counter(valid_lemma_num_definition_list)
test_counter = Counter(test_lemma_num_definition_list)
max_num  = max(counter.keys())
# counter_sorted = sorted(counter.items(),key=lambda x:x[0],reverse=False)
fw_log.write(f'\nthe statistics of lemmas over the number of definition.\n')
fw_log.write('Partition, Training, Validation, Test, Total\n')
# print(counter_sorted)
for k in range(1, max_num+1):
    fw_log.write(f'{k}, {training_counter[k]}, {valid_counter[k]}, {test_counter[k]}, {counter[k]}\n')

# statistics of lemmas over pos
lemma_num_pos_list = [len(set([definition[1] for definition in definitions])) for lemma, definitions in lemma_definition_dict.items()]
training_lemma_num_pos_list = [len(set([definition[1] for definition in lemma_definition_dict[lemma]])) for lemma in train_lemma_set]
valid_lemma_num_pos_list = [len(set([definition[1] for definition in lemma_definition_dict[lemma]])) for lemma in valid_lemma_set]
test_lemma_num_pos_list = [len(set([definition[1] for definition in lemma_definition_dict[lemma]])) for lemma in test_lemma_set]

counter = Counter(lemma_num_pos_list)
training_counter = Counter(training_lemma_num_pos_list)
valid_counter = Counter(valid_lemma_num_pos_list)
test_counter = Counter(test_lemma_num_pos_list)
max_num  = max(counter.keys())
# counter_sorted = sorted(counter.items(), key=lambda x: x[0], reverse=False)

fw_log.write(f'\nthe statistics of lemmas over the number of pos.\n')
fw_log.write('Partition, Training, Validation, Test, Total\n')
for k in range(1, max_num+1):
    fw_log.write(f'{k}, {training_counter[k]}, {valid_counter[k]}, {test_counter[k]}, {counter[k]}\n')

fw_log.close()

