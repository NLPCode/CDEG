# -*- coding: utf-8 -*-
# @Time    : 2021/2/25 8:16 PM
# @Author  : He Xingwei
"""
This file is used to create the training, validation, and test sets, which are created based on definitions.
Therefore, the word definitions are mutually exclusive in the training, validation and test sets.
We should use (lemma, pos, definition) to represent a unique definition.
"""
import random
from nltk.tokenize import word_tokenize
from transformers import BartTokenizer
from collections import Counter

tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')

random.seed(0)
max_len = 100
max_example_len = 60
max_definition_len = 60
definitions_dict = {}
definitions_dict2 = {}
i = 0
pos_dict = {}

fw_log = open('../data/oxford/statistics_train_val_test.txt','a')

with open('../data/oxford/word_definition_example/1words.txt','r') as fr, \
        open('../data/oxford/word_definition_example/1words_filtered.txt','w') as fw:
    num_removed = 0.0
    total_intances = 0
    for index, line in enumerate(fr):

        word, lemma, pos, definition, example = line.split('\t\t')
        word = word.split('::: ')[1].strip()
        lemma = lemma.split('::: ')[1].strip()
        pos = pos.split('::: ')[1].strip()
        definition = definition.split('::: ')[1].strip()
        example = example.split('::: ')[1].strip()
        ids1 = tokenizer.encode(example, add_special_tokens=False)
        ids2 = tokenizer.encode(definition, add_special_tokens=False)
        ids3 = tokenizer.encode(word, add_special_tokens=False)
        total_intances += 1
        if definitions_dict2.get((lemma, pos, definition),-1) ==-1:
            definitions_dict2[(lemma, pos, definition)] = [(word, lemma, pos, example),]

        if len(ids1)+len(ids2) + len(ids3)>max_len or len(ids2)>max_definition_len or len(ids1) >max_example_len:
            fw.write(line)
            num_removed+=1
            continue
        if lemma.encode('UTF-8').isalnum() and len(lemma) > 1 and len(word) <= 20:
            if definitions_dict.get((lemma, pos, definition),-1) ==-1:
                definitions_dict[(lemma, pos, definition)] = [(word, lemma, pos, example),]
            else:
                definitions_dict[(lemma, pos, definition)].append((word, lemma, pos, example))
        else:
            # remove the (lemma, pos, definition ) tuple, when word contains non-english or non-digital letter,
            # or when the lemma is too long or too short
            fw.write(line)
            num_removed += 1
        # if index>10000:
        #     break
        if index%100==0:
            print(f'\r{index}',end='')

    # remove the (lemma, pos, definition ) tuple,
    # when the number of examples for any inflected word of the lemma are less than 2
    new_definitions_dict ={}
    print()
    print(len(definitions_dict))
    for key, value in definitions_dict.items():
        words = [e[0] for e in value]
        d = {}
        for word in words:
            d[word] = d.get(word, 0) + 1
        max_num = max(d.values())
        if max_num < 2:
            fw.write(line)
            num_removed +=1
        else:
            t = sorted(d.items(), key=lambda x: -x[1])
            new_l = []
            for k, v in t:
                for e in value:
                    if e[0] == k:
                        new_l.append(e)
            assert len(new_l) == len(value)
            new_definitions_dict[key] = new_l
    definitions_dict = new_definitions_dict

fw_log.write(f'We remove {num_removed} instances from {total_intances}, '
      f'the percentage is {num_removed/total_intances:.3f}.\n')
fw_log.write(f'We remove {len(definitions_dict2)-len(definitions_dict)} definitions from {len(definitions_dict2)}.\n\n')

extracted_pos = \
    ['Noun','Adjective', 'Verb', 'Adverb', 'Preposition','Interjection','Numeral', 'Pronoun','Determiner','Conjunction']
train_num = {}
valid_num = {}
test_num = {}

for e in definitions_dict:
    pos_dict[e[1]] = pos_dict.get(e[1],0)+1
print(sorted(pos_dict.items(), key=lambda x:x[0], reverse=True))
total_definitions =0
for _pos in extracted_pos:
    total_definitions += pos_dict[_pos]
fw_log.write(f"The number of definitions are {len(definitions_dict)}.\n")
fw_log.write(f"The number of selected definitions are {total_definitions}.")
# valid_size = 10000
valid_size = int(total_definitions/10)
test_size = valid_size

train_size = total_definitions - valid_size - test_size
fw_log.write(f"The number of definitions for the training set is {train_size}, "
         f"validation set is {valid_size}, test set is {test_size}.\n")
for e in extracted_pos:
    test_num[e] = int(pos_dict[e]*test_size/total_definitions)
    valid_num[e] = int(pos_dict[e]*valid_size/total_definitions)
remain = test_size - sum(test_num.values())
for e in extracted_pos:
    test_num[e]+=1
    valid_num[e]+=1
    remain-=1
    if remain==0:
        break
for e in extracted_pos:
    train_num[e] = pos_dict[e] - test_num[e] - valid_num[e]

fw_log.write("The definition distribution over pos.\n")
for e in extracted_pos:
    print(e)
    fw_log.write(f'{e},{train_num[e]},{valid_num[e]}, {test_num[e]}\n')


print(train_num)
print(test_num)
print(valid_num)
assert sum(train_num.values()) == train_size and sum(test_num.values()) == test_size and sum(valid_num.values()) == valid_size

train_definition_index = {}
valid_definition_index = {}
test_definition_index = {}

# pre-select some definitions as the test set
pre_seleted_test_set = \
    [('bank',"Noun",'the land alongside or sloping down to a river or lake'),
    ('bank',"Noun",'a financial establishment that uses money deposited by customers '
    'for investment, pays it out when required, makes loans at interest, and exchanges currency'),
    ('bank',"Verb",'deposit (money or valuables) in a bank'),
    ('bank',"Verb",'heap (a substance) into a mass or mound'),
    ('dream',"Noun","a series of thoughts, images, and sensations occurring in a person's mind during sleep"),
    ('dream',"Noun","a cherished aspiration, ambition, or ideal"),
    ('dream',"Verb","indulge in daydreams or fantasies about something greatly desired"),
    ('dream',"Verb","contemplate the possibility of doing something or that something might be the case"),
    ('large',"Adjective",'pursuing a commercial activity on a significant scale'), # generate with larger, largest, large
    ('happy',"Adjective",'feeling or showing pleasure or contentment'),
    ('happy',"Adjective",'satisfied with the quality or standard of'),
    ('happy',"Adjective",'willing to do something'),
    ('happily',"Adverb",'in a happy way'),
    ('happiness',"Noun",'the state of being happy'),
    ('satisfying',"Adjective",'giving fulfilment or the pleasure associated with this'),
    ('satisfied',"Adjective",'contented; pleased'),
    ('satisfyingly',"Adverb",'in a way that gives fulfilment or the pleasure associated with this'),
    ('satisfy',"Verb",'meet the expectations, needs, or desires of (someone)'),
    ('satisfy',"Verb",'fulfil (a desire or need)'),
    ('satisfactory',"Adjective",'fulfilling expectations or needs; acceptable, though not outstanding or perfect'),
    ('satisfactorily',"Adverb",'in a way that fulfils expectations or needs; acceptably'),
    ('sentence','Noun','a set of words that is complete in itself, typically containing '
                    'a subject and predicate, conveying a statement, question, exclamation, '
                    'or command, and consisting of a main clause and sometimes one or more subordinate clauses.'),
    ('sentence','Verb','declare the punishment decided for (an offender)'),
    # ("2.0", "Adjective","used to denote a superior or more advanced version of an original concept, product, service, etc."),
    ( "star", "Noun", "an outstandingly successful person or thing in a group"),
    ( "star", "Noun", "a fixed luminous point in the night sky which is a large, remote incandescent body like the sun."),
    ]
unseen_lemmas = ['sentence', 'drawback']

for e in pre_seleted_test_set:
    _pos = e[1]
    if test_definition_index.get(_pos, -1) == -1:
        test_definition_index[_pos] = [e, ]
    else:
        test_definition_index[_pos].append(e)
    test_num[_pos] -= 1

definition_index_list = []
for e in definitions_dict:
    if e[1] in extracted_pos and e not in pre_seleted_test_set:
        definition_index_list.append(e)

random.shuffle(definition_index_list)

for e in definition_index_list:
    _pos = e[1]
    if valid_num[_pos]:
        if valid_definition_index.get(_pos,-1) == -1:
            valid_definition_index[_pos] = [e,]
        else:
            valid_definition_index[_pos].append(e)
        valid_num[_pos] -= 1
    elif test_num[_pos]:
        if test_definition_index.get(_pos,-1) == -1:
            test_definition_index[_pos] = [e,]
        else:
            test_definition_index[_pos].append(e)
        test_num[_pos] -= 1
    elif train_num[_pos]:
        if train_definition_index.get(_pos,-1) == -1:
            train_definition_index[_pos] = [e,]
        else:
            train_definition_index[_pos].append(e)
        train_num[_pos] -=1
    else:
        pass
assert sum(train_num.values()) == 0 and sum(test_num.values()) == 0 and sum(valid_num.values()) == 0


train_lemma_set = set()
valid_lemma_set = set()
test_lemma_set = set()
for d, s in zip([train_definition_index, valid_definition_index, test_definition_index],
                [train_lemma_set, valid_lemma_set,  test_lemma_set]):
    num_examples = 0.0
    num_definitions = 0.0
    num_tokens_example = 0.0
    num_tokens_definition = 0.0
    for _pos in extracted_pos:
        pos_definition_index_list = d[_pos]
        for pos_definition_index in pos_definition_index_list:
            lemma, pos, definition = pos_definition_index
            num_tokens_definition +=  len(word_tokenize(definition))
            num_definitions +=1
            s.add(lemma)
            for word, lemma, pos, example in definitions_dict[pos_definition_index]:
                num_examples += 1
                num_tokens_example += len(word_tokenize(example))
    fw_log.write("The number of words, definitions, examples, tokens in examples, tokens in definitions, average examples len, definition len, "
             "for training/valid/test sets:\n")
    fw_log.write(f'{len(s)}, {num_definitions}, {num_examples}, {num_tokens_example}, {num_tokens_definition}, '
             f'{num_tokens_example/num_examples:.3f}, {num_tokens_definition/num_definitions:.3f}\n')
fw_log.write(f"the dataset contains {len(train_lemma_set| valid_lemma_set|  test_lemma_set)} unique lemmas\n")
fw_log.write(f"the test set contains {len(test_lemma_set - train_lemma_set)} unique lemmas, which are not in the train set.\n")
fw_log.write(f"the validation set contains {len(valid_lemma_set - train_lemma_set)} unique lemmas, which are not in the train set.\n")

for data_definition_index, filename in zip([train_definition_index, valid_definition_index, test_definition_index],
                       ['../data/oxford/training.txt','../data/oxford/validation.txt','../data/oxford/test.txt']):
    # if 'valid' in filename or 'test' in filename:
    #     fw2 = open(filename.replace('.txt','_inference.txt'), 'w')
    # else:
    #     fw2 = None
    fw2 = open(filename.replace('.txt', '_inference.txt'), 'w')

    with open(filename, 'w') as fw:
        for _pos in extracted_pos:
            #sort based on lemma
            pos_definition_index_list = data_definition_index[_pos]
            pos_definition_index_list = sorted(pos_definition_index_list, key=lambda e: (e[0], e[2]))
            for pos_definition_index in pos_definition_index_list:
                lemma, pos, definition = pos_definition_index

                examples_list = []
                words_list = []
                for word, lemma, pos, example in definitions_dict[pos_definition_index]:
                    fw.write(f"word::: {word}\t\tlemma::: {lemma}\t\tpos::: {pos}\t\t"
                             f"definition::: {definition}\t\texample::: {example}\n")
                    examples_list.append(example)
                    words_list.append(word)
                if fw2 is not None:
                    examples = "\t\t".join(examples_list)
                    words = "\t".join(words_list)
                    fw2.write(f"word::: {words}\t\tlemma::: {lemma}\t\tpos::: {pos}\t\tdefinition::: {definition}\n")
                    fw2.write(f"Reference examples::: {examples}\n")
    if fw2 is not None:
        fw2.close()

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


# statistics of lemmas over the number of definitions
lemma_num_definition_list = [len(definitions) for lemma, definitions in lemma_definition_dict.items()]

counter = Counter(lemma_num_definition_list)
counter_sorted = sorted(counter.items(),key=lambda x:x[0],reverse=False)
fw_log.write(f'the statistics of lemmas over the number of definition.\n')

# print(counter_sorted)
for e in counter_sorted:
    fw_log.write(f'{e[0]}, {e[1]}\n')

# statistics of lemmas over pos
lemma_num_pos_list = [len(set([definition[1] for definition in definitions]))
                      for lemma, definitions in lemma_definition_dict.items()]
counter = Counter(lemma_num_pos_list)
counter_sorted = sorted(counter.items(),key=lambda x:x[0],reverse=False)

fw_log.write(f'the statistics of lemmas over the number of pos.\n')
for e in counter_sorted:
    fw_log.write(f'{e[0]}, {e[1]}\n')

fw_log.close()

