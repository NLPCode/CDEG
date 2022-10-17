'''
This file is used to select examples, which contain the given word.
'''
import json
from pattern.en import conjugate, pluralize, comparative, superlative
from lemminflect import getInflection, getAllInflections
import inflect
import os, shutil

p = inflect.engine()

def process_example(example):
    example = example.strip()
    example = example[0].upper()+example[1:]
    if example[-1].isalnum():
        example = example+'.'
    example = example.replace('“','"').replace('”', '"').replace('‘', '\'').replace('’', '\'')
    return example

def get_single_word_forms(word, pos):
    """
    get the conjugated form of the verb, the plural form of the noun
    or the comparative and superlative forms of the adjective.
    :param word:
    :param pos:
    :return: list
    """
    if pos.lower() == 'noun':
        pos = 'NOUN'
    elif pos.lower() == 'verb':
        pos = 'VERB'
    elif pos.lower() == 'adjective':
        pos = 'ADJ'
    else:
        pos = None
    s = set()
    if pos:
        ans = getAllInflections(word, upos=pos)
    else:
        ans = getAllInflections(word)
    for i in ans.values():
        for j in i:
            if j!=word:
                s.add(j)
    return [word,]+list(s)

def get_verb_conjugation(word):
    """
    This function aims to get conjugated forms of the given verb.
    word: str
    return: sorted list
    """
    if word == 'bias':
        return ["biases", "biasing", "biased"]
    if word == 'barbecue':
        return ["barbecues", "barbecuing", "barbecued"]
    if word == 'draft':
        return ["drafts", "drafting", "drafted"]
    if word == 'dream':
        return ["dreams", "dreaming", "dreamt",'dreamed']
    if word == 'feed':
        return ["feeds", "feeding", "fed"]
    s = set()
    for alias in ["inf","1sg",'2sg','3sg','pl','part','p','1sgp','2sgp','3gp','ppl','ppart']:
        w = conjugate(word,alias)
        if w:
            s.add (w)
    return sorted(list(s))

def get_word_forms_v2(word, pos):
    if pos.lower() == 'verb':
        l = get_verb_conjugation(word)
    elif pos.lower() == 'noun':
        l = [pluralize(word),   p.plural_noun(word)]
    elif pos.lower() == 'Adjective':
        l = [comparative(word), superlative(word)]
    else:
        l = []
    new_l = [word]
    for e in list(set(l)):
        if e!=word:
            new_l.append(e)
    return new_l

def get_multiple_word_forms(word, pos):
    l = []
    words = word.split()
    single_word_forms = get_single_word_forms(words[-1], pos)
    single_word_forms2 = get_word_forms_v2(words[-1], pos)
    single_word_forms = list(set(single_word_forms2) | set(single_word_forms))
    for e in single_word_forms:
        words[-1] = e
        l.append(" ".join(words))

    words = word.split()
    single_word_forms = get_single_word_forms(words[0], pos)
    single_word_forms2 = get_word_forms_v2(words[0], pos)
    single_word_forms = list(set(single_word_forms2) | set(single_word_forms))
    for e in single_word_forms:
        words[0] = e
        l.append(" ".join(words))
    l = list(set(l))
    if word not in l:
        l = [word, ]+l
    return l

def check(word, pos, example):
    """
    this function aims to check whether the example contains the given word.
    :param word:
    :param pos: part of speech
    :param example:
    :return: True or False
    """

    words = word.split()
    if len(words)==1:
        example = ' ' + example + ' '
        single_word_forms = get_single_word_forms(word, pos)
        single_word_forms2 = get_word_forms_v2(word, pos)
        single_word_forms += list(set(single_word_forms2)-set(single_word_forms))
        # single_word_forms = [word,]
        new_l = []
        for w in single_word_forms:
            new_l.append(w)         # original form
            new_l.append(w.capitalize()) # Hello world
            new_l.append(w.title()) # Hello World
            new_l.append(" ".join(w.split('-'))) # Hello World
            new_l.append("".join(w.split('-'))) # Hello World
            new_l.append(w.lower()) # hello world
            new_l.append(w.upper())  # HELLO WORLD
        single_word_forms = list(set(new_l))
        for w in single_word_forms:
            start = 0
            while True:
                start = example.find(w, start)
                if start!=-1:
                    start_char = example[start-1]
                    end_char = example[start+len(w)]
                    if not start_char.isalnum() and not end_char.isalnum():
                        if start_char == '-' and word[0]=='-':
                            w = '-'+w

                        if w == word:
                            return 1, w
                        else:
                            return 2, w
                    start += len(w)
                else:
                    break
        return -1, word

    else:
        example = ' ' + example +' '
        l = get_multiple_word_forms(word, pos)
        # l = [word,]
        new_l = []
        for w in l:
            new_l.append(w)         # original form
            new_l.append(w.capitalize()) # Hello world
            words = [e.capitalize() for e in w.split()]
            # new_l.append(w.title()) # Hello World
            new_l.append(" ".join(words)) # Hello World
            new_l.append(w.lower()) # hello world
            new_l.append(w.upper())  # HELLO WORLD
        l = list(set(new_l))
        new_l = []
        for w in l:
            words = w.split()
            new_l += [w, "".join(words),'-'.join(words)]

        l = new_l
        for w in l:
            start = 0
            while True:
                start = example.find(w, start)
                if start!=-1:
                    start_char = example[start-1]
                    end_char = example[start+len(w)]
                    # if word == 'bench press':
                    #     print(start, example, w, start_char, end_char)
                    if not start_char.isalnum() and not end_char.isalnum():
                        return 3, w
                    start += len(w)
                else:
                    break
        return -2, word

# if __name__ == '__main___':
word_pos_dict = {}
contain_word = {'phrase':0,'headword':0}
contain_example = {'phrase':0,'headword':0}
contain_definition = {'phrase':0,'headword':0}
not_contain_word = {'phrase':0,'headword':0}
not_contain_example = {'phrase':0,'headword':0}
not_contain_definition = {'phrase':0,'headword':0}

contain_word_len = {}
contain_example_len = {}
contain_definition_len = {}
not_contain_word_len = {}
not_contain_example_len = {}
not_contain_definition_len = {}
if os.path.exists('../data/oxford/word_definition_example'):
    shutil.rmtree('../data/oxford/word_definition_example')
os.mkdir('../data/oxford/word_definition_example')
with open('../data/oxford/oxford_all.json','r') as fr, open('../data/oxford/statistic.csv','a') as fw:

    for instance in fr:
        instance = json.loads(instance)
        _type = instance['type']
        # if _type == 'phrase':
        #     continue
        word = instance['word']
        if word in ['Bell','Hero','ICE']:
            word = word.lower()
        flag_word = 0
        for i, definition_examples in enumerate(instance['senses']):
            definition = definition_examples['definition']
            pos = definition_examples['pos']
            word_pos_dict[f'{word}_|_{pos}'] = 1
            flag_definition = 0
            for j, example in enumerate(definition_examples['examples']):
                example = process_example(example)
                flag, w = check(word, pos,example )
                # if word =='be' and flag>1:
                #     print(flag, w, example)
                if flag<0:
                    not_contain_example[_type] += 1
                else:
                    flag_definition = 1
                    flag_word = 1
                    contain_example[_type] +=1
                _length = len(word.split())
                if _type == 'headword':
                    if flag>0:
                        contain_example_len[_length] = contain_example_len.get(_length,0)+1
                        with open(f'../data/oxford/word_definition_example/{_length}words.txt','a') as fw2:
                            fw2.write(f"word::: {w}\t\tlemma::: {word}\t\tpos::: {pos}\t\tdefinition {i+1}::: {definition}\t\texample {j+1}::: {example}\n")
                    else:
                        with open(f'../data/oxford/word_definition_example/{_length}words_not_contain.txt','a') as fw2:
                            fw2.write(f"word::: {w}\t\tlemma::: {word}\t\tpos::: {pos}\t\tdefinition {i+1}::: {definition}\t\texample {j+1}::: {example}\n")
                        not_contain_example_len[_length] = not_contain_example_len.get(_length,0)+1
            if flag_definition:
                if _type == 'headword':
                    contain_definition_len[_length] = contain_definition_len.get(_length,0)+1
                contain_definition[_type] += 1
            else:
                if _type == 'headword':
                    not_contain_definition_len[_length] = not_contain_definition_len.get(_length,0)+1
                not_contain_definition[_type] +=1
        if flag_word:
            if _type == 'headword':
                contain_word_len[_length] = contain_word_len.get(_length,0)+1
            contain_word[_type] +=1
        else:
            if _type == 'headword':
                not_contain_word_len[_length] = not_contain_word_len.get(_length,0)+1
            not_contain_word[_type] += 1
    # exit(0)
    fw.write('\n')
    fw.write(f'Word types, headword, phrase, total\n')
    fw.write(' ,Examples containing the given word, Examples not containing the given word, Total, Examples containing the given word, Examples not containing the given word, Total\n')
    a,b,c,d= contain_word['headword'],not_contain_word['headword'], contain_word['phrase'], not_contain_word['phrase']
    fw.write(f'The number of words/phrases, {a},{b},{a+b}, {c},{d},{c+d}, {a+b+c+d}\n')
    a,b,c,d= contain_definition['headword'],not_contain_definition['headword'], contain_definition['phrase'], not_contain_definition['phrase']
    fw.write(f'The number of definitions, {a},{b},{a+b}, {c},{d},{c+d},{a+b+c+d}\n')
    a,b,c,d= contain_example['headword'],not_contain_example['headword'], contain_example['phrase'], not_contain_example['phrase']
    fw.write(f'The number of examples, {a},{b},{a+b}, {c},{d},{c+d},{a+b+c+d}\n\n')

    keys = sorted(list(set(list(contain_example_len.keys())+ list(not_contain_example_len.keys()))))

    for n in keys:
        a,b = contain_example_len.get(n,0), not_contain_example_len.get(n,0)
        fw.write(f'N={n},{a},{b},{a+b}\n')
    a,b = sum(contain_example_len.values()), sum(not_contain_example_len.values())
    fw.write(f'Total,{a},{b},{a+b}\n\n')
    for n in keys:
        a,b = contain_definition_len.get(n,0), not_contain_definition_len.get(n,0)
        fw.write(f'N={n},{a},{b},{a+b}\n')
    a,b = sum(contain_definition_len.values()), sum(not_contain_definition_len.values())
    fw.write(f'Total,{a},{b},{a+b}\n\n')
    for n in keys:
        a,b = contain_word_len.get(n,0), not_contain_word_len.get(n,0)
        fw.write(f'N={n},{a},{b},{a+b}\n')
    a,b = sum(contain_word_len.values()), sum(not_contain_word_len.values())
    fw.write(f'Total,{a},{b},{a+b}\n\n')

