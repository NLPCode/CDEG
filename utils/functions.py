# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 4:42 PM
# @Author  : He Xingwei
import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from nltk.corpus import stopwords as nltk_stopwords

stopwords = set(nltk_stopwords.words('english'))

def set_seed(seed, n_gpu=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def remove_punctuation_words(words):
    """
    remove punctuations from the given words
    :param words: list of word
    :return: list
    """
    new_word_list = []
    for word in words:
        word = ''.join([char for char in word if char not in punctuation])
        if word !='':
            new_word_list.append(word)
    return new_word_list


def remove_stopwords(words):
    """
    remove the stopwords from the given words
    :param words: list of word
    :return: list
    """
    return [w for w in words if w.lower() not in stopwords]

def count_word_syllables(word):
    """
    Count the number of syllables of the given word.
    Note: this function is more accurate than the syllable 'count' function in syllables_en.py
    (https://github.com/mmautner/readability).
    The number of syllables for some words computed by the 'count' function (in syllables_en.py) is wrong.
    For example: count('the') =0, count('we') =0, count('be') =0
    :param word:
    :return:
    """
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def count_syllables(words):
    """
    Count the number of syllables of the words.
    :param words: list of word
    :return:
    """
    syllableCount = 0
    for word in words:
        syllableCount += count_word_syllables(word)
    return syllableCount

def get_words(text):
    """
    this function is used tokenize the given text
    :param text:
    :param tokenizer:
    :return:
    """
    SPECIAL_CHARS = ['.', ',', '!', '?', ' ','*', '&', '%', '#',  '0', '1','2', '3', '4', '5', '6', '7', '8', '9']
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        # for e in SPECIAL_CHARS:
        #     if e in word:
        #         continue
        # filtered_words.append(word)
        if word in SPECIAL_CHARS:
            pass
        else:
            new_word = word.replace(",","").replace(".","").replace("!","").replace("?","")
            if new_word!='':
                filtered_words.append(new_word)
    return filtered_words

def get_sentences(text):
    """
    tokenize the text at the level of sentences,
    :param text:
    :param sent_tokenizer:
    :return:
    """
    sentences = sent_tokenize(text)
    return sentences


def convert_continues_to_discrete(numbers, start = None, end = None, num_bins = 10):
    """
    This functions discretize continuous numbers.
    :param numbers:
    :param start: the start of the bins
    :param end: the end of the bins
    :param num_bins: the number of bins
    :return: discrete numbers
    """
    max_num = np.max(numbers)
    min_num = np.min(numbers)

    if start is None:
        start = min_num

    if end is None:
         end = max_num

    assert start<end

    bin_width = (end-start)/num_bins
    discrete_labels = []
    for e in numbers:
        label = int((e-start-1e-10)/bin_width)
        if label<0:
            label = 0
        elif label>=num_bins:
            label = num_bins-1
        else:
            pass
        discrete_labels.append(label)
    return np.array(discrete_labels), min_num, max_num, start, end, bin_width


if __name__ == "__main__":
    from syllables_en import count
    for e in "the we be".split():
        if count_word_syllables(e)!= count(e):
            print(e, count_word_syllables(e), count(e))
