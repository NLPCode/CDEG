# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 9:53 PM
# @Author  : He Xingwei
import sys
sys.path.append('../')
from utils.functions import get_words, get_sentences, count_syllables, count_word_syllables, \
    remove_punctuation_words,remove_stopwords
from transformers import BartTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
import os,glob

class WordRank:
    """
    This class is used to compute the lexical complexity based on the word rank for the given sentence.
    WordRank is regarded as a proxy to lexical complexity. We compute a sentence-level measure, that we call
    WordRank, by taking the third-quartile of log-ranks (inverse frequency order) of all words in a sentence.
    """

    def __init__(self, dataset, token_level, use_external_data=False, add_space = True):
        """
        :param dataset: Specify the dataset used to count the number of word.
        :param token_level: if token_level is true, count the number of word. Otherwise, count the number of sub-word.
        """
        self.dataset = dataset
        self.add_space = add_space # whether add a space before the sentence when tokenizing.
        if token_level is True:
            print('Use the BartTokenizer to tokenize sentences. You can freely change it to another suitable tokenizer.')
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            tokenize_func = tokenizer.tokenize
        else:
            tokenize_func = word_tokenize
            self.add_space = False
        if self.add_space:
            print('prepend a space to the sentence')
        self.tokenize_func = tokenize_func

        if use_external_data:
            if token_level is True:
                if self.add_space:
                    word2rank_dict_file = f'../data/one-billion-word_add_space_subword2rank_dict.txt'
                else:
                    word2rank_dict_file = f'../data/one-billion-word_subword2rank_dict.txt'
            else:  # word level
                word2rank_dict_file = f'../data/one-billion-word_word2rank_dict.txt'

            test_files = glob.glob('../../../corpora/one-billion-words/'
                                   '1-billion-word-language-modeling-benchmark-r13output/'
                                   'heldout-monolingual.tokenized.shuffled/*')
            training_files = glob.glob('../../../corpora/one-billion-words'
                                       '/1-billion-word-language-modeling-benchmark-r13output/'
                                       'training-monolingual.tokenized.shuffled/*')
            training_files += test_files
            print(f'Count the number of words from ../../../corpora/one-billion-words, '
                  f'which has {len(training_files)} files.')
        else:
            if token_level is True:
                if self.add_space:
                    word2rank_dict_file = f'../data/{dataset}/add_space_subword2rank_dict.txt'
                else:
                    word2rank_dict_file = f'../data/{dataset}/subword2rank_dict.txt'
            else:  # word level
                word2rank_dict_file = f'../data/{dataset}/word2rank_dict.txt'

            training_files = [f"../data/{dataset}/training.txt"]
            print(f'Count the number of words from {training_files}.')
        if not os.path.exists(word2rank_dict_file):
            print('Create word2rank dictionary:')
            self.compute_word2rank_dict(training_files, word2rank_dict_file)
        print('Loading the word2rank_dict.')
        self.word2rank_dict = self.get_word2rank_dict(word2rank_dict_file)
        self.length = len(self.word2rank_dict)
        print(f'The length of the vocabulary is {self.length}.')

    def compute_word2rank_dict(self, training_files, word2rank_dict_file):
        """
        This function is used to count the number of words in the training set and
        sort them by the word frequency in descending order.
        :param token_level: if token_level is true, count the number of word. Otherwise, count the number of sub-word.
        :param training_file:
        :param word2rank_dict_file:
        :return:
        """
        print("Count the number of words  and "
              "sort them by the word frequency in descending order.")
        vocab = {}
        num_lines = 0

        for training_file in training_files:
            if training_file == f"../data/{self.dataset}/training.txt":
                is_split = True
                print(is_split)
            else:
                is_split = False
            with open(training_file) as fr:
                for line in fr:
                    num_lines+=1
                    line = line.strip()
                    if is_split:
                        word_seg, lemma_seg, pos_seg, definition_seg, example_seg = line.split('\t\t')
                        # prepend a space to the example
                        example =  example_seg.split('example::: ')[1].strip()
                    else:
                        example =  line
                    if self.add_space: # added for bart tokenizer
                        example = ' '+example
                    tokens = self.tokenize_func(example)
                    for token in tokens:
                        vocab[token] = vocab.get(token, 0) + 1

        print(f"Count the number of words from {num_lines} lines.")
        # sort words by frequency in descending order
        vocab = vocab.items()
        vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
        print(f"Output the word and rank into {word2rank_dict_file}.")
        with open(word2rank_dict_file, 'w') as fw:
            rank = 1
            for k, v in vocab:
                fw.write(f'{k}\t\t{rank}\t\t{v}\n')
                rank += 1

    def get_word2rank_dict(self, word2rank_dict_file):
        """
        loading the word2rank_dict
        :param word2rank_dict_file:
        :return:
        """
        word2rank_dict = {}
        with open(word2rank_dict_file, 'r') as fr:
            for rank, line in enumerate(fr):
                word = line.split('\t\t')[0]
                word2rank_dict[word] = rank + 1
        return word2rank_dict

    def get_log_rank(self, word):
        """
        get the log rank for the word
        :param word:
        :return:
        """
        return np.log(self.word2rank_dict.get(word, self.length))

    def get_proportion_high_frequency_word(self, sentence_list, rank_list =[1000, 2000, 5000, 10000]):
        """
        this function is used to get the proportion of high-frequency words.
        :sentence_list: list of sentence
        :param rank_list: each number in the list is regarded as the threshold of  high-frequency words.
        :return:
        """
        num_low_frequenct_word_list = [0]*len(rank_list)
        num_word = 0.0
        for sentence in sentence_list:
            tokens = self.tokenize_func(sentence)
            tokens = remove_punctuation_words(tokens)
            # remove stop words
            tokens = remove_stopwords(tokens)
            num_word += len(tokens)
            for token in tokens:
                _rank = self.word2rank_dict.get(token, self.length)
                for i, rank in enumerate(rank_list):
                    if _rank<=rank:
                        num_low_frequenct_word_list[i] += 1
        return [e/num_word for e in num_low_frequenct_word_list]

    def get_lexical_complexity_score(self, text):
        """
        this function is used to compute the lexical complexity for the given sentence.
        :param word2rank_dict:
        :param sentence:
        :return:
        """
        if self.add_space:
            text = ' ' + text

        # tokenize
        words = self.tokenize_func(text)
        # remove punctuation tokens
        words = remove_punctuation_words(words)
        # remove stop words
        words = remove_stopwords(words)

        ranks = [self.get_log_rank(word) for word in words if word in self.word2rank_dict]
        if len(ranks) == 0:
            return np.log(1 + self.length)  # TODO: This is completely arbitrary
        return round(np.quantile(ranks, 0.75), 3)


class Readability:
    def __init__(self):
        pass
    @staticmethod
    def analyze_text(text):
        """
        Count the number of words, sentences, syllables of the text.
        :param text:
        :return:
        """
        text = text.lower()
        words = get_words(text)
        # # tokenize
        # words = word_tokenize(text.strip())
        # # remove punctuation tokens
        words = remove_punctuation_words(words)
        # # remove stop words
        words = remove_stopwords(words)

        total_words = len(words)
        total_sentences = len(get_sentences(text))
        total_syllables = count_syllables(words)
        return total_words, total_sentences, total_syllables

    @staticmethod
    def FleschReadingEase(total_words, total_sentences, total_syllables):
        """
        In the Flesch reading-ease test, higher scores indicate material that is easier to read;
        lower numbers mark passages that are more difficult to read.
        :param total_words:
        :param total_sentences:
        :param total_syllables:
        :return:
        """
        if total_sentences ==0 or total_words==0:
            score = 206.835 - (1.015 * 1) - (84.6 * (1))
        else:
            score = 206.835 - (1.015 * total_words / total_sentences) - (84.6 * (total_syllables / total_words))
        return round(score, 3)

    @staticmethod
    def FleschKincaidGradeLevel(total_words, total_sentences, total_syllables):
        """
        The Flesch–Kincaid readability tests are readability tests designed to
        indicate how difficult a passage in English is to understand.
        See details at https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests.
        :param total_words:
        :param total_sentences:
        :param total_syllables:
        :return:
        """
        if total_sentences ==0 or total_words==0:
            score = 0.39 * (1) + 11.8 * (1) - 15.59
        else:
            score = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        return round(score, 3)

if __name__ == "__main__":
    text ='I want to go home, which is located in Beijing.'

    total_words, total_sentences, total_syllables = Readability.analyze_text(text)
    print(total_words, total_sentences, total_syllables)
    print(Readability.FleschReadingEase(total_words, total_sentences, total_syllables))
    print(Readability.FleschKincaidGradeLevel(total_words, total_sentences, total_syllables))