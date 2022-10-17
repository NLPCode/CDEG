# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 2:19 PM
# @Author  : He Xingwei

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import numpy as np
import time
import os
import sys
import argparse
from collections import Counter

sys.path.append('../')
from utils.log import Logger
from utils.functions import set_seed, convert_continues_to_discrete


class BARTDataset(Dataset):
    """
    this dataset is for training/validation/testing with the cross entropy loss.
    """

    def __init__(self, dataset, mode, tokenizer, use_word, use_definition, use_pos, use_example_len, use_lexical_complexity,
                 max_example_len, num_bins, add_space=0, dataset_percent=1):
        """

        :param dataset:
        :param mode: values in [training, test, validation]
        :param tokenizer:
        :param use_word: int
        :param use_definition: int 
        :param use_pos: int
        :param use_example_len: int
        :param use_lexical_complexity: int
        :param max_example_len: int
        :param num_bins: int
        :param add_space: int
        :param dataset_percent: float
        """
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_example_len = max_example_len
        word_list = []
        lemma_list = []
        definition_list = []
        example_list = []

        self.word_tokenization_list = []
        self.lemma_tokenization_list = []
        self.pos_list = []
        self.definition_tokenization_list = []
        self.example_tokenization_list = []
        # the example length feature
        self.example_tokenization_length_list = []
        # lexical complexity features
        self.word_rank_lexical_complexity_list = []
        self.token_rank_lexical_complexity_list = []
        self.external_word_rank_lexical_complexity_list = []
        self.external_token_rank_lexical_complexity_list = []
        self.flesch_reading_ease_list = []
        self.flesch_kincaid_grade_level_list = []

        if add_space:
            data_dict_path = f'{dataset}/{mode}_features_add_space.pt'
        else:
            data_dict_path = f'{dataset}/{mode}_features.pt'
        # Get special tokens, which we added to the tokenizer
        special_token_dict = tokenizer.get_added_vocab()
        print(special_token_dict)
        if os.path.exists(data_dict_path):
            print(f'Loading data from  {data_dict_path}.')
            data_dict = torch.load(data_dict_path)
            self.word_tokenization_list = data_dict['word_tokenization_list']
            self.lemma_tokenization_list = data_dict['lemma_tokenization_list']
            self.pos_list = data_dict['pos_list']
            self.definition_tokenization_list = data_dict['definition_tokenization_list']
            self.example_tokenization_list = data_dict['example_tokenization_list']

            # the example length feature
            self.example_tokenization_length_list = data_dict['example_tokenization_length_list']
            # lexical complexity features
            self.word_rank_lexical_complexity_list = data_dict['word_rank_lexical_complexity_list']
            self.token_rank_lexical_complexity_list = data_dict['token_rank_lexical_complexity_list']
            self.external_word_rank_lexical_complexity_list = data_dict['external_word_rank_lexical_complexity_list']
            self.external_token_rank_lexical_complexity_list = data_dict['external_token_rank_lexical_complexity_list']
            self.flesch_reading_ease_list = data_dict['flesch_reading_ease_list']
            self.flesch_kincaid_grade_level_list = data_dict['flesch_kincaid_grade_level_list']
        else:
            print(f'Tokenize the {mode} set.')
            filename = f'{dataset}/{mode}_features.txt'
            with open(filename, 'r') as fr:
                for line in fr:
                    line = line.strip()
                    word_seg, lemma_seg, pos_seg, definition_seg, example_seg, \
                    word_rank_lexical_complexity_seg, token_rank_lexical_complexity_add_space_seg, \
                    token_rank_lexical_complexity_seg, \
                    external_word_rank_lexical_complexity_seg, external_token_rank_lexical_complexity_add_space_seg, \
                    external_token_rank_lexical_complexity_seg, \
                    flesch_reading_ease_seg, flesch_kincaid_grade_level_seg = line.split('\t\t')
                    word = word_seg.split("word::: ")[1]
                    lemma = lemma_seg.split('lemma::: ')[1]
                    pos = pos_seg.split('pos::: ')[1]
                    definition = definition_seg.split('definition::: ')[1]
                    example = example_seg.split('example::: ')[1]

                    word_rank_lexical_complexity = \
                        float(word_rank_lexical_complexity_seg.split('word_rank_lexical_complexity::: ')[1])
                    token_rank_lexical_complexity_add_space = \
                        float(token_rank_lexical_complexity_add_space_seg.split('token_rank_lexical_complexity_add_space::: ')[1])
                    token_rank_lexical_complexity = \
                        float(token_rank_lexical_complexity_seg.split('token_rank_lexical_complexity::: ')[1])

                    external_word_rank_lexical_complexity = \
                        float(external_word_rank_lexical_complexity_seg.split
                              ('external_word_rank_lexical_complexity::: ')[1])
                    external_token_rank_lexical_complexity_add_space = \
                        float(external_token_rank_lexical_complexity_add_space_seg.split
                              ('external_token_rank_lexical_complexity_add_space::: ')[1])
                    external_token_rank_lexical_complexity = \
                        float(external_token_rank_lexical_complexity_seg.split('external_token_rank_lexical_complexity::: ')[1])

                    flesch_reading_ease = float(flesch_reading_ease_seg.split('flesch_reading_ease::: ')[1])
                    flesch_kincaid_grade_level = float(flesch_kincaid_grade_level_seg.split('flesch_kincaid_grade_level::: ')[1])

                    word_list.append(word)
                    lemma_list.append(lemma)
                    self.pos_list.append(pos)
                    definition_list.append(definition)
                    example_list.append(example)

                    self.word_rank_lexical_complexity_list.append(word_rank_lexical_complexity)
                    self.external_word_rank_lexical_complexity_list.append(external_word_rank_lexical_complexity)
                    if add_space:
                        self.token_rank_lexical_complexity_list.append(token_rank_lexical_complexity_add_space)
                        self.external_token_rank_lexical_complexity_list.append(external_token_rank_lexical_complexity_add_space)
                    else:
                        self.token_rank_lexical_complexity_list.append(token_rank_lexical_complexity)
                        self.external_token_rank_lexical_complexity_list.append(external_token_rank_lexical_complexity)

                    self.flesch_reading_ease_list.append(flesch_reading_ease)
                    self.flesch_kincaid_grade_level_list.append(flesch_kincaid_grade_level)

            # tokenize the word, lemma, definition, example
            for word, lemma, definition, example in zip(word_list, lemma_list, definition_list, example_list):
                if add_space:
                    word = ' ' + word
                    lemma = ' ' + lemma
                    definition = ' ' + definition
                    example = ' ' + example

                ids = tokenizer.encode(word, add_special_tokens=False)
                self.word_tokenization_list.append(ids)

                ids = tokenizer.encode(lemma, add_special_tokens=False)
                self.lemma_tokenization_list.append(ids)

                ids = tokenizer.encode(definition, add_special_tokens=True)
                self.definition_tokenization_list.append(ids)

                ids = tokenizer.encode(example, add_special_tokens=True)
                self.example_tokenization_list.append(ids)

                # the example length feature
                self.example_tokenization_length_list.append(len(ids) - 2)

            data_dict = {
                'word_tokenization_list': self.word_tokenization_list,
                'lemma_tokenization_list': self.lemma_tokenization_list,
                'pos_list': self.pos_list,
                'definition_tokenization_list': self.definition_tokenization_list,
                'example_tokenization_list': self.example_tokenization_list,
                'example_tokenization_length_list': self.example_tokenization_length_list,
                'word_rank_lexical_complexity_list': self.word_rank_lexical_complexity_list,
                'token_rank_lexical_complexity_list': self.token_rank_lexical_complexity_list,
                'external_word_rank_lexical_complexity_list': self.external_word_rank_lexical_complexity_list,
                'external_token_rank_lexical_complexity_list': self.external_token_rank_lexical_complexity_list,
                'flesch_reading_ease_list': self.flesch_reading_ease_list,
                'flesch_kincaid_grade_level_list': self.flesch_kincaid_grade_level_list
            }

            torch.save(data_dict, data_dict_path)

        lexical_complexity_feature_dict = {
            1: (self.word_rank_lexical_complexity_list, 'word_rank_lexical_complexity'),
            2: (self.token_rank_lexical_complexity_list, 'token_rank_lexical_complexity'),
            3: (self.external_word_rank_lexical_complexity_list, 'external_word_rank_lexical_complexity'),
            4: (self.external_token_rank_lexical_complexity_list, 'external_token_rank_lexical_complexity'),
            5: (self.flesch_reading_ease_list, 'flesch_reading_ease'),
            6: (self.flesch_kincaid_grade_level_list, 'flesch_kincaid_grade_level')
        }

        if add_space:
            _filename = f"{dataset}/training_features_interval_add_space.txt"
        else:
            _filename = f"{dataset}/training_features_interval.txt"

        features_interval_dict = {}
        if os.path.exists(_filename):
            with open(_filename, 'r') as fr:
                for i, line in enumerate(fr):
                    feature_name, min_feature_value, max_feature_value = line.strip().split('\t\t')
                    min_feature_value = float(min_feature_value)
                    max_feature_value = float(max_feature_value)
                    features_interval_dict[feature_name] = (min_feature_value, max_feature_value)
        else:
            if mode != 'training':
                print(f'Please create the {_filename} first with the training set.')
                exit()
            else:
                print(f'Create {_filename}.')
                with open(_filename, 'w') as fw:
                    _max_example_len = np.max(self.example_tokenization_length_list)
                    _min_example_len = np.min(self.example_tokenization_length_list)
                    example_feature_name = 'example_length'
                    features_interval_dict[example_feature_name] = (_min_example_len, _max_example_len)
                    fw.write(f'{example_feature_name}\t\t{_min_example_len}\t\t{_max_example_len}\n')
                    for k in range(1, 7):
                        lexical_complexity_feature_list, feature_name = lexical_complexity_feature_dict[k]
                        max_feature_value = np.max(lexical_complexity_feature_list)
                        min_feature_value = np.min(lexical_complexity_feature_list)
                        fw.write(f'{feature_name}\t\t{min_feature_value}\t\t{max_feature_value}\n')
                        features_interval_dict[feature_name] = (min_feature_value, max_feature_value)

        print(features_interval_dict)

        if use_example_len:
            feature_name = 'example_length'
            min_feature_value, max_feature_value = features_interval_dict[feature_name]
            print(f'The min and max values of {feature_name} are {min_feature_value}, {max_feature_value}.')
        if use_lexical_complexity > 0:
            lexical_complexity_feature_list, feature_name = lexical_complexity_feature_dict[use_lexical_complexity]
            min_feature_value, max_feature_value = features_interval_dict[feature_name]
            print(f'The min and max values of {feature_name} are {min_feature_value}, {max_feature_value}.')
            # discretize the lexical complexity features
            start = time.time()
            lexical_complexity_feature_list, min_num, max_num, start, end, bin_width = \
                convert_continues_to_discrete(lexical_complexity_feature_list, start=min_feature_value,
                                              end=max_feature_value, num_bins=num_bins)

            print(f'Discretizing the lexical complexity feature, {feature_name}, uses {time.time() - start:.1f} seconds.')
            print(f'Unique labels: {np.unique(lexical_complexity_feature_list)}.')
            print(f'The min number is {min_num:.3f}, max number is {max_num:.3f}, '
                  f'the start of the bins is {start:.3f}, the end of the bins is {end:.3f}, and the bin width is {bin_width:.3f}.')

        # create encoder inputs
        self.encoder_input_list = []
        for i, (word, lemma, pos, definition) in enumerate(zip(self.word_tokenization_list, self.lemma_tokenization_list,
                                                               self.pos_list, self.definition_tokenization_list)):
            encoder_input = []
            if use_word == 0:
                encoder_input += [special_token_dict['<word>']] + word
            elif use_word == 1:
                encoder_input += [special_token_dict['<lemma>']] + lemma
            elif use_word == 2:
                encoder_input += [special_token_dict['<word>']] + word + [special_token_dict['<lemma>']] + lemma
            else:
                pass

            if use_pos:
                encoder_input += [special_token_dict['<pos>'], special_token_dict[f'<pos_{pos}>']]

            if use_example_len:
                length = self.example_tokenization_length_list[i]
                encoder_input += [special_token_dict['<example_len>'],
                                  special_token_dict.get(f'<example_len_{length}>',
                                                         special_token_dict[f'<example_len_{max_example_len}>'])
                                  ]

            if use_lexical_complexity > 0:
                lexical_complexity_label = lexical_complexity_feature_list[i]
                encoder_input += [special_token_dict['<lexical_complexity>'],
                                  special_token_dict[f'<lexical_complexity_{lexical_complexity_label}>']]
            if use_definition:
                encoder_input += [special_token_dict['<definition>']] + definition
            self.encoder_input_list.append(encoder_input)

        self.len = len(self.encoder_input_list)

        # output the statistics of discrete feature labels
        # if not os.path.exists(f"{dataset}/{mode}_features_statistics.txt"):
        if add_space:
            _filename = f"{dataset}/{mode}_features_statistics_add_space.txt"
        else:
            _filename = f"{dataset}/{mode}_features_statistics.txt"

        with open(_filename, 'w') as fw:
            # the statistics of pos labels
            fw.write(f'the statistics of pos labels\n')
            counter = Counter(self.pos_list)
            counter_sorted = sorted(counter.items(), key=lambda x: x[0], reverse=False)
            # print(counter_sorted)
            for e in counter_sorted:
                fw.write(f'{e[0]}, {e[1]}\n')

            # the statistics of example lengths
            fw.write(f'the statistics of example lengths\n')
            counter = Counter(self.example_tokenization_length_list)
            counter_sorted = sorted(counter.items(), key=lambda x: x[0], reverse=False)
            # print(counter_sorted)
            for e in counter_sorted:
                fw.write(f'{e[0]}, {e[1]}\n')
            for i in range(1, 7):
                # the statistics of lexical complexity labels
                lexical_complexity_feature_list, feature_name = lexical_complexity_feature_dict[i]
                min_feature_value, max_feature_value = features_interval_dict[feature_name]
                # discretize the lexical complexity features
                lexical_complexity_feature_list, min_num, max_num, start, end, bin_width = convert_continues_to_discrete(
                    lexical_complexity_feature_list, start=min_feature_value,
                    end=max_feature_value, num_bins=num_bins)
                fw.write(f'the statistics of {feature_name} labels\n')
                fw.write(f'The min number is {min_num}, max number is {max_num}, '
                         f'the start of the bins is {start}, the end of the bins is {end}, and the bin width is {bin_width}.\n')
                counter = Counter(lexical_complexity_feature_list)
                counter_sorted = sorted(counter.items(), key=lambda x: x[0], reverse=False)
                # print(counter_sorted)
                for e in counter_sorted:
                    fw.write(f'{e[0]}, {e[1]}\n')

        if dataset_percent < 1 and self.mode == 'training':  # use parts of the training data to train the model
            sampled_index = np.random.choice(np.arange(self.len), int(self.len * dataset_percent))

            encoder_input_list = []
            example_tokenization_list = []
            for i in sampled_index:
                encoder_input_list.append(self.encoder_input_list[i])
                example_tokenization_list.append(self.example_tokenization_list[i])
            self.encoder_input_list = encoder_input_list
            self.example_tokenization_list = example_tokenization_list

            self.len = len(self.encoder_input_list)
            print(f'Using {self.len} data instances to train the model.')

    def __getitem__(self, idx):
        return torch.tensor(self.encoder_input_list[idx], dtype=torch.long), \
               torch.tensor(self.example_tokenization_list[idx], dtype=torch.long)

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_input_list = [s[0] for s in samples]
        decoder_input_list = [s[1][:-1] for s in samples]
        decoder_label_list = [s[1][1:] for s in samples]
        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_input_list, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)
        encoder_inputs = pad_sequence(encoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_labels = pad_sequence(decoder_label_list, batch_first=True, padding_value=-100)

        return encoder_inputs, attention_mask, decoder_inputs, decoder_labels

    def evaluate(self, model, local_rank, mode, dataloader):
        """
        compute the average loss over the test or validation set.
        :param model:
        :param local_rank:
        :param dataloader:
        :return:
        """
        datasize = self.len
        model.eval()
        total_loss = 0
        total_tokens = 0
        step = 0
        start = time.time()
        with torch.no_grad():
            for data in dataloader:
                data = [t.to(device) for t in data]
                encoder_inputs, attention_mask, decoder_inputs, decoder_labels = data
                loss, logits = model(encoder_inputs, attention_mask=attention_mask,
                                     decoder_input_ids=decoder_inputs, labels=decoder_labels,
                                     output_attentions=True)[:2]
                bts = encoder_inputs.shape[0]
                num_tokens = torch.sum(decoder_labels != -100)
                total_loss += loss * num_tokens
                total_tokens += num_tokens
                step += bts
                if local_rank in [-1, 0]:
                    print(
                        f'\r   Evaluating on the {mode} set for {step}/{datasize / torch.cuda.device_count()} '
                        f'takes {time.time() - start:.1f} seconds.', end='')

            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce_multigpu([total_loss])
            total_loss = total_loss.item()

            average_loss = total_loss / total_tokens
            used_time = time.time() - start
            print()
        model.train()
        return average_loss, used_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
    parser.add_argument('--dataset', type=str, default='oxford', help='the path of the dataset.')
    parser.add_argument('--dataset_percent', type=float, default=1, help='The percentage of data used to train the model.')
    parser.add_argument('--initialization', type=str, default='bart-base',
                        choices=['bart-random-base', 'bart-base', 'bart-large'],
                        help='initialize the model with random values, bart-base or bart-large.')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--test_batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate for training.')
    parser.add_argument('--train', type=int, default=1, choices=[0, 1], help='1 for training, 0 for testing.')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='1', help='The ids of gpus for training.')


    # features
    parser.add_argument('--add_space', type=int, default=1, choices=[0, 1],
                        help='Whether add a space before the example, word and lemma '
                             'so that the tokens of the word do appear in the token sequence of the example.')
    parser.add_argument('--use_word', type=int, default=0, choices=[0, 1, 2, 3],
                        help='whether use the word or lemma as the input of the encoder.'
                             '0 for word; 1 for lemma; 2 for both word and lemma; 3 for not using both.')
    parser.add_argument('--use_definition', type=int, default=1, choices=[0, 1],
                        help='whether use the pos as the input of the encoder.')
    parser.add_argument('--use_pos', type=int, default=1, choices=[0, 1],
                        help='whether use the pos as the input of the encoder.')
    parser.add_argument('--use_example_len', type=int, default=0, choices=[0, 1],
                        help='whether use the length of examples as the input of the encoder.')

    parser.add_argument('--use_lexical_complexity', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help='0 denotes the lexical complexity is not used as the input of the encoder;'
                             '1 denotes the word_rank_lexical_complexity is regarded as the lexical complexity;'
                             '2 denotes the token_rank_lexical_complexity is regarded as the lexical complexity;'
                             '3 denotes the external_word_rank_lexical_complexity is regarded as the lexical complexity;'
                             '4 denotes the external_token_rank_lexical_complexity is regarded as the lexical complexity;'
                             '5 denotes the flesch_reading_ease is regarded as the lexical complexity;'
                             '6 denotes the flesch_kincaid_grade_level is regarded as the lexical complexity;')
    parser.add_argument('--num_bins', type=int, default=40,
                        help='the number of bins for lexical complexity features.')
    parser.add_argument('--max_example_len', type=int, default=60,
                        help='the max length of the dictionary examples.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)

    prefix = f'lr_{args.lr}'

    if args.add_space:
        prefix += f"_add_space"

    if args.dataset_percent < 1:
        prefix += f'_data_percent_{args.dataset_percent}'

    if args.use_word == 0:
        prefix += '_use_word'
    elif args.use_word == 1:
        prefix += '_use_lemma'
    elif args.use_word == 2:
        prefix += '_use_word_lemma'
    else:
        pass

    # if args.use_definition == 1:
    #     prefix += '_use_def'

    if args.use_pos:
        prefix += '_use_pos'

    if args.use_example_len:
        prefix += f'_use_example_len_max_len_{args.max_example_len}'

    if args.use_lexical_complexity > 0:
        prefix += f'_lexical_complexity_{args.use_lexical_complexity}_num_bins_{args.num_bins}'

    # we only extract word with the pos in extracted_pos
    extracted_pos = ['Noun', 'Adjective', 'Verb', 'Adverb', 'Preposition',
                     'Interjection', 'Numeral', 'Pronoun', 'Determiner', 'Conjunction']

    model_path = f'../checkpoints/{args.dataset}_{args.initialization}/{prefix}'
    log_path = f'../logs/{args.dataset}_{args.initialization}'
    args.dataset = f'../data/{args.dataset}'
    if args.local_rank in [-1, 0]:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = '{}/{}.log'.format(log_path, prefix)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        args.model_path = model_path
        args.log_file = log_file
        logger = Logger(log_file)
        logger.logger.info(f'The log file is {log_file}.')
        logger.logger.info(f'The checkpoint path is {model_path}.')
        logger.logger.info(args)
        if args.train:
            logger.logger.info('Use {} gpus to train the model.'.format(args.n_gpu))
        else:
            logger.logger.info('Use {} gpus to evaluate the model.'.format(args.n_gpu))

    try:
        # load the pre-trained model and tokenizer
        tokenizer = BartTokenizer.from_pretrained(args.model_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_path)
        if args.local_rank in [-1, 0]:
            logger.logger.info('Initialize BartForConditionalGeneration from checkpoint {}.'.format(args.model_path))
    except:
        if args.initialization == "bart-random-base":
            tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')
            #  load pre-trained config
            config = BartConfig.from_pretrained(f'facebook/bart-base')
            # pass the config to model constructor instead of from_pretrained
            # this creates the model as per the params in config
            # but with weights randomly initialized
            model = BartForConditionalGeneration(config)
        else:
            tokenizer = BartTokenizer.from_pretrained(f'facebook/{args.initialization}')
            model = BartForConditionalGeneration.from_pretrained(f'facebook/{args.initialization}')

        if args.local_rank in [-1, 0]:
            logger.logger.info(f'Initialize BartForConditionalGeneration with default parameters {args.initialization}.')
        # add special tokens to the vocabulary
        special_tokens = []
        # add <word>, <lemma>, <pos>, <definition>
        special_tokens.append('<word>')
        special_tokens.append('<lemma>')
        special_tokens.append('<pos>')
        special_tokens.append('<definition>')
        # add <length>, which denotes the length of the example.
        special_tokens.append('<example_len>')
        # add <lexical_complexity>, which denotes the lexical complexity of the example.
        special_tokens.append('<lexical_complexity>')

        # the pos of word
        for pos in extracted_pos:
            special_tokens.append(f'<pos_{pos}>')

        # the length of the dictionary example
        for length in range(1, args.max_example_len + 1):
            special_tokens.append(f'<example_len_{length}>')

        # the lexical complexity of the dictionary example
        for lexical_complexity_label in range(0, args.num_bins):
            special_tokens.append(f'<lexical_complexity_{lexical_complexity_label}>')

        special_tokens_dict = {
            'additional_special_tokens': special_tokens,
        }
   
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        if args.local_rank in [-1, 0]:
            print(f'We have added {num_added_toks} special tokens to the vocabulary: {tokenizer.get_added_vocab()}.')
            print(f"The original vocabulary size is {tokenizer.vocab_size}; "
                  f"the extended vocabulary size is {len(tokenizer)}.")

        # randomly initialize the newly added special tokens.
        # see https://huggingface.co/transformers/main_classes/model.html for details
        model.resize_token_embeddings(len(tokenizer))
    if args.local_rank == -1 or args.n_gpu <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device {device}.')
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(f'local rank: {args.local_rank}, device {device}.')
    model = model.to(device)

    if args.train == 1:
        training_set = BARTDataset(args.dataset, "training", tokenizer, args.use_word, args.use_definition, args.use_pos,
                                   args.use_example_len, args.use_lexical_complexity, args.max_example_len, args.num_bins,
                                   add_space=args.add_space, dataset_percent=args.dataset_percent)
        validation_set = BARTDataset(args.dataset, "validation", tokenizer, args.use_word, args.use_definition, args.use_pos,
                                     args.use_example_len, args.use_lexical_complexity, args.max_example_len, args.num_bins,
                                     add_space=args.add_space)
        if args.local_rank in [-1, 0]:
            logger.logger.info(f'The size of the training set is {len(training_set)}; '
                               f'the size of the validation set is {len(validation_set)}.')
        if args.local_rank == -1 or args.n_gpu <= 1:
            training_sampler = torch.utils.data.RandomSampler(training_set)
            validation_sampler = torch.utils.data.SequentialSampler(validation_set)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            training_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
            validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_set)
        training_dataloader = DataLoader(training_set, num_workers=0, batch_size=args.batch_size,
                                         sampler=training_sampler, collate_fn=training_set.create_mini_batch)
        validation_dataloader = DataLoader(validation_set, num_workers=0, batch_size=args.test_batch_size,
                                           sampler=validation_sampler, collate_fn=training_set.create_mini_batch)
    else:
        test_set = BARTDataset(args.dataset, "test", tokenizer, args.use_word, args.use_definition, args.use_pos,
                               args.use_example_len, args.use_lexical_complexity, args.max_example_len, args.num_bins,
                               add_space=args.add_space)
        if args.local_rank in [-1, 0]:
            logger.logger.info(f'The size of the test set is {len(test_set)}.')
        if args.local_rank == -1 or args.n_gpu <= 1:
            test_sampler = torch.utils.data.SequentialSampler(test_set)
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        test_dataloader = DataLoader(test_set, num_workers=0, batch_size=args.test_batch_size,
                                     sampler=test_sampler, collate_fn=test_set.create_mini_batch)
    if args.train == 0:
        average_loss, used_time = test_set.evaluate(model, args.local_rank, 'test', test_dataloader)
        if args.local_rank in [-1, 0]:
            logs = f'   Evaluate on the test set: average loss {average_loss:.3f}, ' \
                   f' taking {used_time:.1f} seconds.\n'
            logger.logger.info(logs)
    else:
        average_loss, used_time = validation_set.evaluate(model, args.local_rank, 'validation', validation_dataloader)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True,
                                                               min_lr=1e-6)
        scheduler.step(average_loss)
        best_loss = average_loss
        if args.local_rank in [-1, 0]:
            logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, ' \
                   f' taking {used_time:.1f} seconds.\n'
            logger.logger.info(logs)
        evaluate_steps = int(len(training_set) / args.batch_size / 5)
        print_steps = 10
        global_steps = 0
        local_step = 0
        total_loss = 0
        start = time.time()
        # fine-tune bart on the training dataset
        for epoch in range(args.epochs):
            for i, data in enumerate(training_dataloader):
                global_steps += 1
                local_step += 1
                data = [t.to(device) for t in data]
                encoder_inputs, attention_mask, decoder_inputs, decoder_labels, = data
                loss, logits = model(encoder_inputs, attention_mask=attention_mask,
                                     decoder_input_ids=decoder_inputs, labels=decoder_labels,
                                     output_attentions=True)[:2]
                # zero the parameter gradients
                optimizer.zero_grad()
                # backward
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if global_steps % print_steps == 0 and args.local_rank in [-1, 0]:
                    print("\rEpoch {}/{}, {}/{}, global steps {}, average loss is {:.3f}, "
                          " {} steps uses {:.1f} seconds.".format(epoch + 1, args.epochs, i + 1, len(training_dataloader),
                                                                  global_steps, total_loss / local_step,
                                                                  local_step, time.time() - start), end='')
                if global_steps % evaluate_steps == 0:
                    if args.local_rank in [-1, 0]:
                        print()
                    average_loss, used_time = validation_set.evaluate(model, args.local_rank, 'validation',
                                                                      validation_dataloader)
                    if args.local_rank in [-1, 0]:
                        logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, ' \
                               f' taking {used_time:.1f} seconds.'
                        logger.logger.info(logs)
                    if average_loss < best_loss:
                        best_loss = average_loss
                        if args.local_rank in [-1, 0]:
                            logger.logger.info('Save the model at {}.'.format(args.model_path))
                            # Simple serialization for models and tokenizers
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.model_path)
                            tokenizer.save_pretrained(args.model_path)

                    logger.logger.info('')
                    scheduler.step(average_loss)
                    start = time.time()
                    total_loss = 0
                    local_step = 0
