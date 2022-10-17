# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 10:03 AM
# @Author  : He Xingwei

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import sys
import argparse
from transformers import BartTokenizer, BartForConditionalGeneration
sys.path.append('../')
from utils.log import Logger
from utils.functions import set_seed, convert_continues_to_discrete


class BARTDataset(Dataset):
    """
    this dataset is used to generate dictionary examples with the test/validation set without any loss.
    """
    def __init__(self, dataset, mode, tokenizer, use_word, use_definition, use_pos, use_example_len, use_lexical_complexity,
                 max_example_len, num_bins, expected_len =-1, expected_lexical_complexity = -1, add_space=0):
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
        :param expected_len: int
        :param expected_lexical_complexity: int
        :param add_space: int
        """
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_example_len = max_example_len

        self.word_list = []
        self.lemma_list = []
        self.definition_list = []
        self.example_list = []

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

        # load txt, which is used for the output
        filename = f'{dataset}/{mode}_features.txt'
        if add_space:
            data_dict_path = f'{dataset}/{mode}_features_add_space.pt'
        else:
            data_dict_path = f'{dataset}/{mode}_features.pt'
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

                self.word_list.append(word)
                self.lemma_list.append(lemma)
                self.definition_list.append(definition)
                self.example_list.append(example)

                if not os.path.exists(data_dict_path):
                    self.pos_list.append(pos)

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
                              ('external_token_rank_lexical_complexity_add_space::: ')[1] )
                    external_token_rank_lexical_complexity = \
                        float(external_token_rank_lexical_complexity_seg.split('external_token_rank_lexical_complexity::: ')[1])

                    flesch_reading_ease = float(flesch_reading_ease_seg.split('flesch_reading_ease::: ')[1])
                    flesch_kincaid_grade_level = float(flesch_kincaid_grade_level_seg.split('flesch_kincaid_grade_level::: ')[1])

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


                    if add_space:
                        word = ' ' + word
                        lemma = ' ' + lemma
                        definition = ' ' + definition
                        example = ' ' + example
                    # tokenize the word, lemma, definition, example
                    ids = tokenizer.encode(word, add_special_tokens=False)
                    self.word_tokenization_list.append(ids)

                    ids = tokenizer.encode(lemma, add_special_tokens=False)
                    self.lemma_tokenization_list.append(ids)

                    ids = tokenizer.encode(definition, add_special_tokens=True)
                    self.definition_tokenization_list.append(ids)

                    ids = tokenizer.encode(example, add_special_tokens=True)
                    self.example_tokenization_list.append(ids)

                    # the example length feature
                    self.example_tokenization_length_list.append(len(ids)-2)

            if not os.path.exists(data_dict_path):
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
            else:
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
                self.external_word_rank_lexical_complexity_list = data_dict[
                    'external_word_rank_lexical_complexity_list']
                self.external_token_rank_lexical_complexity_list = data_dict[
                    'external_token_rank_lexical_complexity_list']
                self.flesch_reading_ease_list = data_dict['flesch_reading_ease_list']
                self.flesch_kincaid_grade_level_list = data_dict['flesch_kincaid_grade_level_list']

        lexical_complexity_feature_dict = {
                                           1: (self.word_rank_lexical_complexity_list,'word_rank_lexical_complexity'),
                                           2: (self.token_rank_lexical_complexity_list, 'token_rank_lexical_complexity'),
                                           3: (self.external_word_rank_lexical_complexity_list,
                                               'external_word_rank_lexical_complexity'),
                                           4: (self.external_token_rank_lexical_complexity_list,
                                               'external_token_rank_lexical_complexity'),
                                           5: (self.flesch_reading_ease_list, 'flesch_reading_ease'),
                                           6: (self.flesch_kincaid_grade_level_list, 'flesch_kincaid_grade_level')
                                           }
        if add_space:
            _filename = f"{dataset}/training_features_interval_add_space.txt"
        else:
            _filename = f"{dataset}/training_features_interval.txt"
        if os.path.exists(_filename):
            with open(_filename, 'r') as fr:
                for i, line in enumerate(fr):
                    if i==0:
                        example_feature_name, min_example_len, max_example_len = line.strip().split('\t\t')
                        min_example_len = int(min_example_len)
                        max_example_len = int(max_example_len)
                    if i>0 and i == use_lexical_complexity:
                        feature_name, min_feature_value, max_feature_value = line.strip().split('\t\t')
                        min_feature_value = float(min_feature_value)
                        max_feature_value = float(max_feature_value)
                        assert feature_name == lexical_complexity_feature_dict[use_lexical_complexity][1]
        else:
            if mode != 'training':
                print('Please create the training_features_interval.txt first with the training set.')
                exit()
            else:
                print(f'Create {_filename}.')
                with open(_filename,'w') as fw:
                    max_example_len = np.max(self.example_tokenization_length_list)
                    min_example_len = np.min(self.example_tokenization_length_list)
                    fw.write(f'example_length\t\t{min_example_len}\t\t{max_example_len}\n')
                    for k in range(1,7):
                        lexical_complexity_feature_list, feature_name = lexical_complexity_feature_dict[k]
                        max_num = np.max(lexical_complexity_feature_list)
                        min_num = np.min(lexical_complexity_feature_list)
                        fw.write(f'{feature_name}\t\t{min_num}\t\t{max_num}\n')
                        if k == use_lexical_complexity:
                            min_feature_value = min_num
                            max_feature_value = max_num

        if use_example_len and expected_len ==-1:
            print(f'The min and max {example_feature_name} values are {min_example_len}, {max_example_len}.')
        if use_lexical_complexity>0 and expected_lexical_complexity==-1:
            # discretize the lexical complexity features
            start_time = time.time()
            lexical_complexity_feature_list, feature_name = lexical_complexity_feature_dict[use_lexical_complexity]
            lexical_complexity_feature_list, min_num, max_num, start, end, bin_width \
                = convert_continues_to_discrete( lexical_complexity_feature_list,
                                                 start = min_feature_value,
                                                 end = max_feature_value,
                                                 num_bins=num_bins)

            print(f'Discretizing the lexical complexity feature, {feature_name}, uses {time.time()-start_time:.1f} seconds.')
            print(f'Unique labels: {np.unique(lexical_complexity_feature_list)}.')
            print(f'The min number is {min_num:.3f}, max number is {max_num:.3f}, '
                  f'the start of the bins is {start:.3f}, the end of the bins is {end:.3f}, and the bin width is {bin_width:.3f}.')



        # create encoder inputs

        # Note: Different form BARTDataset in bart.py, we merge the instances with the same (lemma, pos, definition) as one evaluation instances.
        # In addition, the corresponding examples are regarded as the references for this evaluation instance,
        # when evaluating the generated sentences with BLEU, NIST, and METEOR.
        self.encoder_input_list = []
        previous_definition_tuple = []
        self.validation_index = []
        self.examples_index = []

        for i, (word, lemma, pos, definition) in enumerate(zip(self.word_tokenization_list, self.lemma_tokenization_list,
                                                self.pos_list, self.definition_tokenization_list)):
            current_definition_tuple = lemma + [pos] + definition
            if current_definition_tuple == previous_definition_tuple:
                self.examples_index[-1].append(i)
                continue
            else:
                previous_definition_tuple = current_definition_tuple
                self.validation_index.append(i)
                self.examples_index.append([i])

            encoder_input = []
            if use_word==0:
                encoder_input += [special_token_dict['<word>']] + word
            elif use_word==1:
                encoder_input += [special_token_dict['<lemma>']] + lemma
            elif use_word == 2:
                encoder_input += [special_token_dict['<word>']] + word + [special_token_dict['<lemma>']] + lemma
            else:
                pass

            if use_pos:
                encoder_input += [special_token_dict['<pos>'], special_token_dict[f'<pos_{pos}>'] ]

            if use_example_len and expected_len>=-1: # -2 denotes not using it
                if expected_len!=-1: # expected
                    length = expected_len
                elif expected_len==-1: # gold
                    length = self.example_tokenization_length_list[i]
                encoder_input += [special_token_dict['<example_len>'],
                                  special_token_dict.get(f'<example_len_{length}>',
                                                         special_token_dict[f'<example_len_{self.max_example_len}>']
                                                         )
                                  ]

            if use_lexical_complexity > 0 and expected_lexical_complexity>=-1: # -2 denotes not using it
                if expected_lexical_complexity !=-1:
                    lexical_complexity_label = expected_lexical_complexity
                else:
                    lexical_complexity_label = lexical_complexity_feature_list[i]
                encoder_input += [special_token_dict['<lexical_complexity>'],
                                  special_token_dict[f'<lexical_complexity_{lexical_complexity_label}>']]
            if use_definition:
                encoder_input += [special_token_dict['<definition>']] + definition
            self.encoder_input_list.append(encoder_input)

        self.len = len(self.encoder_input_list)

    def __getitem__(self, idx):
        return [torch.tensor(self.encoder_input_list[idx], dtype=torch.long),]

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_input_list = [s[0] for s in samples]
        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_input_list, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)
        encoder_inputs = pad_sequence(encoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return encoder_inputs, attention_mask

def contain(words, example):
    example = ' ' + example + ' '
    for i, word in enumerate(words):
        start = 0
        while True:
            start = example.find(word, start)
            if start!=-1:
                start_char = example[start-1]
                end_char = example[start+len(word)]
                if not start_char.isalnum() and not end_char.isalnum():
                    if i == 0:
                        return 1
                    return 2
                start += 1
            else:
                break
    return 0

def generate(word_list, input_ids = None, attention_mask = None, max_decoding_len=None, num_beams=1, repetition_penalty=1,
             top_k=50, top_p=0.9, decoder_chain = 1):
    """

    :param input_ids:
    :param attention_mask:
    :param max_decoding_len:
    :param num_beams:
    :param repetition_penalty:
    :param top_k:
    :param top_p:
    :param decoder_chain: run multiple parallel chains for top-k or top-p sampling, then choose the one contains the given word
    :return:
    """
    batch_size = input_ids.shape[0]
    if decoder_chain>1:
        input_ids = input_ids.repeat(decoder_chain,1,1).reshape(batch_size*decoder_chain, -1)
        attention_mask = attention_mask.repeat(decoder_chain,1,1).reshape(batch_size*decoder_chain, -1)

    # generate text until the output length (which includes the context length) reaches 50
    if args.decoding_strategy == 1:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_decoding_len,
                                num_beams=num_beams, repetition_penalty=repetition_penalty, output_attentions = True)
    elif args.decoding_strategy == 2:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_decoding_len,
                                do_sample=True, top_k=top_k, repetition_penalty=repetition_penalty, output_attentions = True)
    elif args.decoding_strategy == 3:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_decoding_len,
                                do_sample=True, top_p=top_p, top_k=0,
                                repetition_penalty=repetition_penalty, output_attentions = True)

    generated_example_list = []
    selected_example_list = []

    for i in range(batch_size*decoder_chain):
        generated_example = tokenizer.decode(output[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_example_list.append(generated_example)
    # for e in generated_example_list:
    #     print(e)
    if decoder_chain>1:
        for i in range(batch_size):
            examples = [generated_example_list[j*batch_size+i]   for j in range(decoder_chain)]
            selected_example_list.append(examples[0])
            for example in examples:
                contain_code = contain([word_list[i]], example)
                if contain_code>0:
                    selected_example_list[i] = example
                    break
    else:
        selected_example_list = generated_example_list
    return selected_example_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
    # hyper-parameters for well-trained models
    parser.add_argument('--dataset', type=str, default='oxford', help='the path of the dataset.')
    parser.add_argument('--dataset_percent', type=float, default=1,
                        help='The percentage of data used to train the model.')
    parser.add_argument('--initialization', type=str, default='bart-base',
                        choices=['bart-random-base', 'bart-base', 'bart-large'],
                        help='initialize the model with random values, bart-base or bart-large.')
    parser.add_argument('--test_mode', type=int, default=1, choices=[0, 1, 2], help='0 for validation, 1 for test set, '
                                                                               '2 for specified inputs')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate for training.')
    parser.add_argument('--gpu', type=str, default='3', help='The ids of gpus for training.')

    # hyper-parameters for features
    parser.add_argument('--add_space', type=int, default=1, choices=[0,1],
                        help='Whether add a space before the example, word and lemma '
                             'so that the tokens of the word do appear in the token sequence of the example.')
    parser.add_argument('--use_word', type=int, default=0, choices=[0,1,2,3],
                        help='whether use the word or lemma as the input of the encoder.'
                             '0 for word; 1 for lemma; 2 for both word and lemma; 3 for not using both.')
    parser.add_argument('--use_definition', type=int, default=1, choices=[0, 1],
                        help='whether use the pos as the input of the encoder.')
    parser.add_argument('--use_pos', type=int, default=0, choices=[0,1],
                        help='whether use the pos as the input of the encoder.')
    parser.add_argument('--use_example_len', type=int, default=0, choices=[0,1],
                        help='whether use the length of examples as the input of the encoder.')
    parser.add_argument('--use_lexical_complexity', type=int, default=0, choices=[0,1,2,3,4,5,6],
                        help='0 denotes the lexical complexity is not used as the input of the encoder;'
                             '1 denotes the word_rank_lexical_complexity is regarded as the lexical complexity;'
                             '2 denotes the token_rank_lexical_complexity is regarded as the lexical complexity;'
                             '3 denotes the external_word_rank_lexical_complexity is regarded as the lexical complexity;'
                             '4 denotes the external_token_rank_lexical_complexity is regarded as the lexical complexity;'
                             '5 denotes the flesch_reading_ease is regarded as the lexical complexity;'
                             '6 denotes the flesch_kincaid_grade_level is regarded as the lexical complexity;')
    parser.add_argument('--num_bins', type=int, default=40,
                        help='the number of bins for lexical complexity features.')
    parser.add_argument('--max_example_len', type=int, default= 60,
                        help='the max length of the dictionary examples.')

    # hyper-parameters for decoding strategy
    parser.add_argument('--decoding_strategy', type=int, default=1, choices=[1,2,3],
                        help='1 for greedy/beam search decoding; 2 for top-k decoding; 3 for top-p decoding')
    parser.add_argument('--num_beams', type=int, default=1,
                        help='1 for greedy decoding; '
                             'the number greater than 1 denotes beam search decoding.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering. '
                             'Between 1 and infinity.')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='The cumulative probability of parameter highest probability vocabulary tokens to keep '
                             'for nucleus sampling. Must be between 0 and 1.')
    parser.add_argument('--decoder_chain', type=int, default=1,
                        help='the number of parallel chains for top-k or top-p, each chain refers to an unique token sequence.')
    parser.add_argument('--repetition_penalty', type=float, default=1.3,
                        help='Between 1.0 and infinity.1.0 means no penalty.Default to 1.0.')

    parser.add_argument('--max_decoding_len', type=int, default= 60,
                        help='the max length of the dictionary examples.')

    parser.add_argument('--expected_len', type=int, default= -2,
                        help='Specify the expected length of generated examples.'
                             '-2 denotes not using this token.'
                             '-1 denotes use the gold label of the validation/test set.'
                             'the value should be integer in [0, num_bins).')

    parser.add_argument('--expected_lexical_complexity', type=int, default= -2,
                        help='Specify the expected lexical complexity of generated examples.'
                             '-2 denotes not using this token.'
                             '-1 denotes use the gold label of the validation/test set.'
                             'the value should be integer in [1, max_example_len].')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)

    if args.test_mode == 0:
        print('Evaluate the model on the validation set.')
        mode = 'validation'
    elif args.test_mode == 1:
        print('Evaluate the model on the test set.')
        mode = 'test'
    else:
        print('Evaluate the model on specified inputs.')
        mode = 'specified'
    if args.decoding_strategy ==1:
        args.decoder_chain=1

    prefix = f'lr_{args.lr}'

    if args.add_space:
        prefix += f"_add_space"

    if args.dataset_percent<1:
        prefix += f'_data_percent_{args.dataset_percent}'

    if args.use_word==0:
        prefix += '_use_word'
    elif args.use_word==1:
        prefix += '_use_lemma'
    elif args.use_word==2:
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

    if args.use_example_len and args.expected_len >0:
        # specify the expected len. otherwise, use the gold len
        assert args.expected_len >= 1
        assert args.expected_len <= args.max_example_len
        prefix += f'_expected_len_{args.expected_len}'

    # specify the expected lexical complexity. otherwise, use the gold label
    if args.use_lexical_complexity > 0 and args.expected_lexical_complexity >=0:
        assert args.expected_lexical_complexity>=0
        assert args.expected_lexical_complexity<args.num_bins
        prefix += f'_expected_lexical_complexity_{args.expected_lexical_complexity}'

    if args.use_example_len and args.expected_len == -2 and \
            args.use_lexical_complexity > 0 and args.expected_lexical_complexity == -2:
        # add the prefix to differ from the case when expected_len =-1 and expected_lexical_complexity=-1 (using gold label)
        prefix += f'_not_use_expected_len_expected_lexical_complexity'

    prefix += f'_max_decoding_len_{args.max_decoding_len}'
    if args.decoding_strategy==1:
        if args.num_beams==1:
            prefix += f'_greedy'
        else:
            prefix += f'_beam_search_{args.num_beams}'
    elif args.decoding_strategy==2:
        prefix += f'_top_k_{args.top_k}_{args.decoder_chain}'
    elif args.decoding_strategy==3:
        prefix += f'_top_p_{args.top_p}_{args.decoder_chain}'
    else:
        raise ValueError('Please input the correct decoding strategy index (1, 2, or 3).')
    if args.repetition_penalty>1:
        prefix += f'_repetition_penalty_{args.repetition_penalty}'

    log_path = f'../logs/{args.dataset}_{args.initialization}_generate'
    output_path = f'../outputs/{args.dataset}_{args.initialization}_generate'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    args.model_path = model_path

    args.output_file = f'{output_path}/{mode}_{prefix}.txt'
    args.log_file = f'{log_path}/{mode}_{prefix}.log'

    logger = Logger(args.log_file)
    logger.logger.info(f'The log file is {args.log_file}.')
    logger.logger.info(f'The output file is {args.output_file}.')
    logger.logger.info(args)
    # load the pre-trained model and tokenizer
    logger.logger.info(f'Loading the model from checkpoint {args.model_path}.')
    args.dataset = f'../data/{args.dataset}'

    tokenizer = BartTokenizer.from_pretrained(args.model_path)
    model = BartForConditionalGeneration.from_pretrained(args.model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device {device}.')
    model = model.to(device)
    model.eval()

    special_token_dict = tokenizer.get_added_vocab()
    if args.test_mode == 2:
        with torch.no_grad():
            word = 'sentence'
            pos = 'Verb'
            definition = "declare the punishment decided for (an offender)"
            if args.add_space:
                word = ' ' + word
                definition = ' ' + definition

            word_ids = tokenizer.encode(word, add_special_tokens=False)
            definition_ids = tokenizer.encode(definition, add_special_tokens=True)

            input_ids = [special_token_dict['<word>']] + word_ids + \
                        [special_token_dict['<pos>'], special_token_dict[f'<pos_{pos}>']] \
                        + [special_token_dict['<definition>']] + definition_ids

            word = 'putten'
            pos = 'Noun'
            definition = "an outstandingly successful person or thing in a group"
            if args.add_space:
                word = ' ' + word
                definition = ' ' + definition

            word_ids = tokenizer.encode(word, add_special_tokens=False)
            definition_ids = tokenizer.encode(definition, add_special_tokens=True)

            input_ids2 = [special_token_dict['<word>']] + word_ids + \
                        [special_token_dict['<pos>'], special_token_dict[f'<pos_{pos}>']] \
                        + [special_token_dict['<definition>']] + definition_ids

            input_ids_list = [input_ids, input_ids2]
            input_ids_list = [torch.tensor(e) for e in input_ids_list]
            batch = len(input_ids_list)

            # Mask to avoid performing attention on padding token indices in encoder_inputs.
            _mask = pad_sequence(input_ids_list, batch_first=True, padding_value=-100)
            attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
            attention_mask = attention_mask.masked_fill(_mask != -100, 1)
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            generated_examples = generate(['sentence','putten'], input_ids=input_ids, attention_mask=attention_mask,
                                          max_decoding_len=args.max_decoding_len, num_beams=args.num_beams,
                                          repetition_penalty=args.repetition_penalty,
                                          top_k=args.top_k, top_p=args.top_p, decoder_chain=args.decoder_chain)

            print("Output:\n" + 100 * '-')
            for i in range(batch):
                print(generated_examples[i])
    else:
        test_set = BARTDataset(args.dataset, mode, tokenizer, args.use_word, args.use_definition, args.use_pos,
                               args.use_example_len, args.use_lexical_complexity, args.max_example_len, args.num_bins,
                               add_space=args.add_space, expected_len=args.expected_len,
                               expected_lexical_complexity=args.expected_lexical_complexity)
        logger.logger.info(f'The size of the {mode} set is {len(test_set)}.')
        test_sampler = torch.utils.data.SequentialSampler(test_set)
        test_dataloader = DataLoader(test_set, num_workers=0, batch_size=args.batch_size,
                                    sampler=test_sampler, collate_fn=test_set.create_mini_batch)


        with torch.no_grad(), open(args.output_file, 'w') as fw:
            start = time.time()
            batch_index = -1
            for data in test_dataloader:
                batch_index += 1
                data = [t.to(device) for t in data]
                input_ids, attention_mask = data

                batch_size = input_ids.shape[0]
                word_list = []
                for i in range(args.batch_size*batch_index, args.batch_size*batch_index+batch_size):
                    index = test_set.validation_index[i]
                    word = test_set.word_list[index]
                    word_list.append(word)

                generated_examples = generate(word_list, input_ids=input_ids, attention_mask=attention_mask,
                                              max_decoding_len=args.max_decoding_len, num_beams=args.num_beams,
                                              repetition_penalty=args.repetition_penalty,
                                              top_k=args.top_k, top_p=args.top_p, decoder_chain=args.decoder_chain)

                # write the output into the output file
                for i, j in zip(range(args.batch_size*batch_index, args.batch_size*batch_index+batch_size), range(batch_size)):
                    index = test_set.validation_index[i]
                    lemma = test_set.lemma_list[index]
                    word =test_set.word_list[index]
                    pos = test_set.pos_list[index]
                    definition = test_set.definition_list[index]


                    # the example length feature
                    example_tokenization_length = test_set.example_tokenization_length_list[index]
                    # lexical complexity features
                    word_rank_lexical_complexity = test_set.word_rank_lexical_complexity_list[index]
                    token_rank_lexical_complexity = test_set.token_rank_lexical_complexity_list[index]
                    external_word_rank_lexical_complexity = test_set.external_word_rank_lexical_complexity_list[index]
                    external_token_rank_lexical_complexity = test_set.external_token_rank_lexical_complexity_list[index]
                    flesch_reading_ease = test_set.flesch_reading_ease_list[index]
                    flesch_kincaid_grade_level = test_set.flesch_kincaid_grade_level_list[index]

                    examples = []
                    for index in test_set.examples_index[i]:
                        examples.append(test_set.example_list[index])
                    examples = "\t\t".join(examples)
                    generated_example = generated_examples[j]
                    if generated_example[0] == ' ':  # remove the space of the begin
                        generated_example = generated_example[1:]

                    fw.write(f"word::: {word}\t\tlemma::: {lemma}\t\tpos::: {pos}\t\t"
                             f"definition::: {definition}\t\texample_tokenization_length::: {example_tokenization_length}\t\t"
                             f"word_rank_lexical_complexity::: {word_rank_lexical_complexity}\t\t"
                             f"token_rank_lexical_complexity::: {token_rank_lexical_complexity}\t\t"
                             f"external_word_rank_lexical_complexity::: {external_word_rank_lexical_complexity}\t\t"
                             f"external_token_rank_lexical_complexity::: {external_token_rank_lexical_complexity}\t\t"
                             f"flesch_reading_ease::: {flesch_reading_ease}\t\t"
                             f"flesch_kincaid_grade_level::: {flesch_kincaid_grade_level}\n")

                    fw.write(f"Reference examples::: {examples}\n")
                    fw.write(f"The generated example::: {generated_example}\n")
                # break
                print(f'\rProcess {args.batch_size*(batch_index+1)}/{len(test_set)}, used {time.time()-start:.1f} seconds.', end='')

            logger.logger.info(f'The inference latency is {time.time()-start:.2f}\n')


