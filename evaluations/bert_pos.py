# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 2:08 PM
# @Author  : He Xingwei

"""
this script is used fine-tune BERT to get a model,
which is used to judge whether the pos of the specified word in the given example is the same with the given pos.
Input format: <CLS> word <SEP> pos  <SEP> example <SEP>
Output format: 0, 1
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import time
import os
import sys
import argparse
import numpy as np

sys.path.append('../')
from utils.log import Logger
from utils.functions import set_seed
from automatic_evaluation import load_gold_data
class BERTDataset(Dataset):
    """
    this dataset is for training/validation/testing with the cross entropy loss.
    """
    def __init__(self, dataset, mode, tokenizer):
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = tokenizer

        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []

        self.lemma_to_pos_dict = {}

        extracted_pos = ['Noun', 'Adjective', 'Verb', 'Adverb', 'Preposition',
                         'Interjection', 'Numeral', 'Pronoun', 'Determiner', 'Conjunction']
        num_labels = len(extracted_pos)
        data_dict_path = f'{dataset}/{mode}_bert_pos.pt'
        if os.path.exists(data_dict_path):
            print(f'Loading data from {data_dict_path}.')
            data_dict = torch.load(data_dict_path)
            self.input_ids_list = data_dict['input_ids_list']
            self.token_type_ids_list = data_dict['token_type_ids_list']
            self.attention_mask_list = data_dict['attention_mask_list']
            self.labels_list = data_dict['labels_list']
            self.num_positive = data_dict['num_positive']
            self.num_negative = data_dict['num_negative']
        else:
            if mode!='training':
            # load the training set to create the synthetic validation set.
                data_dict = torch.load(f'{dataset}/training_bert_pos.pt')
                self.lemma_to_pos_dict = data_dict['lemma_to_pos_dict']

            num_positive = 0
            num_negative = 0
            word_list = []
            lemma_list = []
            pos_list = []
            definition_list = []
            example_list = []
            print(f'Loading data from {dataset}/{mode}.txt.')
            with open(f'{dataset}/{mode}.txt', 'r') as fr, open(f'{dataset}/{mode}_pos_temp.txt', 'w') as fw:
                start = time.time()
                for line in fr:
                    line = line.strip()
                    word_seg, lemma_seg, pos_seg, definition_seg, example_seg = line.split('\t\t')
                    word = word_seg.split("word::: ")[1]
                    lemma = lemma_seg.split('lemma::: ')[1]
                    pos = pos_seg.split('pos::: ')[1]
                    definition = definition_seg.split('definition::: ')[1]
                    example = example_seg.split('example::: ')[1]
                    if self.lemma_to_pos_dict.get(lemma, -1)==-1:
                        self.lemma_to_pos_dict[lemma] = set([pos])
                    else:
                        self.lemma_to_pos_dict[lemma].add(pos)

                    word_list.append(word)
                    lemma_list.append(lemma)
                    pos_list.append(pos)
                    definition_list.append(definition)
                    example_list.append(example)

                num_examples = len(example_list)
                for i, (word, lemma, pos, definition, example) in enumerate(
                            zip(word_list, lemma_list, pos_list, definition_list, example_list)):
                    # create positive instances
                    sentence_a, sentence_b = word + ' [SEP] ' + f'<pos_{pos}>', example
                    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
                    self.input_ids_list.append(inputs['input_ids'][0])
                    self.token_type_ids_list.append(inputs['token_type_ids'][0])
                    self.attention_mask_list.append(inputs['attention_mask'][0])
                    self.labels_list.append(1)
                    num_positive += 1
                    gold_pos = pos
                    fw.write(f'{word}, {lemma}, {pos}, 1, {example}\n')

                    # create negative instances: select other pos of the word, and 2 other pos
                    sampled_pos = self.lemma_to_pos_dict.get(lemma, set())
                    remained_pos = set(extracted_pos)-sampled_pos
                    sampled_pos = list(sampled_pos)
                    remained_pos = list(remained_pos)
                    if len(remained_pos)>0:
                        sampled_num = min(2, len(remained_pos))
                        sampled_pos_indices = np.random.choice(len(remained_pos), sampled_num, replace=False)
                        for sampled_pos_index in sampled_pos_indices:
                            _pos = remained_pos[sampled_pos_index]
                            sampled_pos.append(_pos)

                    for pos in sampled_pos:
                        if pos == gold_pos:
                            continue
                        sentence_a, sentence_b = word + ' [SEP] ' + f'<pos_{pos}>', example
                        inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt',
                                                       add_special_tokens=True)
                        self.input_ids_list.append(inputs['input_ids'][0])
                        self.token_type_ids_list.append(inputs['token_type_ids'][0])
                        self.attention_mask_list.append(inputs['attention_mask'][0])
                        self.labels_list.append(0)
                        num_negative += 1
                        fw.write(f'{word}, {lemma}, {pos}, 0, {example}\n')

                    # create negative instances: select other examples, which doesn't contain the given word.
                    if i % 2 == 0:
                        while True:
                            # randomly select an example,  which doesn't contain the given word.
                            sampled_index = (i + np.random.randint(int(num_examples / 10),
                                                                   int(num_examples / 10 * 9))) % num_examples
                            example = example_list[sampled_index]
                            if word not in example:
                                break
                        sentence_a, sentence_b = word + ' [SEP] ' + f'<pos_{gold_pos}>', example
                        inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt',
                                                       add_special_tokens=True)
                        self.input_ids_list.append(inputs['input_ids'][0])
                        self.token_type_ids_list.append(inputs['token_type_ids'][0])
                        self.attention_mask_list.append(inputs['attention_mask'][0])
                        self.labels_list.append(0)
                        num_negative += 1
                        fw.write(f'{word}, {lemma}, {gold_pos}, 0, {example}\n')

                    if (i+1)%10000==0:
                        print(f'\r Loading data {i+1}, '
                              f'number of positive/negative instances {num_positive}/{num_negative}, '
                              f'used {time.time()-start:.1f} seconds.', end='')
            self.num_positive = num_positive
            self.num_negative = num_negative
            data_dict = {
                        'input_ids_list': self.input_ids_list,
                        'token_type_ids_list': self.token_type_ids_list,
                        'attention_mask_list': self.attention_mask_list,
                        'labels_list': self.labels_list,
                        'lemma_to_pos_dict': self.lemma_to_pos_dict,
                        'num_positive': self.num_positive,
                        'num_negative': self.num_negative
                         }

            torch.save(data_dict, data_dict_path)
        self.len = len(self.labels_list)
        print(f'\nThe number of positive instances is {self.num_positive}, negative instances is {self.num_negative}.')
    def __getitem__(self, idx):
        return self.input_ids_list[idx], self.token_type_ids_list[idx], self.attention_mask_list[idx], self.labels_list[idx]

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        input_ids_list = [s[0] for s in samples]
        token_type_ids_list = [s[1] for s in samples]
        attention_mask_list = [s[2] for s in samples]
        labels = torch.tensor([s[3] for s in samples], dtype=torch.long)
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value = 0)
        token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value = 0)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value = 0)

        return input_ids, token_type_ids, attention_mask, labels

    def evaluate(self, model, local_rank, dataloader, num_labels=2):
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
        step = 0
        start = time.time()
        correct = 0
        corrects = [0.0] *num_labels
        recalls = [0.0] *num_labels
        precisions = [0.0] *num_labels
        f1s = [0.0] *num_labels
        with torch.no_grad():
            for data in dataloader:
                data = [t.to(device) for t in data]
                input_ids, token_type_ids, attention_mask, labels  = data

                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids, labels=labels)[:2]
                values, predict_label = torch.max(logits, dim=1)
                correct += (predict_label == labels).sum()
                bts = input_ids.shape[0]
                total_loss += bts*loss
                step += bts

                for i in range(num_labels):
                    corrects[i] += ((predict_label == i) & ( labels == i)).sum()
                    recalls[i] += (labels == i).sum()
                    precisions[i] += (predict_label == i).sum()

                if local_rank in [-1, 0]:
                    print(
                        f'\r   Evaluating on the {self.mode} set for {step}/{datasize/torch.cuda.device_count()} '
                        f'takes {time.time()-start:.1f} seconds.', end='')

            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce_multigpu([total_loss])
                torch.distributed.all_reduce_multigpu([correct])

            # merge results
            for i in range(num_labels):
                if torch.cuda.device_count()>1:
                    torch.distributed.all_reduce_multigpu([corrects[i]])
                    torch.distributed.all_reduce_multigpu([recalls[i]])
                    torch.distributed.all_reduce_multigpu([precisions[i]])
                corrects[i] = corrects[i].item()
                recalls[i] = recalls[i].item()
                precisions[i] = precisions[i].item()

            for i in range(num_labels):
                if recalls[i]!=0:
                    recalls[i] = corrects[i]/recalls[i]
                else:
                    recalls[i] = 0

                if precisions[i]!=0:
                    precisions[i] = corrects[i]/precisions[i]
                else:
                    precisions[i] = 0

                if precisions[i]!=0:
                    f1s[i] = 2*recalls[i]*precisions[i]/(recalls[i]+precisions[i])
                else:
                    f1s[i] = 0

            total_loss = total_loss.item()
            average_loss = total_loss/datasize
            accuracy = correct*1.0/datasize
            used_time = time.time() - start
            print()
        model.train()
        return average_loss, accuracy, used_time, recalls, precisions, f1s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this script is used fine-tune BERT to get a model, which "
                                                 "is used to judge whether the specified word in the given example "
                                                 "conveys the corresponding meaning of the definition.")
    parser.add_argument('--dataset', type=str, default='oxford', help='the path of the dataset.')
    parser.add_argument('--initialization', type=str, default='bert-base-cased',
                        choices=['bert-base-cased', 'bert-large-cased'],
                        help='initialize the model with bert-base-cased or bert-large-cased.')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--test_batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate for training.')
    parser.add_argument('--train', type=int, default=1, choices=[0, 1], help='1 for training, 0 for testing.')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='1', help='The ids of gpus for training.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)

    prefix = f"lr_{args.lr}"
    model_path = f'../checkpoints/{args.dataset}_{args.initialization}_pos/{prefix}'
    log_path = f'../logs/{args.dataset}_{args.initialization}_pos'
    args.dataset = f'../data/{args.dataset}'
    if args.local_rank in [-1, 0]:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        args.model_path = model_path
        args.log_file = f'{log_path}/{prefix}.log'
        logger = Logger(args.log_file)
        logger.logger.info(f'The log file is {args.log_file}.')
        logger.logger.info(args)
        if args.train:
            logger.logger.info('Use {} gpus to train the model.'.format(args.n_gpu))
        else:
            logger.logger.info('Use {} gpus to evaluate the model.'.format(args.n_gpu))
    # we only extract word with the pos in extracted_pos
    extracted_pos = ['Noun', 'Adjective', 'Verb', 'Adverb', 'Preposition',
                     'Interjection', 'Numeral', 'Pronoun', 'Determiner', 'Conjunction']
    try:
        # load the pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        model = BertForSequenceClassification.from_pretrained(args.model_path)
        if args.local_rank in [-1, 0]:
            logger.logger.info('Initialize BertForSequenceClassification from checkpoint {}.'.format(args.model_path))
    except:
        tokenizer = BertTokenizer.from_pretrained(f'{args.initialization}')
        model = BertForSequenceClassification.from_pretrained(f'{args.initialization}', num_labels=2)
        if args.local_rank in [-1, 0]:
            logger.logger.info('Initialize BertForSequenceClassification with default parameters.')
        # add special tokens to the vocabulary
        special_tokens = []
        # the pos of word
        for pos in extracted_pos:
            special_tokens.append(f'<pos_{pos}>')

        special_tokens_dict = {
            'additional_special_tokens': special_tokens,
        }
        # print(tokenizer.vocab_size, len(tokenizer))
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        if args.local_rank in [-1, 0]:
            print(f'We have added {num_added_toks} special tokens to the vocabulary: {tokenizer.get_added_vocab()}.')
            print(f"The original vocabulary size is {tokenizer.vocab_size}; "
                  f"the extended vocabulary size is {len(tokenizer)}.")
        # randomly initialize the newly added special tokens.
        # see https://huggingface.co/transformers/main_classes/model.html for details
        model.resize_token_embeddings(len(tokenizer))
    if args.local_rank == -1 or args.n_gpu<=1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device {device}.')
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(f'local rank: {args.local_rank}, device {device}.')
    model = model.to(device)

    if args.train==1:
        training_set = BERTDataset(args.dataset, "training", tokenizer)
        validation_set = BERTDataset(args.dataset, "validation", tokenizer)
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
                                           sampler=validation_sampler, collate_fn=validation_set.create_mini_batch)

    else:
        test_set = BERTDataset(args.dataset, "validation", tokenizer)
        if args.local_rank in [-1, 0]:
            logger.logger.info(f'The size of the validation set is {len(test_set)}.')
        if args.local_rank == -1 or args.n_gpu <= 1:
            test_sampler = torch.utils.data.SequentialSampler(test_set)
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        test_dataloader = DataLoader(test_set, num_workers=0, batch_size=args.test_batch_size,
                                    sampler=test_sampler, collate_fn=test_set.create_mini_batch)
    if args.train==0:
        average_loss, accuracy, used_time, recalls, precisions, f1s\
            = test_set.evaluate(model, args.local_rank, test_dataloader)
        if args.local_rank in [-1, 0]:
            logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, accuracy {accuracy:.3f},' \
                   f' taking {used_time:.1f} seconds.'
            Macro_P = np.mean(precisions)
            Macro_R = np.mean(recalls)
            Macro_F1 = np.mean(f1s)

            for i in range(len(f1s)):
                logs += f'\n\t\t\t\tLabel_{i}: Precision={precisions[i]:.3f}, ' \
                        f'Recall={recalls[i]:.3f}, F1={f1s[i]:.3f};'
            logs += f'\n\t\t\t\tMacro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.\n'

            logger.logger.info(logs)
    else:
        average_loss, accuracy, used_time, recalls, precisions, f1s\
            = validation_set.evaluate(model, args.local_rank, validation_dataloader)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2, verbose=True,
                                                               min_lr=1e-6)
        best_loss = average_loss

        scheduler.step(accuracy)
        best_accuracy = accuracy
        if args.local_rank in [-1, 0]:
            logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, accuracy {accuracy:.3f},' \
                   f' taking {used_time:.1f} seconds.'
            Macro_P = np.mean(precisions)
            Macro_R = np.mean(recalls)
            Macro_F1 = np.mean(f1s)

            for i in range(len(f1s)):
                logs += f'\n\t\t\t\tLabel_{i}: Precision={precisions[i]:.3f}, ' \
                        f'Recall={recalls[i]:.3f}, F1={f1s[i]:.3f};'
            logs += f'\n\t\t\t\tMacro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.\n'

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
                input_ids, token_type_ids, attention_mask, labels  = data
                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids, labels=labels)[:2]
                # zero the parameter gradients
                optimizer.zero_grad()
                # backward
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if global_steps % print_steps == 0 and args.local_rank in [-1, 0]:
                    print(
                        "\rEpoch {}/{}, {}/{}, global steps {}, average loss is {:.3f}, "
                        " {} steps uses {:.1f} seconds.".format(epoch + 1, args.epochs, i + 1, len(training_dataloader),
                                                                global_steps, total_loss / local_step,
                                                                local_step, time.time() - start), end='')
                if global_steps % evaluate_steps == 0:
                    if args.local_rank in [-1, 0]:
                        print()
                    average_loss, accuracy, used_time, recalls, precisions, f1s\
                        = validation_set.evaluate(model, args.local_rank, validation_dataloader)

                    if average_loss < best_loss:
                        best_loss = average_loss
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        if args.local_rank in [-1, 0]:
                            logger.logger.info('Save the model at {}.'.format(args.model_path))
                            # Simple serialization for models and tokenizers
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.model_path)
                            tokenizer.save_pretrained(args.model_path)

                    if args.local_rank in [-1, 0]:
                        logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, ' \
                               f'accuracy {accuracy:.3f},' \
                               f' minimum average loss {best_loss:.3f}, best accuracy {best_accuracy:.3f}, ' \
                               f'taking {used_time:.1f} seconds.'
                        Macro_P = np.mean(precisions)
                        Macro_R = np.mean(recalls)
                        Macro_F1 = np.mean(f1s)

                        for i in range(len(f1s)):
                            logs += f'\n\t\t\t\tLabel_{i}: Precision={precisions[i]:.3f}, ' \
                                    f'Recall={recalls[i]:.3f}, F1={f1s[i]:.3f};'
                        logs += f'\n\t\t\t\tMacro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.\n'

                        logger.logger.info(logs)

                    scheduler.step(accuracy)
                    start = time.time()
                    total_loss = 0
                    local_step = 0



