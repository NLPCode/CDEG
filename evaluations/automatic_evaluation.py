# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 12:08 PM
# @Author  : He Xingwei

"""
this script is used to automatically evaluate the generated text.
We use NLL for the sentence quality.
We use BLEU, NIST, METEOR to measure the similarity between human-written text and compute-generated sentence.
We use unique n-gram and SELF-BLEU to measure the sentence diversity.
We also measure the n-gram repetitions.
"""
import  numpy as np
import glob,argparse
from nltk.translate.nist_score import corpus_nist
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
import os, time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer
from nlgeval import NLGEval

class Evaluation:
    def __init__(self, model=None, tokenizer=None, bert_definition_model = None, bert_definition_tokenizer = None,
                 bert_pos_model=None, bert_pos_tokenizer = None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_definition_model = bert_definition_model
        self.bert_definition_tokenizer = bert_definition_tokenizer
        self.bert_pos_model = bert_pos_model
        self.bert_pos_tokenizer = bert_pos_tokenizer

    @staticmethod
    def tokenize(sentences_list):
        """
        this is used to tokenize the given sentence
        :param sentences_list: list of str
        :return: list of list
        """
        tokenized_sentences_list = []
        for sentence in sentences_list:
            if type(sentence) is list:
                tokens = []
                for s in sentence:
                    tokens.append(word_tokenize(s.strip()))
                    # tokens.append(s.strip().split())
            else:
                tokens = word_tokenize(sentence.strip())
                # tokens = sentence.strip().split()
            tokenized_sentences_list.append(tokens)
        return tokenized_sentences_list

    @staticmethod
    def n_gram_repetition(tokens, n_grams=[1,2,3,4], thresholds=[3,3,2,2], maxlen=20):
        """
        this function is used to judge whether the given sentence contains n-gram repetitions.
        If the a n-gram appears at least t times, them the sentence contains n-gram repetitions.
        :param tokens: list of token
        :param n_grams:
        :param thresholds:
        :param maxlen:
        :return:
        """
        is_repetition = []
        for n,t in zip(n_grams, thresholds):
            ngram_fdist = nltk.FreqDist()
            tokens = tokens[:maxlen]
            ngrams = nltk.ngrams(tokens, n)
            ngram_fdist.update(ngrams)
            flag = 0
            for k, v in ngram_fdist.items():
                if v>=t:
                    flag=1
                    break
            is_repetition.append(flag)
        return is_repetition

    @staticmethod
    def repetition( tokenized_sentences_list, n_grams=[1,2,3,4], thresholds=[3,3,2,2]):
        """
        this function is used to check whether each sentence in the list contains n-gram repetitions.
        :param tokenized_sentences_list:
        :param n_grams:
        :param thresholds:
        :return:
        """
        repetition_array=None
        for tokenized_sentence in tokenized_sentences_list:
            is_repetition = Evaluation.n_gram_repetition(tokenized_sentence, n_grams, thresholds)
            if any(is_repetition):
                is_repetition.append(1)
            else:
                is_repetition.append(0)
            is_repetition = np.array(is_repetition)
            if repetition_array is None:
                repetition_array = is_repetition
            else:
                repetition_array += is_repetition
        return repetition_array/len(tokenized_sentences_list)

    @staticmethod
    def distances_and_entropy(tokenized_sentences_list, ngrams=[1, 2, 3, 4], num_tokens=0):
        """
        this function is used to calculate the percentage of unique n-grams to measure the generation diversity.
        this function is also used to calculate entropy to measure the generation diversity.
        :param tokenized_sentences_list:
        :param ngrams:
        :param num_tokens:
        :return:
        """
        distances = []
        entropies = []
        if num_tokens > 0:
            cur_num = 0
            new_tokenized_sentences_list = []
            for tokenized_sentence in tokenized_sentences_list:
                cur_num += len(tokenized_sentence)
                new_tokenized_sentences_list.append(tokenized_sentence)
                if cur_num >= num_tokens:
                    break
            tokenized_sentences_list = new_tokenized_sentences_list

        for n in ngrams:
            # calculate (n-gram, frequency) pairs
            ngram_fdist = nltk.FreqDist()
            for tokens in tokenized_sentences_list:
                ngrams = nltk.ngrams(tokens, n)
                ngram_fdist.update(ngrams)
            unique = ngram_fdist.B()  # the number of unique ngrams
            total = ngram_fdist.N()  # the number of ngrams
            distances.append(unique * 1.0 / total)
            # calculate entropies
            ans = 0.0
            for k, v in ngram_fdist.items():
                ans += v * np.log(v * 1.0 / total)
            ans = -ans / total
            entropies.append(ans)
        return distances, entropies

    @staticmethod
    def compute_self_bleu(tokenized_sentences_list, weights=[0.25, 0.25, 0.25, 0.25],
                          number_sentences = 1000, num_tokens=0):
        """
        compute the similarity between generated sentences by regarding one as hypothesis and the other as references.
        :param tokenized_sentences_list:
        :param weights: weight for uni-gram, bi-gram, tri-gram, 4-gram
        :param num_tokens: if the number less than 0, we will use all other sentences as the references.
        Otherwise, we select num_tokens as the references. default -1
        :return:
        """
        hypothesis_list = []
        for tokenized_sentence in tokenized_sentences_list:
            hypothesis_list.append(tokenized_sentence)
        reference_list_list = []
        number = len(tokenized_sentences_list)
        for i in range(number):
            reference_list = []
            cur_num = 0
            cur_sen = 0
            for j in range(number):
                if j != i:
                    if num_tokens > 0:
                        if cur_num >= num_tokens:
                            break
                        cur_num += len(hypothesis_list[j])
                    if number_sentences>0:
                        if  cur_sen >= number_sentences:
                            break
                        cur_sen +=1
                    reference_list.append(hypothesis_list[j])
            reference_list_list.append(reference_list)
        BLEUscore = corpus_bleu(reference_list_list, hypothesis_list, weights=weights)
        return BLEUscore

    @staticmethod
    def compute_bleu(generated_tokenized_sentences_list, gold_tokenized_sentences_list, weights=[0.25, 0.25, 0.25, 0.25]):
        """
        This is used compute the corpus BLEU score between the generated sentence and human references.
        :param generated_tokenized_sentences_list:
        :param gold_tokenized_sentences_list:
        :param weights:
        :return:
        """
        hypothesis_list = generated_tokenized_sentences_list
        reference_list_list = gold_tokenized_sentences_list
        BLEUscore  = corpus_bleu(reference_list_list, hypothesis_list, weights=weights)
        return BLEUscore

    @staticmethod
    def compute_nist( generated_tokenized_sentences_list, gold_tokenized_sentences_list, ngrams=2):
        hypothesis_list = generated_tokenized_sentences_list
        reference_list_list = gold_tokenized_sentences_list
        nist = corpus_nist(reference_list_list, hypothesis_list, n=ngrams)
        return nist

    def compute_perplexity(self, input_ids_list=None, sentences_list=None):
        """
        compute the probabilities with the model for the input sentences.
        :param input_ids:  2 dimensional list, each element is a list for a sentence.
        :param sentences_list: list of sentences.
        :return: log_ppl: a list of log-perplexity value.
        """
        if sentences_list:
            input_ids_list = []
            for sentence in sentences_list:
                input_ids_list.append(self.tokenizer.encode(sentence, add_special_tokens=False))

        with torch.no_grad():
            # tokenizer.bos_token_id = tokenizer.eos_token_id = 50256
            input_tensors = []
            lengths = []
            for input_ids in input_ids_list:
                input_ids = [self.tokenizer.bos_token_id] + input_ids[:] + [self.tokenizer.eos_token_id]
                # print(self.forward_lm_tokenizer.convert_ids_to_tokens(input_ids))
                lengths.append(len(input_ids) - 1)
                input_tensors.append(torch.tensor(input_ids))

            # pad label with -100 (can not be other number.)
            labels_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=-100)
            labels_tensors = labels_tensors.to(self.device)
            labels_tensors = labels_tensors[:, 1:]

            # pad input with 0 (the padded value can be arbitrary number.)
            input_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=0)
            input_tensors = input_tensors.to(self.device)
            input_tensors = input_tensors[:, :-1]
            lengths_tensors = torch.tensor(lengths).to(self.device).double()

            outputs = self.model(input_tensors)
            logits = outputs[0]

            # compute sentence probs and log_ppls
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss_ = loss_func(logits.reshape(-1, logits.shape[-1]), labels_tensors.reshape(-1))
            loss_ = loss_.reshape(labels_tensors.shape)
            loss_ = torch.sum(loss_, dim=-1).double()
            log_ppls = (loss_ / lengths_tensors).cpu().numpy()
            # probs = torch.exp(-loss_).cpu().numpy()
        return log_ppls, np.array(lengths)
    #
    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
        )
        return out_string

    def compute_average_perplexity(self, sentences_list, batch_size=50):
        """
        compute the average log-perplexity of the given sentences.
        :param model:
        :param tokenizer:
        :param sentences_list:
        :param batch_size:
        :return:
        """
        log_perplexity = 0
        total_len = 0
        number = len(sentences_list)
        for i in range(0, number, batch_size):
            sentences = sentences_list[i:i+batch_size]
            for i, sentence in enumerate(sentences):
                sentence = self.clean_up_tokenization(sentence)
                sentences[i] = sentence
            log_ppl, length = self.compute_perplexity( sentences_list=sentences)
            log_perplexity += np.sum(log_ppl * length)
            total_len += np.sum(length)
        return log_perplexity / total_len, total_len * 1.0 / number

    def evaluate_word_definition(self, sentences_list, batch_size = 50, word_coverage_list=None):
        """
         judge whether the meaning of the specified word in the given example conveys the corresponding meaning of the definition.
        :param sentences_list: (word, lemma, pos, definition, example) of list
        :return: accuracy
        """
        num_same_meaning = 0.0
        number = len(sentences_list)
        for i in range(0, number, batch_size):
            sentences = sentences_list[i:i+batch_size]
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_list = []
            for j, (word, lemma, pos, definition, example) in enumerate(sentences):
                sentence_a, sentence_b = word + ' [SEP] ' + definition, example
                inputs = self.bert_definition_tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
                input_ids_list.append(inputs['input_ids'][0])
                token_type_ids_list.append(inputs['token_type_ids'][0])
                attention_mask_list.append(inputs['attention_mask'][0])
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(self.device)
            token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0).to(self.device)
            attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(self.device)
            with torch.no_grad():
                logits = self.bert_definition_model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)[0]
            values, indices = torch.max(logits, dim=1)
            if word_coverage_list is not None:
                indices *= torch.tensor(word_coverage_list[i:i+batch_size], device=self.device)
            num_same_meaning += indices.sum()
        num_same_meaning = num_same_meaning.item()
        return num_same_meaning/number, num_same_meaning, number


    def evaluate_word_pos(self, sentences_list, batch_size = 50, word_coverage_list=None):
        """
        judge whether the pos of the specified word in the given example is the same with the given pos.
        :param sentences_list: (word, lemma, pos, definition, example) of list
        :return: accuracy
        """
        num_same_pos = 0.0
        number = len(sentences_list)
        for i in range(0, number, batch_size):
            sentences = sentences_list[i:i+batch_size]
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_list = []
            for j, (word, lemma, pos, definition, example) in enumerate(sentences):
                sentence_a, sentence_b = word + ' [SEP] ' + f'<pos_{pos}>', example
                inputs = self.bert_pos_tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
                input_ids_list.append(inputs['input_ids'][0])
                token_type_ids_list.append(inputs['token_type_ids'][0])
                attention_mask_list.append(inputs['attention_mask'][0])
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(self.device)
            token_type_ids = pad_sequence(token_type_ids_list, batch_first=True, padding_value=0).to(self.device)
            attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(self.device)
            with torch.no_grad():
                logits = self.bert_pos_model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)[0]
            values, indices = torch.max(logits, dim=1)
            if word_coverage_list is not None:
                indices *= torch.tensor(word_coverage_list[i:i+batch_size], device=self.device)
            num_same_pos += indices.sum()
        num_same_pos = num_same_pos.item()
        return num_same_pos/number, num_same_pos, number

    def word_coverage(self, word_list, example_list, lemma_list, pos_list):
        """
        this function aims to check whether the word appears in the example.
        :param word_list:
        :param example_list:
        :return:
        """
        word_coverage_list = []
        def _contain(words, example):
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

        total = len(word_list)
        num_contained= 0.0
        num_contained2 = 0.0
        for words, example, lemma, pos in zip(word_list, example_list, lemma_list, pos_list):
            example = ' ' + example + ' '
            contain_code = _contain(words, example)
            new_l = []
            if contain_code ==0:
                word_coverage_list.append(0)
            elif contain_code==1:
                num_contained+=1
                num_contained2 += 1
                word_coverage_list.append(1)
            else:
                num_contained2+=1
                word_coverage_list.append(1)

        return num_contained/total, num_contained2/total, word_coverage_list


def load_gold_data(filename):
    """
    this function aims to load the gold data.
    :param filename:
    :return:
    """
    word_list = []
    lemma_list = []
    pos_list = []
    definition_list = []
    example_list = []

    with open(filename, 'r') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                word_seg, lemma_seg, pos_seg, definition_seg = line.strip().split('\t\t')
                words = word_seg.split("word::: ")[1].split('\t')
                lemma = lemma_seg.split('lemma::: ')[1]
                pos = pos_seg.split('pos::: ')[1]
                definition = definition_seg.split('definition::: ')[1]

                word_list.append(words)
                lemma_list.append(lemma)
                pos_list.append(pos)
                definition_list.append(definition)
            else:
                examples = line.strip().split('Reference examples::: ')[1].split('\t\t')
                example_list.append(examples)
    same_word_example_list = []
    one_reference_example_list = []

    s = 0
    for examples, words in zip(example_list, word_list):
        word = words[0]
        one_reference_example_list.append([examples[0]])
        l = []
        for e, w in zip(examples, words):
            if w==word:
                l.append(e)
                s+=1
        same_word_example_list.append(l)
    print(f'The length of the same_word_reference_example_list is {s}.')

    gold_data = {
    "word_list": word_list,
    "lemma_list": lemma_list,
    "pos_list": pos_list,
    "definition_list": definition_list,
    "reference_example_list": example_list,
    "same_word_reference_example_list": same_word_example_list,
    'one_reference_example_list': one_reference_example_list
    }
    return gold_data

def load_generated_data(filename):
    """
    this function aims to load the generated data.
    :param filename:
    :return:
    """
    word_list = []
    lemma_list = []
    pos_list = []
    definition_list = []
    example_list = []
    reference_example_list = []

    example_tokenization_length_list = []
    # lexical complexity features
    word_rank_lexical_complexity_list = []
    token_rank_lexical_complexity_list = []
    external_word_rank_lexical_complexity_list = []
    external_token_rank_lexical_complexity_list = []
    flesch_reading_ease_list = []
    flesch_kincaid_grade_level_list = []
    with open(filename, 'r') as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            if i % 3 == 0:
                word_seg, lemma_seg, pos_seg, definition_seg, example_length_seg, \
                word_rank_lexical_complexity_seg, token_rank_lexical_complexity_seg, \
                external_word_rank_lexical_complexity_seg, external_token_rank_lexical_complexity_seg, \
                flesch_reading_ease_seg, flesch_kincaid_grade_level_seg = line.split('\t\t')
                word = word_seg.split("word::: ")[1]
                lemma = lemma_seg.split('lemma::: ')[1]
                pos = pos_seg.split('pos::: ')[1]
                definition = definition_seg.split('definition::: ')[1]
                example_tokenization_length = int(example_length_seg.split('example_tokenization_length::: ')[1])

                word_rank_lexical_complexity = float(
                    word_rank_lexical_complexity_seg.split('word_rank_lexical_complexity::: ')[1])
                token_rank_lexical_complexity = float(
                    token_rank_lexical_complexity_seg.split('token_rank_lexical_complexity::: ')[1])

                external_word_rank_lexical_complexity = float(external_word_rank_lexical_complexity_seg.split
                                                              ('external_word_rank_lexical_complexity::: ')[1])
                external_token_rank_lexical_complexity = float(external_token_rank_lexical_complexity_seg.split
                                                               ('external_token_rank_lexical_complexity::: ')[1])

                flesch_reading_ease = float(flesch_reading_ease_seg.split('flesch_reading_ease::: ')[1])
                flesch_kincaid_grade_level = float(
                    flesch_kincaid_grade_level_seg.split('flesch_kincaid_grade_level::: ')[1])


                word_list.append(word)
                lemma_list.append(lemma)
                pos_list.append(pos)
                definition_list.append(definition)
                example_tokenization_length_list.append(example_tokenization_length)
                word_rank_lexical_complexity_list.append(word_rank_lexical_complexity)
                token_rank_lexical_complexity_list.append(token_rank_lexical_complexity)
                external_word_rank_lexical_complexity_list.append(external_word_rank_lexical_complexity)
                external_token_rank_lexical_complexity_list.append(external_token_rank_lexical_complexity)
                flesch_reading_ease_list.append(flesch_reading_ease)
                flesch_kincaid_grade_level_list.append(flesch_kincaid_grade_level)

            elif i%3==1:
                examples = line.split('Reference examples::: ')[1].split('\t\t')
                reference_example_list.append(examples)
            elif i%3==2:
                example = line.split('The generated example::: ')[1]
                example_list.append(example)
    generated_data = {
            "word_list": word_list,
            "lemma_list": lemma_list,
            "pos_list": pos_list,
            "definition_list": definition_list,
            "example_list": example_list,
            "reference_example_list":reference_example_list,
            "example_tokenization_length_list":example_tokenization_length_list,
            # lexical complexity features
            "word_rank_lexical_complexity_list":word_rank_lexical_complexity_list,
            "token_rank_lexical_complexity_list":token_rank_lexical_complexity_list,
            "external_word_rank_lexical_complexity_list":external_word_rank_lexical_complexity_list,
            "external_token_rank_lexical_complexity_list": external_token_rank_lexical_complexity_list,
            "flesch_reading_ease_list":flesch_reading_ease_list,
            "flesch_kincaid_grade_level_list":flesch_kincaid_grade_level_list
    }
    return generated_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Automatic evaluation for generated sentences.")
    parser.add_argument('--dataset', type=str, default='oxford')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'validation'])
    parser.add_argument('--polysemous', type=int, default=1, help = 'whether evaluate on the polysemous test/validation set.', choices=[0,1])
    parser.add_argument('--num_tokens', type=int, default=0,
                        help='0 denotes using all the rest sentences as the reference to compute diversity, '
                             'otherwise use num_tokens as the reference.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bart_tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-base')
    # load gpt2 model
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    GPT2_model.to(device)

    # load the pre-trained model and tokenizer
    model_path = f'../checkpoints/{args.dataset}_bert-base-cased_definition/lr_1e-05'
    bert_definition_tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_definition_model = BertForSequenceClassification.from_pretrained(model_path)
    bert_definition_model = bert_definition_model.to(device)

    model_path = f'../checkpoints/{args.dataset}_bert-base-cased_pos/lr_1e-05'
    bert_pos_tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_pos_model = BertForSequenceClassification.from_pretrained(model_path)
    bert_pos_model = bert_pos_model.to(device)

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['Bleu_1','ROUGE_L', 'CIDEr'])  # loads the models

    evaluation = Evaluation( model=GPT2_model, tokenizer=GPT2_tokenizer,
                             bert_definition_model = bert_definition_model,
                             bert_definition_tokenizer = bert_definition_tokenizer,
                             bert_pos_model = bert_pos_model,
                             bert_pos_tokenizer=bert_pos_tokenizer,
                             device = device)
    if args.polysemous:
        gold_data = load_gold_data(f"../data/oxford/polysemous_{args.mode}_inference.txt")
    else:
        gold_data = load_gold_data(f"../data/oxford/{args.mode}_inference.txt")


    path_dir = f'../polysemous_outputs/oxford_bart*/*.txt'
    filenames = glob.glob(path_dir)
    filenames = sorted(filenames)

    ouput_file = 'polysemous_results.txt'
    print(f"Evaluate {len(filenames)} files.")
    print(f'Evaluate {filenames}.')
    print(f'Output the evauation results into {ouput_file}.')
    
    # tokenize the gold data
    tokenized_reference_sentences_list = evaluation.tokenize(gold_data['reference_example_list'])
    tokenized_same_word_reference_sentences_list = evaluation.tokenize(gold_data['same_word_reference_example_list'])
    tokenized_one_reference_sentences_list = evaluation.tokenize(gold_data['one_reference_example_list'])

    one_references = []
    for l in tokenized_one_reference_sentences_list:
        new_l = []
        for s in l:
            new_l.append(" ".join(s))
        one_references.append(new_l)
    one_references = list(zip(*one_references))

    max_num_examples = max([len(e) for e in gold_data['reference_example_list']])
    multiple_references = []
    for l in tokenized_reference_sentences_list:
        new_l = []
        for s in l:
            new_l.append(" ".join(s))
        new_l += (max_num_examples - len(new_l)) * ['']
        multiple_references.append(new_l)
    multiple_references = list(zip(*multiple_references))

    max_num_examples = max([len(e) for e in gold_data['same_word_reference_example_list']])
    same_word_references = []
    for l in tokenized_same_word_reference_sentences_list:
        new_l = []
        for s in l:
            new_l.append(" ".join(s))
        new_l += (max_num_examples - len(new_l)) * ['']
        same_word_references.append(new_l)
    same_word_references = list(zip(*same_word_references))

    with open(ouput_file,'a') as fw:
        fw.write('File name, GPT2-NLL, '
                #  'BLEU2, BLEU4, NIST2, NIST4, Meteor, '
                 'BLEU2_same_word, BLEU4_same_word, '
                #  NIST2_same_word, NIST4_same_word, Meteor_same_word, '
                #  'BLEU2_one, BLEU4_one, NIST2_one, NIST4_one, Meteor_one, '
                 'Dist1, Dist2, Dist3, Dist4, '
                 'Sentence repetition, Sentence length, Sentence Length MSE/std, Bert pos accuracy, Bert definition accuracy, '
                 'Word Coverage, Inflection Coverage\n')
        # example_list = [e[0] for e in gold_data['reference_example_list']]

        # NLL, _ = evaluation.compute_average_perplexity(example_list, batch_size=args.batch_size)
        # tokenized_sentences_list = evaluation.tokenize(example_list)

        # # BLEU2 = evaluation.compute_bleu(tokenized_sentences_list, [l[1:] for l in tokenized_reference_sentences_list],
        # #                                 weights=[0.5, 0.5])
        # # BLEU4 = evaluation.compute_bleu(tokenized_sentences_list, [l[1:] for l in tokenized_reference_sentences_list],
        # #                                 weights=[0.25, 0.25, 0.25, 0.25])

        # # NIST2 = evaluation.compute_nist(tokenized_sentences_list, [l[1:] for l in tokenized_reference_sentences_list],
        # #                                 ngrams=2)
        # # NIST4 = evaluation.compute_nist(tokenized_sentences_list, [l[1:] for l in tokenized_reference_sentences_list],
        # #                                 ngrams=4)

        # BLEU2_same = evaluation.compute_bleu(tokenized_sentences_list,
        #                                      [l[1:] for l in tokenized_same_word_reference_sentences_list],
        #                                      weights=[0.5, 0.5])
        # BLEU4_same = evaluation.compute_bleu(tokenized_sentences_list,
        #                                      [l[1:] for l in tokenized_same_word_reference_sentences_list],
        #                                      weights=[0.25, 0.25, 0.25, 0.25])

        # # NIST2_same = evaluation.compute_nist(tokenized_sentences_list,
        # #                                      [l[1:] for l in tokenized_same_word_reference_sentences_list], ngrams=2)
        # # NIST4_same = evaluation.compute_nist(tokenized_sentences_list,
        # #                                      [l[1:] for l in tokenized_same_word_reference_sentences_list], ngrams=4)

        # # BLEU2_1 = evaluation.compute_bleu(tokenized_sentences_list, [ [l[1]] for l in tokenized_same_word_reference_sentences_list],
        # #                                   weights=[0.5, 0.5])
        # # BLEU4_1 = evaluation.compute_bleu(tokenized_sentences_list, [ [l[1]] for l in tokenized_same_word_reference_sentences_list],
        # #                                   weights=[0.25, 0.25, 0.25, 0.25])

        # # NIST2_1 = evaluation.compute_nist(tokenized_sentences_list, [ [l[1]] for l in tokenized_same_word_reference_sentences_list],
        # #                                   ngrams=2)
        # # NIST4_1 = evaluation.compute_nist(tokenized_sentences_list, [ [l[1]] for l in tokenized_same_word_reference_sentences_list],
        # #                                   ngrams=4)

        # # generated_sentences = [" ".join(e) for e in tokenized_sentences_list]

        # # metrics_dict = nlgeval.compute_metrics(multiple_references[1:], generated_sentences)
        # # METEOR = metrics_dict['METEOR']

        # # metrics_dict = nlgeval.compute_metrics(same_word_references[1:], generated_sentences)
        # # METEOR_same = metrics_dict['METEOR']

        # # metrics_dict = nlgeval.compute_metrics([same_word_references[1]], generated_sentences)
        # # METEOR_1 = metrics_dict['METEOR']

        # distances, entropies = evaluation.distances_and_entropy(tokenized_sentences_list, ngrams=[1, 2, 3, 4],
        #                                                         num_tokens=args.num_tokens)
        # repetition_array = evaluation.repetition(tokenized_sentences_list, n_grams=[1, 2, 3, 4],
        #                                          thresholds=[3, 3, 2, 2])

        # example_length_list = []
        # for sentence in example_list:
        #     example_length = bart_tokenizer.tokenize(sentence)
        #     example_length_list.append(len(example_length))
        # ave_len = np.mean(example_length_list)
        # mse_len = np.std(example_length_list)
        # word_lemma_pos_definition_example_list = []
        # for words, lemma, pos, definition, example in zip(gold_data['word_list'],
        #                                                   gold_data['lemma_list'],
        #                                                   gold_data['pos_list'],
        #                                                   gold_data['definition_list'],
        #                                                   gold_data['reference_example_list']):
        #     word_lemma_pos_definition_example_list.append([words[0], lemma, pos, definition, example[0]])

        # pos_accuracy, _, _ = evaluation.evaluate_word_pos(word_lemma_pos_definition_example_list,
        #                                                   batch_size=args.batch_size)
        # definition_accuracy, _, _ = evaluation.evaluate_word_definition(word_lemma_pos_definition_example_list,
        #                                                                 batch_size=args.batch_size)

        # fw.write(f'Gold data, '
        #          f'{NLL:.3f}, '
        #         #  f'{BLEU2:.3f}, {BLEU4:.3f}, {NIST2:.3f}, {NIST4:.3f}, {METEOR:.3f}, '
        #          f'{BLEU2_same:.3f}, {BLEU4_same:.3f}, '
        #         #  '{NIST2_same:.3f}, {NIST4_same:.3f}, {METEOR_same:.3f}, '
        #         #  f'{BLEU2_1:.3f}, {BLEU4_1:.3f}, {NIST2_1:.3f}, {NIST4_1:.3f}, {METEOR_1:.3f}, '
        #          f'{distances[0]:.3f}, {distances[1]:.3f}, {distances[2]:.3f}, {distances[3]:.3f}, '
        #          f'{repetition_array[4]:.3f}, '
        #          f'{ave_len:.1f}, {mse_len:.3f}, '
        #          f'{pos_accuracy:.3f}, {definition_accuracy:.3f}, '
        #          f'1, 1\n')
        # fw.write('\n')
        # fw.flush()



        start = time.time()
        for i, filename in enumerate(filenames):
            # print(filename)
            generated_data = load_generated_data(filename)

            NLL, _ = evaluation.compute_average_perplexity(generated_data['example_list'], batch_size =args.batch_size)
            tokenized_generated_sentences_list = evaluation.tokenize(generated_data['example_list'])
            assert len(tokenized_generated_sentences_list) == len(tokenized_reference_sentences_list)

            # BLEU2 = evaluation.compute_bleu(tokenized_generated_sentences_list, tokenized_reference_sentences_list,
            #                                 weights=[0.5, 0.5])
            # BLEU4 = evaluation.compute_bleu(tokenized_generated_sentences_list, tokenized_reference_sentences_list,
            #                                 weights=[0.25, 0.25, 0.25, 0.25])

            # NIST2 = evaluation.compute_nist(tokenized_generated_sentences_list, tokenized_reference_sentences_list, ngrams=2)
            # NIST4 = evaluation.compute_nist(tokenized_generated_sentences_list, tokenized_reference_sentences_list, ngrams=4)


            BLEU2_same = evaluation.compute_bleu(tokenized_generated_sentences_list, tokenized_same_word_reference_sentences_list,
                                            weights=[0.5, 0.5])
            BLEU4_same = evaluation.compute_bleu(tokenized_generated_sentences_list, tokenized_same_word_reference_sentences_list,
                                            weights=[0.25, 0.25, 0.25, 0.25])

            # NIST2_same = evaluation.compute_nist(tokenized_generated_sentences_list,
            #                                 tokenized_same_word_reference_sentences_list, ngrams=2)
            # NIST4_same = evaluation.compute_nist(tokenized_generated_sentences_list,
            #                                 tokenized_same_word_reference_sentences_list, ngrams=4)

            # BLEU2_1 = evaluation.compute_bleu(tokenized_generated_sentences_list, tokenized_one_reference_sentences_list,
            #                                 weights=[0.5, 0.5])
            # BLEU4_1 = evaluation.compute_bleu(tokenized_generated_sentences_list, tokenized_one_reference_sentences_list,
            #                                 weights=[0.25, 0.25, 0.25, 0.25])

            # NIST2_1 = evaluation.compute_nist(tokenized_generated_sentences_list, tokenized_one_reference_sentences_list, ngrams=2)
            # NIST4_1 = evaluation.compute_nist(tokenized_generated_sentences_list, tokenized_one_reference_sentences_list, ngrams=4)

            # generated_sentences = [" ".join(e) for e in tokenized_generated_sentences_list]

            # metrics_dict = nlgeval.compute_metrics(multiple_references, generated_sentences)
            # METEOR = metrics_dict['METEOR']

            # metrics_dict = nlgeval.compute_metrics(same_word_references, generated_sentences)
            # METEOR_same = metrics_dict['METEOR']

            # metrics_dict = nlgeval.compute_metrics(one_references, generated_sentences)
            # METEOR_1 = metrics_dict['METEOR']

            distances, entropies = evaluation.distances_and_entropy(tokenized_generated_sentences_list, ngrams=[1, 2, 3, 4],
                                                                    num_tokens=args.num_tokens)
            repetition_array = evaluation.repetition(tokenized_generated_sentences_list, n_grams=[1,2,3,4], thresholds=[3,3,2,2])

            if "add_space" in filename:
                _add_space = True
            else:
                _add_space = False
            example_length_list = []
            for sentence in generated_data['example_list']:
                if _add_space is True:
                    sentence = ' '+sentence
                tokenized_example = bart_tokenizer.tokenize(sentence)
                generated_example_length = len(tokenized_example)
                example_length_list.append(generated_example_length)
            ave_len = np.mean(example_length_list)
            if 'use_example_len' in filename: # compute mean square error
                mse_len_list = []
                if "expected_len" in filename:
                    reference_example_length = int(filename.split("expected_len_")[1].split('_')[0])
                    print(filename)
                    print(reference_example_length)
                    for j, generated_example_length in enumerate(example_length_list):
                        mse_len_list.append((generated_example_length-reference_example_length)**2)
                else:
                    for j, reference_example_length in enumerate(generated_data['example_tokenization_length_list']):
                        generated_example_length = example_length_list[j]
                        mse_len_list.append((generated_example_length-reference_example_length)**2)
                mse_len = np.mean(mse_len_list)
            else: # compute the standard error
                mse_len = np.std(example_length_list)

            word_coverage, inflection_coverage, word_coverage_list = evaluation.word_coverage(
                                                                                        gold_data['word_list'], generated_data['example_list'],
                                                                                        generated_data['lemma_list'], generated_data['pos_list'])

            word_lemma_pos_definition_example_list = []
            for word, lemma, pos, definition, example in zip(generated_data['word_list'],
                                                             generated_data['lemma_list'],
                                                             generated_data['pos_list'],
                                                             generated_data['definition_list'],
                                                             generated_data['example_list']):
                word_lemma_pos_definition_example_list.append([word, lemma, pos, definition, example])

            pos_accuracy, _, _ = evaluation.evaluate_word_pos(word_lemma_pos_definition_example_list,
                                                              batch_size=args.batch_size,
                                                              word_coverage_list = word_coverage_list)
            definition_accuracy, _, _ = evaluation.evaluate_word_definition(word_lemma_pos_definition_example_list,
                                                                            batch_size=args.batch_size,
                                                                            word_coverage_list=word_coverage_list)


            fw.write(f'{filename}, '
                     f'{NLL:.3f}, '
                    #  f'{BLEU2:.3f}, {BLEU4:.3f}, {NIST2:.3f}, {NIST4:.3f}, {METEOR:.3f}, '
                      f'{BLEU2_same:.3f}, {BLEU4_same:.3f}, '
                    #   f'{NIST2_same:.3f}, {NIST4_same:.3f}, {METEOR_same:.3f}, '
                    #  f'{BLEU2_1:.3f}, {BLEU4_1:.3f}, {NIST2_1:.3f}, {NIST4_1:.3f}, {METEOR_1:.3f}, '
                     f'{distances[0]:.3f}, {distances[1]:.3f}, {distances[2]:.3f}, {distances[3]:.3f}, '
                     f'{repetition_array[4]:.3f}, '
                     f'{ave_len:.1f}, {mse_len:.3f}, '
                     f'{pos_accuracy:.3f}, {definition_accuracy:.3f}, '
                     f'{word_coverage:.3f}, {inflection_coverage:.3f}\n')
            # if i%8==7:
            #     fw.write('\n')
            fw.flush()
            print(f'\r Process {i+1}/{len(filenames)} used {time.time()-start:.1f} seconds.', end='')
