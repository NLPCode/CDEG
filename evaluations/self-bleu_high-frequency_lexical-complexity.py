# -*- coding: utf-8 -*-
# @Time    : 2021/4/19 10:57 PM
# @Author  : He Xingwei
from multiprocessing import Pool
import sys
sys.path.append('../')
from automatic_evaluation import Evaluation,load_gold_data,load_generated_data
from utils.readability import WordRank, Readability
from utils.functions import set_seed, convert_continues_to_discrete
import numpy as np

word_rank_extractor = WordRank('oxford',token_level=False, use_external_data=False)
external_word_rank_extractor = WordRank('oxford', token_level=False, use_external_data=True)
lexical_complexity_feature_dict = {
    1: 'word_rank_lexical_complexity',
    2: 'token_rank_lexical_complexity',
    3: 'external_word_rank_lexical_complexity',
    4: 'external_token_rank_lexical_complexity',
    5: 'flesch_reading_ease',
    6: 'flesch_kincaid_grade_level'
}

high_frequency_list = [100, 500, 1000, 2000, 5000, 10000]


def compute_self_bleu_high_frequency(line, sentence_list, weights=[0.25, 0.25, 0.25, 0.25], number_sentences = 1000, num_tokens=0,
                                     high_frequency_list=[100, 500, 1000, 2000, 5000, 10000]):
    """

    :param sentence_list:
    :param weights:
    :param number_sentences:
    :param num_tokens:
    :param high_frequency_list:
    :return:
    """
    tokenized_sentences_list = Evaluation.tokenize(sentence_list)

    selfbleu = Evaluation.compute_self_bleu(tokenized_sentences_list, weights=weights,
                          number_sentences = number_sentences, num_tokens=num_tokens)

    proportion_list = word_rank_extractor.get_proportion_high_frequency_word(sentence_list, rank_list=high_frequency_list)

    # compute FleschReadingEase and FleschKincaidGradeLevel
    total_words, total_sentences, total_syllables = 0.0, 0.0, 0.0
    for example in sentence_list:
        _total_words, _total_sentences, _total_syllables = Readability.analyze_text(example)
        total_words += _total_words
        total_sentences += _total_sentences
        total_syllables += _total_syllables
    FleschReadingEase = Readability.FleschReadingEase(total_words, total_sentences, total_syllables)
    FleschKincaidGradeLevel = Readability.FleschKincaidGradeLevel(total_words, total_sentences, total_syllables)

    # FleschReadingEase2 = Readability.FleschReadingEase(total_words, len(sentence_list), total_syllables)
    # FleschKincaidGradeLevel2 = Readability.FleschKincaidGradeLevel(total_words, len(sentence_list), total_syllables)
    ave_syllables = total_syllables/ total_words
    # extract lexical complexity
    if 'use_word_use_pos_use_example_len_max_len_60_lexical_complexity' in line and 'txt' in line:
        if "add_space" in line:
            _add_space = True
            _filename = f"../data/oxford/training_features_interval_add_space.txt"

        else:
            _add_space = False
            _filename = f"../data/oxford/training_features_interval.txt"

        features_interval_dict = {}
        with open(_filename, 'r') as fr:
            for i, _line in enumerate(fr):
                feature_name, min_feature_value, max_feature_value = _line.split('\t\t')
                min_feature_value = float(min_feature_value)
                max_feature_value = float(max_feature_value)
                features_interval_dict[feature_name] = (min_feature_value, max_feature_value)
        use_lexical_complexity = int(
            line.split('use_word_use_pos_use_example_len_max_len_60_lexical_complexity_')[1][:1])
        feature_name = lexical_complexity_feature_dict[use_lexical_complexity]
        min_feature_value, max_feature_value = features_interval_dict[feature_name]

        lexical_complexity_list = []
        for example in sentence_list:
            # if _add_space is True:
            #     example = ' '+sentence
            if use_lexical_complexity == 1:
                word_rank_lexical_complexity = word_rank_extractor.get_lexical_complexity_score(example)
                lexical_complexity_list.append(word_rank_lexical_complexity)
            elif use_lexical_complexity == 3:
                external_word_rank_lexical_complexity = external_word_rank_extractor.get_lexical_complexity_score(
                    example)
                lexical_complexity_list.append(external_word_rank_lexical_complexity)
            else:
                raise ValueError()

        # convert continuous numbers into discrete labels
        lexical_complexity_feature_list, min_num, max_num, start, end, bin_width = \
            convert_continues_to_discrete(lexical_complexity_list, start=min_feature_value,
                                          end=max_feature_value, num_bins=40)

        word_rank_ave = np.mean(lexical_complexity_feature_list)
        word_rank_std = np.std(lexical_complexity_feature_list)
        if 'expected_lexical_complexity' in line:
            expected_lexical_complexity = int(line.split('expected_lexical_complexity_')[1].split('_')[0])
            print(expected_lexical_complexity)
            mse_list = [(e - expected_lexical_complexity) ** 2 for e in lexical_complexity_feature_list]
            mse = np.mean(mse_list)
        else:
            mse = word_rank_std
    else:
        word_rank_ave = -100
        mse =-100

    return [selfbleu] + proportion_list + [word_rank_ave, mse] + [FleschReadingEase, FleschKincaidGradeLevel, ave_syllables]
    # return [FleschReadingEase, FleschKincaidGradeLevel,  ave_syllables]

with open('polysemous_results.txt', 'r') as fr, open('tmp.txt','w') as fw:
# with open('evaluation_results_baselines.txt', 'r') as fr, open('tmp.txt','w') as fw:
    p = Pool(40)
    obj_l = []
    line_dict = {}
    for i, line in enumerate(fr):
        # if i<500:
        #     continue
        line = line.strip()
        if 'Gold data' in line:
            filename = f"../data/oxford/polysemous_test_inference.txt"
            # filename = f"../data/oxford/test_inference.txt"
            data = load_gold_data(filename)
            sentence_list = data['one_reference_example_list']
            sentence_list = [e[0] for e in sentence_list]
        elif 'txt' in line:
            # filename = '../outputs2/'+line.split(',')[0]
            filename = line.split(',')[0]

            if 'baseline' in filename:
                data = load_baseline_generated_data(filename)
            else:
                data = load_generated_data(filename)
            sentence_list = data['example_list']
        else:
            if i==0:
                line_dict[i] = [line]
                line_dict[i].append('SelfBLEU4')
                for e in high_frequency_list:
                    line_dict[i].append(f'High-frequency threshold {e}')
                line_dict[i].append('Average lexical complexity')
                line_dict[i].append('Lexical complexity MSE/std')

                line_dict[i].append('FleschReadingEase')
                line_dict[i].append('FleschKincaidGradeLevel')
                line_dict[i].append('Ave syllables')
                # line_dict[i].append('FleschReadingEase2')
                # line_dict[i].append('FleschKincaidGradeLevel2')
            else:
                line_dict[i] = [line,]
            sentence_list = None
        if sentence_list:
            obj = p.apply_async(compute_self_bleu_high_frequency, args=(line, sentence_list,
                                                                    [0.25, 0.25, 0.25, 0.25],
                                                                    1000, 0, high_frequency_list))
            print(i, len(sentence_list))
            obj_l.append((obj,i, line))
            # if 'lexical_complexity' in line:
            #     obj_l.append((obj,i, ','.join(line.split(',')[:-8])))
            # else:
            #     obj_l.append((obj,i, ','.join(line.split(',')[:-6])))


    p.close()
    p.join()
    for obj, i, line in obj_l:
        results = obj.get()
        line_dict[i] = [line]
        for e in results:
            line_dict[i].append(f'{e:.3f}')
    for i in range(len(line_dict)):
        values = line_dict[i]
        fw.write(', '.join(values)+'\n')
    fw.flush()

# # with open('evaluation_results_baselines.txt', 'r') as fr, open('tmp.txt','r') as fr2:
with open('polysemous_results.txt', 'r') as fr, open('tmp.txt','r') as fr2:

    for i, (line1, line2) in enumerate(zip(fr, fr2)):
        assert line1.strip() in line2
print('Finish!')