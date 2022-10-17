# -*- coding: utf-8 -*-
# @Time    : 2021/3/4 3:40 PM
# @Author  : He Xingwei

"""
this file is used to extract features, such lexical complexity of examples.
"""
import sys
sys.path.append('../')
from utils.readability import WordRank, Readability
import time

word_rank_extractor = WordRank('oxford',token_level=False, use_external_data=False)
token_rank_extractor = WordRank('oxford',token_level=True, use_external_data=False, add_space = True)
token_rank_extractor_2 = WordRank('oxford',token_level=True, use_external_data=False, add_space = False)

external_word_rank_extractor = WordRank('oxford', token_level=False, use_external_data=True)
external_token_rank_extractor = WordRank('oxford',token_level=True, use_external_data=True, add_space = True)
external_token_rank_extractor_2 = WordRank('oxford',token_level=True, use_external_data=True, add_space = False)

readability_extractor = Readability()
#
for filename in ['../data/oxford/training.txt','../data/oxford/validation.txt','../data/oxford/test.txt']:
    with open(filename, 'r') as fr, open(filename.replace('.txt','_features.txt'), 'w') as fw:
        i = 0
        start = time.time()
        for line in fr:
            line = line.strip()
            word_seg, lemma_seg, pos_seg, definition_seg, example_seg = line.split('\t\t')
            word = word_seg.split("word::: ")[1].strip()
            lemma = lemma_seg.split('lemma::: ')[1].strip()
            pos = pos_seg.split('pos::: ')[1].strip()
            definition = definition_seg.split('definition::: ')[1].strip()
            example = example_seg.split('example::: ')[1].strip()
            word_rank_lexical_complexity = word_rank_extractor.get_lexical_complexity_score(example)
            token_rank_lexical_complexity = token_rank_extractor.get_lexical_complexity_score(example)
            token_rank_lexical_complexity_2 = token_rank_extractor_2.get_lexical_complexity_score(example)

            external_word_rank_lexical_complexity = external_word_rank_extractor.get_lexical_complexity_score(example)
            external_token_rank_lexical_complexity = external_token_rank_extractor.get_lexical_complexity_score(example)
            external_token_rank_lexical_complexity_2 = external_token_rank_extractor_2.get_lexical_complexity_score(example)

            total_words, total_sentences, total_syllables = readability_extractor.analyze_text(example)
            if total_words==0 or total_sentences==0:
                print(total_words, total_sentences, total_syllables, example, line)
            flesch_reading_ease = readability_extractor.FleschReadingEase(total_words, total_sentences, total_syllables)
            flesch_kincaid_grade_level = readability_extractor.FleschKincaidGradeLevel(total_words, total_sentences, total_syllables)

            assert line in f"word::: {word}\t\tlemma::: {lemma}\t\tpos::: {pos}\t\t" \
                           f"definition::: {definition}\t\texample::: {example}\t\t" \
                           f"word_rank_lexical_complexity::: {word_rank_lexical_complexity}\t\t" \
                           f"token_rank_lexical_complexity_add_space::: {token_rank_lexical_complexity}\t\t" \
                           f"token_rank_lexical_complexity::: {token_rank_lexical_complexity_2}\t\t" \
                           f"external_word_rank_lexical_complexity::: {external_word_rank_lexical_complexity}\t\t" \
                           f"external_token_rank_lexical_complexity_add_space::: {external_token_rank_lexical_complexity}\t\t" \
                           f"external_token_rank_lexical_complexity::: {external_token_rank_lexical_complexity_2}\t\t" \
                           f"flesch_reading_ease::: {flesch_reading_ease}\t\t" \
                           f"flesch_kincaid_grade_level::: {flesch_kincaid_grade_level}\n"

            fw.write(f"word::: {word}\t\tlemma::: {lemma}\t\tpos::: {pos}\t\t"
                     f"definition::: {definition}\t\texample::: {example}\t\t"
                     f"word_rank_lexical_complexity::: {word_rank_lexical_complexity}\t\t"
                     f"token_rank_lexical_complexity_add_space::: {token_rank_lexical_complexity}\t\t"
                     f"token_rank_lexical_complexity::: {token_rank_lexical_complexity_2}\t\t"
                     f"external_word_rank_lexical_complexity::: {external_word_rank_lexical_complexity}\t\t"
                     f"external_token_rank_lexical_complexity_add_space::: {external_token_rank_lexical_complexity}\t\t"
                     f"external_token_rank_lexical_complexity::: {external_token_rank_lexical_complexity_2}\t\t"
                     f"flesch_reading_ease::: {flesch_reading_ease}\t\t"
                     f"flesch_kincaid_grade_level::: {flesch_kincaid_grade_level}\n")
            i +=1
            if i%100==0:
                print(f'\rProcess {i}, used {time.time()-start:.1f}',end='')
        print()
