# -*- coding: utf-8 -*-
# @Time    : 2021/3/16 5:41 PM
# @Author  : He Xingwei

from transformers import BartTokenizer,GPT2Tokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
print(tokenizer.mask_token)
print(tokenizer.pad_token)
# print(tokenizer.convert_ids_to_tokens(tokenizer.encode('hope this is a good place .',add_special_tokens=True)))
vocab = []
for i in range(tokenizer.vocab_size):
    vocab.append((tokenizer.convert_ids_to_tokens([i])[0], i))
with open('../data/tokens/bart_vocab.txt','w') as fw, open('../data/tokens/bart_stop_tokens.txt','w') as fw2, \
        open('../data/tokens/bart_sub_tokens.txt','w') as fw3:
    for (word, id) in vocab:
        fw.write(str(id)+'\t'+word+'\n')
        if ')' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '(' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '..' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '<s>' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '</s>' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '<pad>' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '<mask>' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        else:
            pass

        if ord(word[0]) != 288:
            fw3.write(str(id) + '\t' + word + '\n')


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(tokenizer.mask_token_id)
print(tokenizer.pad_token_id)
j = 0
vocab = []
for i in range(tokenizer.vocab_size):
    vocab.append((tokenizer.convert_ids_to_tokens([i])[0], i))
with open('../data/tokens/gpt2_vocab.txt','w') as fw, open('../data/tokens/gpt2_stop_tokens.txt','w') as fw2, \
        open('../data/tokens/gpt2_sub_tokens.txt','w') as fw3:
    for (word, id) in vocab:
        fw.write(str(id)+'\t'+word+'\n')
        if ')' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '(' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        elif '..' in word:
            fw2.write(str(id) + '\t' + word + '\n')
        # elif '<|endoftext|>' in word:
        #     fw2.write(str(id) + '\t' + word + '\n')
        else:
            pass

        if ord(word[0]) != 288:
            fw3.write(str(id) + '\t' + word + '\n')