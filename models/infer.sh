
# greedy decoding
python infer.py --test_mode 1 --batch_size 20 --decoding_strategy 1 --gpu 4 --use_word 0 --use_pos 1 --use_example_len 1 --use_lexical_complexity 1  --add_space 1 --expected_len 14 --expected_lexical_complexity 25 --initialization bart-base
# beam search decoding
python infer.py --test_mode 1 --batch_size 20 --decoding_strategy 1 --gpu 4 --use_word 0 --use_pos 1 --use_example_len 1 --use_lexical_complexity 1  --add_space 1 --expected_len 14 --expected_lexical_complexity 25 --initialization bart-base --num_beams 5


