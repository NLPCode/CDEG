
# README
This repository contains the implementation of the ACL 2022 paper: 
"[**Controllable Dictionary Example Generation: Generating Example Sentences for Specific Targeted Audiences**](https://aclanthology.org/2022.acl-long.46/)".
****
##  Abstract
Example sentences for targeted words in a dictionary play an important role to help readers understand the usage of words. Traditionally, example sentences in a dictionary are usually created by linguistics experts, which are labor-intensive and knowledge-intensive. In this paper, we introduce the problem of dictionary example sentence generation, aiming to automatically generate dictionary example sentences for targeted words according to the corresponding definitions. This task is challenging especially for polysemous words, because the generated sentences need to reflect different usages and meanings of these targeted words. Targeted readers may also have different backgrounds and educational levels. It is essential to generate example sentences that can be understandable for different backgrounds and levels of audiences. To solve these problems, we propose a controllable target-word-aware model for this task. Our proposed model can generate reasonable examples for targeted words, even for polysemous words. In addition, our model allows users to provide explicit control over attributes related to readability, such as length and lexical complexity, thus generating suitable examples for targeted audiences. Automatic and human evaluations on the Oxford dictionary dataset show that our model can generate suitable examples for targeted words with specific definitions while meeting the desired readability.
****
## Requirements
python 3.6  
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  
pip install transformers==4.3.3  
pip install lemminflect==0.2.1  
pip install inflect==5.2.0  
pip install nltk==3.6.2  
****
## Dataset
The oxford dataset used in our paper are available at http://www.statmt.org/lm-benchmark/: 
You should download the data, put them into the root directory of this project, and decompress them with the following command:
```bash
tar -xzvf data.tar.gz # replace 'checkpoint_name' with the corresponding checkpoint name.
```
In the decompressed 'data' directory , there are three 'txt' files, which are used to compute the readability features. These files are constructed based on part of the [One-Billion-Word](http://www.statmt.org/lm-benchmark/) corpus.

In the 'data/oxford' directory, there are of two versions of data. 
Full version: the training/validation/test contains words with only one definition and words with multiple definitions;
Ploysemous version: the training/validation/test only contains words with multiple definitions. 
****
## Try our model with the well-trained generation model, POS and definition evalution model checkpoints: 
| Model           |  Download link
|----------------------|--------|
| Generation Model| [\[link\]](https://drive.google.com/file/d/1JPPhqdapW_p2AQ9jyx0MuYeD31gHuQAD/view?usp=sharing)  | 
| POS Evaluation Model| [\[link\]](https://drive.google.com/file/d/1tbkF2yAEFJ-wE6iG2nd_iWxzCXfH2boU/view?usp=sharing)  | 
| Definition Evalution Model| [\[link\]](https://drive.google.com/file/d/1A6BU_hc3O5ppy89im4g3Z9hXVUkgFqnw/view?usp=sharing)  | 

If you want to quickly try our model, you should download these checkpoints, put them into the 'checkpoints' directory, and decompress them with the following command:

```bash
tar -xzvf checkpoint_name.tar.gz # replace 'checkpoint_name' with the corresponding checkpoint name.
```

Then, you need to generate the data used for inference:
```bash
cd models
python train.py --gpu 6 --batch_size 40 --test_batch_size 80 --use_word 0 --use_pos 1 --use_example_len 1 --use_lexical_complexity 1 --train 1 --epochs 0
```
Finally, you can directly go to [Generate example sentences for the given words and definitions](#generate).


If you want to train our model from scratch, please refer to the following steps.
****
## Train our model from scratch 
For this case, you only need to download the pos and definition evalution checkpoints, which will be used to evaluate the outputs.

* Step 1: Create features such as length and lexical complexity.

```bash
cd utils  
python feature_extraction.py
```

* Step 2: Train the generation model with word, length, POS, and lexical complexity features.
```bash
cd models
sh train.sh
```


## <span id="generate"> Generate example sentences for the given words and definitions </span>
Generate dictionary examples with specified length and lexically complexity. 
In the inference script, we set the length to 14 and lexicially to 25. You can freely change both values and test the effects of these two features, but you need to pay attention to the range of these two features (please refer to our paper for more details). 
In the inference script, we generate dictionary examples by running greed and beam search decoding strategies on the generation model and generated results are stored in the 'outputs' directory. 
```bash
cd models
sh infer.sh 
```

## Evaluation

Note that the default generation model is trained on the training set of the full version data, while we verify the model on the test set of the polysemous version data. 
(We have clearly mentioned this in the paper: "Different from the training set, the validation/test set only contains polysemous words with at least two definitions, since it is more challenging to generate examples for polysemy.")
* Step 1: Extract results for the polysemous words and store the results in the 'polysemous_outputs' directory.
```bash
cd evaluations
python extract_polysemous_outputs.py
```

* Step 2: Evaluate results of the polysemous test set.
```bash
cd evaluations
python extract_polysemous_outputs.py
```
If you want to compute Self-BLEU, then you can run the following script:

```bash
cd evaluations
python self-bleu_high-frequency_lexical-complexity.py
```

## Citation
If you want to use this code in your research, you can cite our [paper](https://aclanthology.org/2022.acl-long.46/):
```bash

@inproceedings{he-yiu-2022-controllable,
    title = "Controllable Dictionary Example Generation: Generating Example Sentences for Specific Targeted Audiences",
    author = "He, Xingwei  and Yiu, Siu Ming",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.46",
    doi = "10.18653/v1/2022.acl-long.46",
    pages = "610--627",
}

```
