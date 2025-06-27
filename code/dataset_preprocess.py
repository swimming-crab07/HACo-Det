from datasets import load_dataset, concatenate_datasets
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import torch
from transformers import AutoTokenizer

DATASETS = ['paper','story','news','wikipedia']
MODELS = ['llama', 'mixtral', 'gpt-4o', 'gpt-4o-mini']
datasets_fullnames = {'story': 'euclaise/writingprompts',
                    'news': 'xsum',
                    'wikipedia': 'wikipedia',
                    'paper': "./dataset/processed_paper_data.json"
                     }
datasets_tokens = {'story': 1700,
                    'news': 1300,
                    'wikipedia': 6000,
                    'paper': 6000,
                     }
datasets_keys = {'story': 'story',
                     'news': 'text',
                     'wikipedia': 'text',
                     'paper': 'abstract',
                     }
PAPER_SEED = 42
SAMPLE_NUMBER = 100
TRAIN_NUMBER = 400
VAL_NUMBER = 100
TEST_NUMBER = 200


if __name__ == "__main__":
    for domain in DATASETS:
        if domain == 'paper':
            dataset = load_dataset("json" , data_files="dataset/processed_paper_data.json", split="train")
        elif domain == 'news' or domain == 'clean_news':
            xsum_dataset = load_dataset(datasets_fullnames[domain], cache_dir="../cache")
            xsum_dataset = concatenate_datasets([xsum_dataset['train'], xsum_dataset['test'], xsum_dataset['validation']]).shuffle(seed=PAPER_SEED)
            xsum_dataset = xsum_dataset.rename_column('document', 'text')
            dataset = xsum_dataset
        else:
            if domain == 'wikipedia':
                dataset = load_dataset(datasets_fullnames[domain], "20220301.en", cache_dir="../cache")
            else:
                dataset = load_dataset(datasets_fullnames[domain], cache_dir="../cache")
            if len(dataset) > 1:
                all_dataset=[dataset[dataset_split] for dataset_split in dataset]    
                dataset = concatenate_datasets(all_dataset).shuffle(seed=PAPER_SEED)
            else:
                dataset = dataset['train']
        tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir="../cache",
        use_fast=True,
        )
        datasets = []
        for i in range(4):
            seg_dataset = dataset.shard(num_shards=4, index=i)
            seg_dataset = seg_dataset.train_test_split(test_size=0.43)
            dataset_remain = seg_dataset['train'].select(range(TRAIN_NUMBER,len(seg_dataset['train'])))
            dataset_remain = concatenate_datasets([dataset_remain, seg_dataset['test'].select(range(VAL_NUMBER + TEST_NUMBER,len(seg_dataset['test'])))])
            seg_dataset['train'] = seg_dataset['train'].select(range(TRAIN_NUMBER))
            seg_dataset['validation'] = seg_dataset['test'].select(range(VAL_NUMBER))
            seg_dataset['test'] = seg_dataset['test'].select(range(VAL_NUMBER,VAL_NUMBER + TEST_NUMBER))
            datasets.append(seg_dataset)
        print(datasets)
        for i in range(len(datasets)):
            datasets[i]['train'].to_json('./dataset/' + domain + '_' + MODELS[i] + '_train.json')
            datasets[i]['validation'].to_json('./dataset/' + domain + '_' + MODELS[i] +'_validation.json')
            datasets[i]['test'].to_json('./dataset/' + domain + '_' + MODELS[i] + '_test.json')
            dataset_remain.to_json('./dataset/' + domain + '_' + MODELS[i] + '_remain.json')

