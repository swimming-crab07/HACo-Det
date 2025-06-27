from datasets import load_dataset, concatenate_datasets  
import pandas as pd
import torch
from nltk.metrics import edit_distance
from difflib import SequenceMatcher
import json
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np
from tqdm.auto import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _strip_newlines(text):
    return ' '.join(text.split('\n'))

def check_passages(papers, remain_papers):
    idx = 0
    passages = []
    counter= 0
    for paper in tqdm(papers):
        counter = counter + 1
        paper = _strip_newlines(paper)
        paper = ' '.join(paper.split())
        paper = paper.replace("—", "")
        paper = paper.replace("_", "")
        paper = paper.replace("*", "")
        paper = paper.replace("^", "")
        paper = paper.replace(" n't", "n't")
        paper = paper.replace("( ", "(")
        paper = paper.replace(" )", ")")
        paper = paper.replace("''", "\"")
        paper = paper.replace("`` ", "\"")
        paper = paper.replace(" ``", "\"")
        paper = paper.replace("``", "\"")
        paper = paper.replace("“ ", "\"")
        paper = paper.replace(" ”", "\"")
        paper = paper.replace("’", "\'")
        paper = paper.replace("‘ ", "\'")
        paper = paper.replace(" ' ", "\'")
        paper = paper.replace("Media playback is not supported on this device", "")
        paper = paper.replace("[", "")
        paper = paper.replace("]", "") 
        paper = ' '.join(paper.split())
        print(paper)
        user_input = input("check for this passage:")
        while user_input != '1':
            paper = remain_papers[idx]
            idx = idx + 1
            paper = _strip_newlines(paper)
            paper = ' '.join(paper.split())
            paper = paper.replace("—", "")
            paper = paper.replace("*", "")
            paper = paper.replace("^", "")
            paper = paper.replace("_", "")
            paper = paper.replace(" n't", "n't")
            paper = paper.replace("( ", "(")
            paper = paper.replace(" )", ")")
            paper = paper.replace("''", "\"")
            paper = paper.replace("`` ", "\"")
            paper = paper.replace(" ``", "\"")
            paper = paper.replace("``", "\"")
            paper = paper.replace("“ ", "\"")
            paper = paper.replace(" ”", "\"")
            paper = paper.replace("’", "\'")
            paper = paper.replace("‘ ", "\'")
            paper = paper.replace(" ' ", "\'")
            paper = paper.replace("Media playback is not supported on this device", "")
            paper = paper.replace("[", "")
            paper = paper.replace("]", "") 
            paper = ' '.join(paper.split())
            print(paper)
            user_input = input("check for this passage:")
        print(idx)
        passages.append(paper)
        
    return passages

def write_df_to_json(df: pd.DataFrame, path_to_json: str):
    """
    This function writes pandas dataframes into a compatible json format
    to be used by hf_token_classification.py
    """
    origin_list = df["text"].values.tolist()
    data_list = []
    for i in tqdm(range(len(origin_list)), total=len(origin_list)):
        origin = origin_list[i]
        data_list.append(
            {"text": origin}
        )
    with open(path_to_json, "w") as f:
        f.write(json.dumps(data_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument('--dataset_file', type=str, default="")
    parser.add_argument('--remain_file', type=str, default="")
    args = parser.parse_args()
    
    dataset = load_dataset("json" , data_files=args.dataset_file, split="train")
    remain_dataset = load_dataset("json" , data_files=args.remain_file, split="train")
    passages = check_passages(dataset["text"],remain_dataset["text"])
    dataset=pd.DataFrame(
        {"text": passages}
    )

    write_df_to_json(dataset,args.output_file)

