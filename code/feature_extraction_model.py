import argparse
import tqdm
import json
import os
import numpy as np
import torch
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from huggingface_hub import login

def chunk_tokens(tokens, max_length: int, tokenizer):
    """
    This function chunks tokens and their respective labels to
    max_length token length
    """
    tokens_list = []
    num = 0
    tokenized_inputs = tokenizer(tokens,is_split_into_words=True,)
    idx = -1
    last_truc_word = -1
    word_ids = tokenized_inputs.word_ids()[1:-1]
    if len(tokenized_inputs["input_ids"]) > max_length:
        remaining_tokens = tokens
        idx = idx + max_length - 2
        truc_word = word_ids[idx] - last_truc_word -1
        last_truc_word = word_ids[idx] - 1 
        while word_ids[idx] == word_ids[idx-1]:
            idx = idx - 1

        while len(remaining_tokens) > truc_word:
            if truc_word == 0:
                num = num + 1
                print(num)
                while word_ids[idx] == word_ids[idx+1]:
                    idx = idx + 1
                    if idx > len(word_ids) - 2:
                        break
                idx = idx + 1
                if idx > len(word_ids) - 1:
                    break
                last_truc_word = word_ids[idx] - 1
                truc_word = 1
            tokens_list.append(remaining_tokens[:truc_word])
            remaining_tokens = remaining_tokens[truc_word:]
            idx = idx + max_length - 3
            if idx < len(word_ids):
                truc_word = word_ids[idx] - last_truc_word -1
                last_truc_word = word_ids[idx] - 1
                while word_ids[idx] == word_ids[idx-1]:
                    idx = idx - 1
            else:
                remaining_tokens = list(tokens_list[-1]) + list(remaining_tokens)
                last_truc_word = last_truc_word - len(tokens_list[-1])
                idx = idx + 3 - max_length
                idx = idx - len(tokenizer(tokens_list[-1],is_split_into_words=True,)["input_ids"]) + 2
                idx_m = (idx + len(word_ids) - 1) // 2
                idx_f = idx_m
                idx_b = idx_m
                while word_ids[idx_f] == word_ids[idx_f-1]:
                    idx_f = idx_f - 1
                if (len(word_ids) - 1 - idx_f) > max_length:
                    while word_ids[idx_b] == word_ids[idx_b+1]:
                        idx_b = idx_b + 1
                    idx_b = idx_b + 1
                    truc_word = word_ids[idx_b] - last_truc_word -1
                    tokens_list[-1] = remaining_tokens[:truc_word]
                    remaining_tokens = remaining_tokens[truc_word:]
                    tokens_list.append(remaining_tokens)
                    break
                else:
                    truc_word = word_ids[idx_f] - last_truc_word -1
                    tokens_list[-1] = remaining_tokens[:truc_word]
                    remaining_tokens = remaining_tokens[truc_word:]
                    tokens_list.append(remaining_tokens)
                    break
    else:
        tokens_list.append(tokens)

    return tokens_list

def get_likelihood(logits, labels, word_ids):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_prob = []
    metrics = []
    previous_word_idx = None
    for i in range(len(word_ids)):
        word_idx = word_ids[i]
        if word_idx != previous_word_idx:
            if len(metrics) > 0:
                log_prob.append(sum(metrics) / len(metrics))
            metrics = []
            metrics.append(log_likelihood[i].item())
        metrics.append(log_likelihood[i].item())
        previous_word_idx = word_idx
    log_prob.append(sum(metrics) / len(metrics))
    return log_prob

def get_rank(logits, labels, word_ids):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 
    rank = []
    metrics = []
    previous_word_idx = None
    for i in range(len(word_ids)):
        word_idx = word_ids[i]
        if word_idx != previous_word_idx:
            if len(metrics) > 0:
                rank.append(sum(metrics) / len(metrics))
            metrics = []
            metrics.append(ranks[i].item())
        metrics.append(ranks[i].item())
        previous_word_idx = word_idx
    rank.append(sum(metrics) / len(metrics))
    return rank

def get_logrank(logits, labels, word_ids):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  
    ranks = torch.log(ranks)
    log_rank = []
    metrics = []
    previous_word_idx = None
    for i in range(len(word_ids)):
        word_idx = word_ids[i]
        if word_idx != previous_word_idx:
            if len(metrics) > 0:
                log_rank.append(sum(metrics) / len(metrics))
            metrics = []
            metrics.append(ranks[i].item())
        metrics.append(ranks[i].item())
        previous_word_idx = word_idx
    log_rank.append(sum(metrics) / len(metrics))
    return log_rank

def get_entropy(logits, labels, word_ids):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    
    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    entropy = entropy.view(-1)
    entropys = []
    metrics = []
    previous_word_idx = None
    for i in range(len(word_ids)):
        word_idx = word_ids[i]
        if word_idx != previous_word_idx:
            if len(metrics) > 0:
                entropys.append(sum(metrics) / len(metrics))
            metrics = []
            metrics.append(entropy[i].item())
        metrics.append(entropy[i].item())
        previous_word_idx = word_idx
    entropys.append(sum(metrics) / len(metrics))
    return entropys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument('--dataset_file', type=str, default="")
    args = parser.parse_args()
    with open("./dataset/gpt-4o_wikipedia/"+data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    model_fullnames = {
            'llama': 'meta-llama/Meta-Llama-3-8B',
            'gpt2-xl': 'gpt2-xl',
            'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
            'gpt-j-6B': 'EleutherAI/gpt-j-6B',
            }
    for model_name in model_fullnames:
        if model_name != 'llama':
            tokenizer = AutoTokenizer.from_pretrained(model_fullnames[model_name],cache_dir="../cache",use_fast=True,add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_fullnames[model_name],cache_dir="../cache",use_fast=True,)
        model = AutoModelForCausalLM.from_pretrained(model_fullnames[model_name],device_map="auto",cache_dir="../cache",resume_download=True)
        model.eval() 
        device = torch.device("cuda")
        if model_name == "gpt2-xl":
            max_length = 1024
        else:
            max_length = 2048

        criterion_fns = {'likelihood': get_likelihood,
                     'rank': get_rank,
                     'logrank': get_logrank,
                     'entropy': get_entropy}

        for idx in tqdm.tqdm(range(len(data))):
            original_text = data[idx]["text"]
            chunk_text = chunk_tokens(original_text, max_length, tokenizer)
            results=[]
            for text in chunk_text:
                tokenized = tokenizer(text, is_split_into_words=True,return_tensors="pt",).to(device)
                if model_name != 'llama':
                    labels = tokenized.input_ids[:, :]
                    word_ids = tokenized.word_ids()[:]
                else:
                    labels = tokenized.input_ids[:, 1:]
                    word_ids = tokenized.word_ids()[1:]
                with torch.no_grad():
                    if model_name != 'llama':
                        logits = model(**tokenized).logits[:, :]
                    else:
                        logits = model(**tokenized).logits[:, :-1]
                i = -1
                for name in criterion_fns:
                    i = i + 1
                    criterion_fn = criterion_fns[name]
                    torch.manual_seed(0)
                    np.random.seed(0)
                    original_crit = criterion_fn(logits, labels, word_ids)
                    assert len(original_crit) == len(text)
                    if len(results) < 4:
                        results.append(original_crit)
                    else:
                        results[i].extend(original_crit)
            assert len(results[0]) == len(data[idx]["text"])
            assert len(results[1]) == len(data[idx]["text"])
            assert len(results[2]) == len(data[idx]["text"])
            assert len(results[3]) == len(data[idx]["text"])
            results={'log_prob':results[0],
                         'rank':results[1],
                         'log_rank':results[2],
                         'entropy':results[3]}
            data[idx][f"{model_name}"]=results
    with open(args.output_file, "w") as f:
        f.write(json.dumps(data))