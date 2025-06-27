import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

def experiment(args):
    # load data
    with open("./dataset/output_dev/deberta_token/test_predictions.json", "r") as fin:
        data_pred = json.load(fin)
    with open("./dataset/token_mod/data_gen_content_test.json", "r") as fin:
        data_true = json.load(fin)
    # eval criterions
    for idx in tqdm.tqdm(range(len(data_true)), desc=f"Computing criterion train threshold"):
        print(' '.join(data_true[idx]["tokens"][1667:1754]))
        print(data_true[idx]["ner_tags"][1667:1754])
        print(data_pred[idx]["predictions"][1667:1754])
        for i in range(1667,1754):
            if data_pred[idx]["predictions"][i]==1:
                print(data_true[idx]["tokens"][i])
        breakpoint()
        s_idx = -1
        e_idx = -1
        for j in range(len(data_true[idx]["ner_tags"])):
            if int(data_true[idx]["ner_tags"][j]) == data_pred[idx]["predictions"][j]:
                e_idx = j
                if s_idx > 0:
                    print(data_true[idx]["tokens"][s_idx:e_idx])
                    print(data_true[idx]["ner_tags"][s_idx:e_idx])
                s_idx = -1
            else:
                if s_idx < 0:
                    s_idx = j
        breakpoint()
    breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)