import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

def experiment(args):
    with open(args.test_file, "r") as fin:
        data_test = json.load(fin)
    with open(args.train_file, "r") as fin:
        data_train = json.load(fin)
    criterion_fns = ['log_prob', 'rank', 'log_rank', 'entropy']
    for name in criterion_fns:
        human_metric = []
        ai_metric = []
        for idx in tqdm.tqdm(range(len(data_train)), desc=f"Computing {name} criterion train threshold"):
            for j in range(len(data_train[idx]["label"])):
                if data_train[idx]["label"][j] == 1:
                    ai_metric.append(data_train[idx]["metric"]["llama"][name][j])
                else:
                    human_metric.append(data_train[idx]["metric"]["llama"][name][j])
        ai_metric = np.array(ai_metric)
        human_metric = np.array(human_metric)
        print(name)
        print(np.mean(ai_metric))
        print(np.mean(human_metric))
        human_metric = []
        ai_metric = []
        for idx in tqdm.tqdm(range(len(data_test)), desc=f"Computing {name} criterion train threshold"):
            for j in range(len(data_test[idx]["label"])):
                if data_test[idx]["label"][j] == 1:
                    ai_metric.append(data_test[idx]["metric"]["llama"][name][j])
                else:
                    human_metric.append(data_test[idx]["metric"]["llama"][name][j])
        ai_metric = np.array(ai_metric)
        human_metric = np.array(human_metric)
        print(np.mean(ai_metric))
        print(np.mean(human_metric))
        breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default="")
    parser.add_argument('--train_file', type=str, default="")
    args = parser.parse_args()

    experiment(args)