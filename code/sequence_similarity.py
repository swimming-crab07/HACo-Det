import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def experiment():
    with open("news_sim.json", "r") as fin:
        news_sim = json.load(fin)
        print(news_sim)
    with open("paper_sim.json", "r") as fin:
        paper_sim = json.load(fin)
    with open("story_sim.json", "r") as fin:
        story_sim = json.load(fin)
    with open("wiki_sim.json", "r") as fin:
        wiki_sim = json.load(fin)
        
    plt.figure(figsize=(15, 10))  
    sns.kdeplot(news_sim, shade=True, linewidth=0, color='blue', label='Llama3')  
    sns.kdeplot(paper_sim, shade=True, linewidth=0, color='green', label='Mixtral')  
    sns.kdeplot(story_sim, shade=True, linewidth=0, color='red', label='GPT-4o-mini')  
    sns.kdeplot(wiki_sim, shade=True, linewidth=0, color='yellow', label='GPT-4o')  

    font1 ={'weight': 'normal',
            'size':30,
    }
    
    plt.tick_params(labelsize=18)
    plt.xlim(0, 1)
    plt.legend(prop=font1)
    plt.xlabel('Sequence similarity',font1)  
    plt.ylabel('Probability Density',font1)  
    plt.savefig("sequence.pdf",bbox_inches='tight')
    plt.show()  
    breakpoint()

if __name__ == '__main__':

    experiment()