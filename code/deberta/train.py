import argparse
import gc
import json
import logging
import os
import random
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import List

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent


from hf_token_classification import main as hf_token_classification

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
This script performs span prediction and classification with bert models.
"""


def chunk_tokens_labels(df: pd.DataFrame, max_lengths: List[int]):
    """
    This function chunks tokens and their respective labels to
    max_length token length
    """
    index_list = []
    tokens_list = []
    labels_list = []
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base",
        cache_dir="../../cache",
        use_fast=True,
        add_prefix_space=True,
    )
    num = 0
    probs=[1,0.5,0.2]
    for i in range(len(max_lengths)):
        max_length=max_lengths[i]
        prob=probs[i]
        for index, row in tqdm(df.iterrows(), total=len(df)):
            tokenized_inputs = tokenizer(
                row["tokens"],
                # We use this argument because the texts in our dataset are lists
                # of words (with a label for each word).
                is_split_into_words=True,
            )
            idx = -1
            last_truc_word = -1
            word_ids = tokenized_inputs.word_ids()[1:-1]
            if len(tokenized_inputs["input_ids"]) > max_length:
                remaining_tokens = row["tokens"]
                remaining_labels = row["token_label_ids"]
                idx = idx + max_length - 2
                truc_word = word_ids[idx] - last_truc_word -1
                last_truc_word = word_ids[idx] - 1 
                while word_ids[idx] == word_ids[idx-1]:
                    idx = idx - 1
                # While the remaining list is larger than max_length,
                # truncate and append
                while len(remaining_labels) > truc_word:
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
                    if random.random() < prob:
                        index_list.append(index)
                        tokens_list.append(remaining_tokens[:truc_word])
                        labels_list.append(remaining_labels[:truc_word])
                    remaining_tokens = remaining_tokens[truc_word:]
                    remaining_labels = remaining_labels[truc_word:]
                    idx = idx + max_length - 3
                    if idx < len(word_ids):
                        truc_word = word_ids[idx] - last_truc_word -1
                        last_truc_word = word_ids[idx] - 1
                        while word_ids[idx] == word_ids[idx-1]:
                            idx = idx - 1
                    else:
                        remaining_tokens = list(tokens_list[-1]) + list(remaining_tokens)
                        remaining_labels = list(labels_list[-1]) + list(remaining_labels)
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
                            if random.random() < prob:
                                truc_word = word_ids[idx_b] - last_truc_word -1
                                tokens_list[-1] = remaining_tokens[:truc_word]
                                labels_list[-1] = remaining_labels[:truc_word]
                                index_list.append(index)
                                remaining_tokens = remaining_tokens[truc_word:]
                                tokens_list.append(remaining_tokens)
                                remaining_labels = remaining_labels[truc_word:]
                                labels_list.append(remaining_labels)
                            break
                        else:
                            truc_word = word_ids[idx_f] - last_truc_word -1
                            if random.random() < prob:
                                tokens_list[-1] = remaining_tokens[:truc_word]
                                labels_list[-1] = remaining_labels[:truc_word]
                                index_list.append(index)
                                remaining_tokens = remaining_tokens[truc_word:]
                                tokens_list.append(remaining_tokens)
                                remaining_labels = remaining_labels[truc_word:]
                                labels_list.append(remaining_labels)
                            break
            else:
                index_list.append(index)
                tokens_list.append(row["tokens"])
                labels_list.append(row["token_label_ids"])

    return pd.DataFrame(
        {"index": index_list, "tokens": tokens_list, "labels": labels_list}
    )


def write_df_to_json(df: pd.DataFrame, path_to_json: str):
    """
    This function writes pandas dataframes into a compatible json format
    to be used by hf_token_classification.py
    """
    index_list = df["index"].values.tolist()
    tokens_list = df["tokens"].values.tolist()
    labels_list = df["labels"].values.tolist()
    data_list = []
    for i in tqdm(range(len(tokens_list)), total=len(tokens_list)):
        index = index_list[i]
        tokens = tokens_list[i]
        labels = [str(el) for el in labels_list[i]]
        data_list.append(
            {"index": index, "tokens": tokens, "ner_tags": labels}
        )
    with open(path_to_json, "w") as f:
        f.write(json.dumps(data_list))


def prep_data(path_to_file: str, max_length: int, test: bool = False):
    if test == False:
        dataset = "train"
        # lengths=[max_length, max_length // 2, max_length // 4]
        lengths=[max_length]
    else:
        dataset = "test"
        lengths=[max_length]

    logger.info(f"Loading {dataset} dataset from file")
    df = pd.read_json(path_to_file)
    # if df.index.name != "index":
    #     df.set_index("index", inplace=True)

    # the external NER Classification script needs a target column
    # for the test set as well, which is not available.
    # Therefore, we're subsidizing this column with a fake label column
    # Which we're not using anyway, since we're only using the test set
    # for predictions.
    # if "token_label_ids" not in df.columns:
    #     df["token_label_ids"] = df["tokens"].apply(
    #         lambda x: np.zeros(len(x), dtype=int)
    #     )
    # df["tokens"] = df["generated"]
    # df["token_label_ids"] = df["ner_tags"]
    # df = df[["tokens", "token_label_ids"]]
    df["tokens"] = df["text"]
    df["token_label_ids"] = df["label"]
    df = df[["tokens", "token_label_ids"]]

    logger.info(f"Initial {dataset} data length: {len(df)}")
    df = chunk_tokens_labels(df, max_lengths=lengths)
    logger.info(
        f"{dataset} data length after chunking to max {max_length} tokens: {len(df)}"
    )

    return df


def convert_parquet_data_to_json(
    input_folder_path: str,
    input_train_file_name: str,
    input_val_file_name: str,
    input_test_file_name: str,
    max_length: int,
    val_size: float = 0.1,
    output_train_file_name: str = "",
    output_val_file_name: str = "",
    output_test_file_name: str = "",
    seed: int = 0,
):
    """
    This function takes a parquet file with (at least) the text split and the token
    label ids, and converts it to train, validation, and test data.
    Each chunk is saved as a separate json file

    param input_folder_path: path to data dir
    param input_train_file_name: the input train file name
    param input_test_file_name: the input test file name
    param max_length: max token length
    param val_size: validation size as a fraction of the total
    param output_train_file_name: json train file name
    param output_val_file_name: json validation file name
    param output_test_file_name: json test file name

    returns: None
    """

    logger.info("Loading train and test datasets")

    # Loading and prepping train dataset
    train_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_train_file_name),
        max_length=max_length,
        test=False,
    )
    # Loading and prepping val dataset
    val_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_val_file_name),
        max_length=max_length,
        test=False,
    )
    # Loading and prepping test dataset
    test_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_test_file_name),
        max_length=max_length,
        test=True,
    )

    # logger.info("Splitting train data into train and validation splits")
    # Make the kfold object
    #folds = StratifiedKFold(n_splits=10)
    
    # Now make our splits based off of the labels. 
    # We can use `np.zeros()` here since it only works off of indices, we really care about the labels
    #splits = folds.split(np.zeros(datasets["train"].num_rows), datasets["train"]["label"])
    
    # train_df, val_df = train_test_split(
    #     train_df, test_size=val_size, random_state=seed, shuffle=True
    # )

    logger.info(f"Final train size: {len(train_df)}")
    logger.info(f"Final validation size: {len(val_df)}")
    logger.info(f"Final test size: {len(test_df)}")

    logger.info("Writing train df to json...")
    write_df_to_json(
        train_df,
        f"{input_folder_path}/{output_train_file_name}",
    )
    logger.info("Writing val df to json...")
    write_df_to_json(val_df, f"{input_folder_path}/{output_val_file_name}")
    logger.info("Writing test df to json...")
    write_df_to_json(
        test_df,
        f"{input_folder_path}/{output_test_file_name}",
    )

def split_sentence(sentence):
    import re
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(sentence)
    return words

def _get_most_common_tag(tags):
    """most_common_tag is a tuple: (tag, times)"""
    from collections import Counter

    tag_counts = Counter(tags)
    most_common_tag = tag_counts.most_common(1)[0][0]

    return most_common_tag

def get_sent_label(text, label):
    import nltk
    text = ' '.join(text)
    sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_separator.tokenize(text)
    
    offset = 0
    sent_label = []
    start_word_idx = 0
    for sent in sents:
        start = text[offset:].find(sent) + offset
        end = start + len(sent)
        offset = end
        
        sent_words = split_sentence(text[start:end])
        true_sent_words = []
        for sent_word in sent_words:
            if sent_word != ' ':
                true_sent_words.append(sent_word)
        end_word_idx = start_word_idx + len(true_sent_words)
        tags = label[start_word_idx:end_word_idx]
        if len(tags) < 1:
            continue
        start_word_idx = end_word_idx
        most_common_tag = _get_most_common_tag(tags)
        sent_label.append(most_common_tag)
    
    if len(sent_label) == 0:
        print("empty sent label list")
    return sent_label

def _get_precision_recall_acc_macrof1(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    print("Accuracy: {:.1f}".format(accuracy*100))
    print("Macro F1 Score: {:.1f}".format(macro_f1*100))

    precision = precision_score(true_labels, pred_labels, average=None)
    recall = recall_score(true_labels, pred_labels, average=None)
    print("Precision/Recall per class: ")
    precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
    print(precision_recall)

    result = {"precision":precision, "recall":recall, "accuracy":accuracy, "macro_f1":macro_f1}
    return result

def convert_preds_to_original_format(
    path_to_test_data: str = "",
    path_to_test_preds: str = "",
    path_to_final_output: str = "",
):
    """
    This function takes the chunked preds and groups them into the original format
    """

    orig_test_data = pd.read_json(path_to_test_data)

    with open(path_to_test_preds, "r") as f:
        test_preds = json.load(f)
            

    test_preds_df = pd.DataFrame(test_preds).groupby(by="index").agg(list)

    test_preds_df["preds"] = test_preds_df["predictions"].apply(
        lambda x: sum(x, [])
    )
    
    texts = []
    true_labels = []
    pred_labels = []
    for index, row in test_preds_df.iterrows():
        texts.append(orig_test_data.loc[index, "text"])
        true_labels.append(orig_test_data.loc[index, "label"])
        pred_labels.append(row["preds"])
        # if len(row["preds"]) > len(orig_test_data.loc[index, "generated"]):
        if len(row["preds"]) > len(orig_test_data.loc[index, "text"]):
            print("error_long")
            test_preds_df.loc[index, "preds"] = row["preds"][
                : len(orig_test_data.loc[index, "text"])
            ]

        # elif len(row["preds"]) < len(orig_test_data.loc[index, "generated"]):
        elif len(row["preds"]) < len(orig_test_data.loc[index, "text"]):
            print("error_short")
            test_preds_df.loc[index, "preds"] = row["preds"] + [row["preds"][-1]] * (
                len(orig_test_data.loc[index, "text"]) - len(row["preds"])
            )
    for index, row in test_preds_df.iterrows():
        # assert len(row["preds"]) == len(orig_test_data.loc[index, "generated"])
        assert len(row["preds"]) == len(orig_test_data.loc[index, "text"])
    
    # label_name = "ner_tags"
    label_name = "label"
    # print("-- token-level metric --")
    # metric = pd.DataFrame(test_preds_df["preds"]).join(pd.DataFrame(orig_test_data[label_name]), how="left")
    # for index, row in metric.iterrows():
    #     row["ner_tags"]=[int(y) for y in row[label_name]]
    # metric["f1_score"] = metric.apply(lambda x: f1_score(x[label_name], x["preds"], average="macro"),axis=1,)
    # metric["accuracy_score"] = metric.apply(lambda x: accuracy_score(x[label_name], x["preds"]),axis=1,)
    # metric["human_precision_score"] = metric.apply(lambda x: precision_score(x[label_name], x["preds"], average=None, zero_division=1)[0],axis=1,)
    # metric["ai_precision_score"] = metric.apply(lambda x: precision_score(x[label_name], x["preds"], average=None, zero_division=1)[1]
    #                                             if precision_score(x[label_name], x["preds"], average=None, zero_division=1).size > 1
    #                                             else 0.0 ,axis=1,)
    # metric["human_recall_score"] = metric.apply(lambda x: recall_score(x[label_name], x["preds"], average=None, zero_division=1)[0],axis=1,)
    # metric["ai_recall_score"] = metric.apply(lambda x: recall_score(x[label_name], x["preds"], average=None, zero_division=1)[1]
    #                                             if recall_score(x[label_name], x["preds"], average=None, zero_division=1).size > 1
    #                                             else 0.0 ,axis=1,)
    # print("Accuracy: {:0.4f}%".format(metric['accuracy_score'].mean()*100))
    # print("Macro F1 Score: {:0.4f}%".format(metric['f1_score'].mean()*100))
    # print("Precision/Recall per class: ")
    # print("Human Precision Score: {:0.4f}%".format(metric['human_precision_score'].mean()*100))
    # print("AI Precision Score: {:0.4f}%".format(metric['ai_precision_score'].mean()*100))
    # print("Human Recall Score: {:0.4f}%".format(metric['human_recall_score'].mean()*100))
    # print("AI Recall Score: {:0.4f}%".format(metric['ai_recall_score'].mean()*100))
    
    print("-- sentence-level metric --")
    true_sent_labels = []
    pred_sent_labels = []
    i = 0
    for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
        true_sent_label = get_sent_label(text, true_label)
        pred_sent_label = get_sent_label(text, pred_label)
        true_sent_labels.extend(true_sent_label)
        pred_sent_labels.extend(pred_sent_label)
        # print(i)
        i= i + 1
    result = _get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels)
    breakpoint()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition data prep")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config_baseline.yml",
    )
    args = parser.parse_args()

    # loading config params
    project_root: Path = get_project_root()
    with open(str(project_root / "config" / args.config_file)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_data_folder = str(project_root / params["data"]["path_to_data"])

    input_train_file_name = params["data"]["train_file"]
    input_val_file_name = params["data"]["validation_file"]
    input_test_file_name = params["data"]["test_file"]
    output_train_file_name = params["data"]["train_file_name"]
    output_val_file_name = params["data"]["validation_file_name"]
    output_test_file_name = params["data"]["test_file_name"]

    convert_parquet_data_to_json(
        input_folder_path=path_to_data_folder,
        input_train_file_name=input_train_file_name,
        input_val_file_name=input_val_file_name,
        input_test_file_name=input_test_file_name,
        max_length=params["bert"]["MAX_LENGTH"],
        val_size=params["environment"]["val_size"],
        output_train_file_name=output_train_file_name,
        output_val_file_name=output_val_file_name,
        output_test_file_name=output_test_file_name,
        seed=params["environment"]["SEED"],
    )
    # create hf_token_classification.py config file
    config_dict = {
        "train_file": f"{path_to_data_folder}/{output_train_file_name}",
        "validation_file": f"{path_to_data_folder}/{output_val_file_name}",
        "test_file": f"{path_to_data_folder}/{output_test_file_name}",
        "output_dir": f"{path_to_data_folder}/{params['bert']['output_dir']}",
        "model_name_or_path": params["bert"]["model"],
        "cache_dir": "../../cache",
        "num_train_epochs": params["bert"]["num_train_epochs"],
        "per_device_train_batch_size": params["bert"]["per_device_train_batch_size"],
        "per_device_eval_batch_size": params["bert"]["per_device_eval_batch_size"],
        "save_steps": params["bert"]["save_steps"],
        "overwrite_output_dir": params["bert"]["overwrite_output_dir"],
        "seed": params["environment"]["SEED"],
        "do_train": params["bert"]["do_train"],
        "report_to": params["bert"]["report_to"],
        "do_eval": params["bert"]["do_eval"],
        "do_predict": params["bert"]["do_predict"],
        "preprocessing_num_workers": params["bert"]["preprocessing_num_workers"],
        "eval_accumulation_steps": params["bert"]["eval_accumulation_steps"],
        "log_level": params["bert"]["log_level"],
        # "fp16" : True,
        # "fp16_full_eval" : True,
        "evaluation_strategy" : 'epoch',
        "weight_decay" : 0,
        "warmup_ratio" : 0.2,
        "learning_rate" : 5e-5,
    }

    # save hf_token_classification.py config file
    hf_config_file_path = str(project_root / "config/config_huggingface.json")
    with open(hf_config_file_path, "w") as f:
        json.dump(config_dict, f, indent=4)
        
    hf_token_classification(json_config_file_path=hf_config_file_path)

    path_to_test_data = str(
        project_root
        / f'{params["data"]["path_to_data"]}/{input_test_file_name}'
    )
    path_to_test_preds = str(
        project_root
        / f'{params["data"]["path_to_data"]}/{params["bert"]["output_dir"]}/test_predictions.json'
    )
    path_to_final_output = str(
        project_root
        / f'{params["data"]["path_to_data"]}/preds_{params["bert"]["model"].split("/")[0]}.parquet'
    )

    convert_preds_to_original_format(
        path_to_test_data=path_to_test_data,
        path_to_test_preds=path_to_test_preds,
        path_to_final_output=path_to_final_output,
    )
