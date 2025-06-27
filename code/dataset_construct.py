from datasets import load_dataset, concatenate_datasets  
import pandas as pd
import torch
from nltk.metrics import edit_distance
from difflib import SequenceMatcher
import json
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np
from tqdm.auto import tqdm
from openai import OpenAI
import argparse
import torch

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _strip_newlines(text):
    return ' '.join(text.split('\n'))

def generate_passages(papers, sentences_min=10, sentences_max=15):
    similarity = []
    annotations = []
    origins = []
    passages = []
    sentences = []
    labels = []
    
    model = "gpt-4o"

    total_tokens = [0,0]
    polish_times = 0 
    for paper in tqdm(papers):
        # print(paper)
        paper = _strip_newlines(paper)
        paper = ' '.join(paper.split())

        paper = paper.replace(" n't", "n't")
        paper = paper.replace("''", "\"")
        paper = paper.replace("`` ", "\"")
        paper = paper.replace("“", "\"")
        paper = paper.replace("”", "\"")
        paper = paper.replace("’", "\'")

        paper = paper.replace("Media playback is not supported on this device ", "")
        paper = paper.replace("[", "")
        paper = paper.replace("]", "") 
        origins.append(paper)
        sentence_end = 0
        sentences = sent_tokenize(paper)
        sen_label = [0] * len(sentences)
        label = []
        passage_token =[]
        sims = []
        annotation_sent = []

        while sentence_end < len(sentences):
            num_sentences = np.random.randint(sentences_min, sentences_max)
            sentence_start = sentence_end + np.random.randint(0, (len(sentences) - sentence_end)//2 + 1)
            sentence_end = sentence_start + num_sentences
            if sentence_end > len(sentences):
                break
            passage = ' '.join(sentences[sentence_start:sentence_end])
            print("raw text:")
            print(passage)
            if passage == '':
                break
            gen_text=""

            messages = [
                {"role": "system", "content": "You are a wikipedia editor."},
                {"role": "user", "content": "polish the following text to make it more human-like, only output the polished version of the text: " + passage + "\nHere is the polished text:"},
            ]

            time=0
            while len(gen_text)<0.5*len(passage):
                gen_text=""
                gen_text = gen_text + openai_generate(messages, model, total_tokens)
                time = time + 1
                if time>10 : break
            if time > 10 : continue
            sim = similar(passage,gen_text)
            sims.append(sim)
            start = sentences[:sentence_start]
            end = sentences[sentence_end:]
            paper = ' '.join(start) + gen_text + ' '.join(end)

            sentences = start + sent_tokenize(gen_text) + end

            sen_label= sen_label[:sentence_start] + [1] * len(sent_tokenize(gen_text)) + sen_label[sentence_end:]
            sentence_end = len(start) + len(sent_tokenize(gen_text))
            annotation_sent.append([sentence_start,sentence_end-1])
            assert len(sen_label) == len(sentences)
            polish_times = polish_times + 1
        annotation = []
        end_label = 0
        mod_sent_idx = 0
        for i in range(len(sentences)):
            words = word_tokenize(sentences[i])
            label = label + [sen_label[i]] * len(words)
            if mod_sent_idx < len(annotation_sent):
                if i == annotation_sent[mod_sent_idx][end_label]:
                    annotation_start = len(passage_token)
                    end_label = 1
            passage_token = passage_token + words
            if mod_sent_idx < len(annotation_sent):
                if i == annotation_sent[mod_sent_idx][end_label]:
                    annotation_end = len(passage_token)
                    annotation.append([annotation_start,annotation_end])
                    end_label = 0
                    mod_sent_idx = mod_sent_idx + 1

            
        assert len(sims) == len(annotation):
        annotations.append(annotation)
        similarity.append(sims)

        assert len(passage_token) == len(label)
        labels.append(label)
        passages.append(passage_token)
    polish_times = polish_times / len(papers)
    print(polish_times)
    return passages,labels,origins,annotations,similarity

def llama_generate(message, max_length, tokenizer, model):
    input_ids = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_length,        
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    # response = outputs[0]
    gen_text = tokenizer.decode(response, skip_special_tokens=True)
    print("generate:")
    if gen_text.find("Here is the polished text:") > -1:
        gen_text = gen_text[len("Here is the polished text:"):]
    gen_text = _strip_newlines(gen_text)
    gen_text = ' '.join(gen_text.split())
    print(gen_text)
    return gen_text

def openai_generate(messages, model, total_tokens):
    OPENAI_API_KEY=""
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        top_p=0.9,
        max_tokens=1024
    )

    gen_text=completion.choices[0].message.content
    print("generate:")
    gen_text = _strip_newlines(gen_text)
    gen_text = ' '.join(gen_text.split())
    print(gen_text)
    total_tokens[0] = total_tokens[0] + completion.usage.prompt_tokens
    total_tokens[1] = total_tokens[1] + completion.usage.completion_tokens
    print(f'{total_tokens[0]} input tokens counted by the OpenAI API.')
    print(f'{total_tokens[1]} output tokens counted by the OpenAI API.')
    return gen_text

def write_df_to_json(df: pd.DataFrame, path_to_json: str):
    """
    This function writes pandas dataframes into a compatible json format
    to be used by hf_token_classification.py
    """
    origin_list = df["origin"].values.tolist()
    generated_list = df["generated"].values.tolist()
    labels_list = df["labels"].values.tolist()
    annotation_list = df["annotation"].values.tolist()
    similarity_list = df["similarity"].values.tolist()
    data_list = []
    for i in tqdm(range(len(origin_list)), total=len(origin_list)):
        origin = origin_list[i]
        generated = generated_list[i]
        labels = [el for el in labels_list[i]]
        annotation = annotation_list[i]
        similarity = similarity_list[i]
        data_list.append(
            {"raw_text": origin, "text": generated, "label": labels, "annotation":annotation, "similarity":similarity}
        )
    with open(path_to_json, "w") as f:
        f.write(json.dumps(data_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument('--dataset_file', type=str, default="")
    args = parser.parse_args()

    dataset = load_dataset("json" , data_files=args.dataset_file, split="train")
    passages,labels,origins,annotations,similarity = generate_passages(dataset['text'])
    dataset=pd.DataFrame(
        {"origin": origins, "generated": passages, "labels": labels, "annotation": annotations, "similarity":similarity}
    )

    write_df_to_json(dataset,args.output_file)

