import json
from nltk.tokenize import sent_tokenize,word_tokenize
from difflib import SequenceMatcher
parts = ['train','test','validation']
for part in parts:
    with open("dataset/ood_gpt-4o-mini/"+part+"_data.json", "r") as fin:
        data = json.load(fin)
    
    for item in data:
        s = SequenceMatcher(None, item['text'],word_tokenize(item['raw_text'])).get_matching_blocks()
        for block in s:
            item['label'][block.a : (block.a + block.size)] = [0] * block.size

    with open("dataset/token_gpt-4o-mini/"+part+"_data.json", "w") as f:
        f.write(json.dumps(data))