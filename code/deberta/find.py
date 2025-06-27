import json
from nltk.tokenize import sent_tokenize,word_tokenize
from difflib import SequenceMatcher
with open("dataset/final/test_data.json", "r") as fin:
    data = json.load(fin)
    
for item in data:
    index = item['raw_text'].find('flow fields')
    print(item['raw_text'])
    if index > 0:
        print(item['raw_text'][index-100:])
        breakpoint()