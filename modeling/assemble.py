'''assemble training set for wave/generation labeling models'''
import json
import os
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, RobertaForSequenceClassification, pipeline
import re

from util import tokenizer, models_dir, save_model, fine_tune, eval, split_data

indir = Path("../../collections")
outdir = "../../labeling_data/"
num_labels = 2
max_tokens = 512

gen_list = []
wave_list = []

punctuation_pattern = r'([.?!])'

for file in indir.iterdir():
    if file.suffix.lower() == ".json":
        print(f"processing {file}...")
        with open(file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        
        for obj in file_data:
            gen = obj.get("generation")
            wave = obj.get("wave")
            all_speech = obj["all speech"]
            
            if gen in [0, 1] and wave in [0, 1]:
                current_chunk = []
                current_token_count = 0

                split_sentences = re.split(punctuation_pattern, all_speech)
                
                for i in range(0, len(split_sentences), 2):
                    sentence = split_sentences[i].strip() + split_sentences[i + 1] + ' ' if i + 1 < len(split_sentences) else ''
                    
                    sentence_token_count = len(tokenizer.encode(sentence))
                    if current_token_count + sentence_token_count <= max_tokens:
                        current_chunk.append(sentence)
                        current_token_count += sentence_token_count
                    else:
                        gen_list.append({'text': "".join(current_chunk), 'label': gen})
                        wave_list.append({'text': "".join(current_chunk), 'label': wave})
                        current_chunk = [sentence]
                        current_token_count = sentence_token_count

                if current_chunk:
                    gen_list.append({'text': "".join(current_chunk), 'label': gen})
                    wave_list.append({'text': "".join(current_chunk), 'label': wave})
                    
            elif gen in [0, 1]:
                current_chunk = []
                current_token_count = 0

                split_sentences = re.split(punctuation_pattern, all_speech)
                
                for i in range(0, len(split_sentences), 2):
                    sentence = split_sentences[i].strip() + split_sentences[i + 1] + ' ' if i + 1 < len(split_sentences) else ''
                    
                    sentence_token_count = len(tokenizer.encode(sentence))
                    if current_token_count + sentence_token_count <= max_tokens:
                        current_chunk.append(sentence)
                        current_token_count += sentence_token_count
                    else:
                        gen_list.append({'text': "".join(current_chunk), 'label': gen})
                        wave_list.append({'text': "".join(current_chunk), 'label': wave})
                        current_chunk = [sentence]
                        current_token_count = sentence_token_count

                if current_chunk:
                    gen_list.append({'text': "".join(current_chunk), 'label': gen})
                    wave_list.append({'text': "".join(current_chunk), 'label': wave})
                    
            elif wave in [0, 1]:
                current_chunk = []
                current_token_count = 0

                split_sentences = re.split(punctuation_pattern, all_speech)
                
                for i in range(0, len(split_sentences), 2):
                    sentence = split_sentences[i].strip() + split_sentences[i + 1] + ' ' if i + 1 < len(split_sentences) else ''
                    
                    sentence_token_count = len(tokenizer.encode(sentence))
                    if current_token_count + sentence_token_count <= max_tokens:
                        current_chunk.append(sentence)
                        current_token_count += sentence_token_count
                    else:
                        gen_list.append({'text': "".join(current_chunk), 'label': gen})
                        wave_list.append({'text': "".join(current_chunk), 'label': wave})
                        current_chunk = [sentence]
                        current_token_count = sentence_token_count

                if current_chunk:
                    gen_list.append({'text': "".join(current_chunk), 'label': gen})
                    wave_list.append({'text': "".join(current_chunk), 'label': wave})
     
gen_data = pd.DataFrame(gen_list, columns=['text', 'label'])
outfile = outdir + "gen.csv"
gen_data.to_csv(outfile, index=False)
print(f"saved gen data to {outfile}")

wave_data = pd.DataFrame(wave_list, columns=['text', 'label'])
outfile = outdir + "wave.csv"
wave_data.to_csv(outfile, index=False)
print(f"saved wave data to {outfile}")
