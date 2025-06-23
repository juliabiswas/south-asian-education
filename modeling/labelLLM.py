'''training models to label wave & generation for transcripts where they're unknown'''

import json
import os
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, RobertaForSequenceClassification, pipeline
import re

from util import tokenizer, models_dir, save_model, fine_tune, label_eval, split_data, load_model

indir = Path("../../collections")
outdir = "../../labeling_data/"
num_labels = 2

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
            
            if gen in [0, 1]:
                gen_list.append({'text': all_speech, 'label': gen})
            if wave in [0, 1]:
                wave_list.append({'text': all_speech, 'label': wave})
            
gen_data = pd.DataFrame(gen_list, columns=['text', 'label'])
outfile = outdir + "gen.csv"
gen_data.to_csv(outfile, index=False)
print(f"saved gen data to {outfile}")

wave_data = pd.DataFrame(wave_list, columns=['text', 'label'])
outfile = outdir + "wave.csv"
wave_data.to_csv(outfile, index=False)
print(f"saved wave data to {outfile}")

gen_data = pd.read_csv(outdir + "gen.csv")
wave_data = pd.read_csv(outdir + "wave.csv")

print("splitting gen data...")
gen_train, gen_valid, gen_test_sentences, gen_test_labels = split_data(gen_data)

print("splitting wave data...")
wave_train, wave_valid, wave_test_sentences, wave_test_labels = split_data(wave_data)

name = f"generation"
print(f"{name} model...")
model = load_model(f"roberta_domain_adapted_10_v2", num_labels)[0]
fine_tune(gen_train, gen_valid, model, num_train_epochs=21)
save_model(model, name)
label_eval(name, pipeline(task="text-classification", model=model, tokenizer=tokenizer, top_k=None, truncation=True, padding=True), gen_test_sentences, gen_test_labels)

name = f"wave"
print(f"{name} model...")
model = load_model(f"roberta_domain_adapted_10_v2", num_labels)[0]
fine_tune(wave_train, wave_valid, model, num_train_epochs=7)
save_model(model, name)
label_eval(name, pipeline(task="text-classification", model=model, tokenizer=tokenizer, top_k=None, truncation=True, padding=True), wave_test_sentences, wave_test_labels)
