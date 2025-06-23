'''label generations'''

import json
from pathlib import Path
from util import load_model, tokenizer
from transformers import pipeline
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

indir = Path("../../collections")
outdir = "../final_data/"

gen_model = pipeline(
    task="text-classification",
    model=load_model("generation", 2)[0],
    tokenizer=tokenizer,
    top_k=None,
    truncation=True,
    padding=True
)

first_gen = []
second_gen = []
first_gen_count = 0
second_gen_count = 0

for file in indir.iterdir():
    if file.suffix.lower() == ".json":
        print(f"processing {file}...")
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                gen = obj.get("generation")
                all_speech = obj.get("all speech")
                interviewee_speech = obj.get("interviewee speech")
                
                if gen not in [0, 1]:
                    label = max(gen_model(all_speech)[0], key=lambda x: x['score'])['label']
                    gen = int(label.split("_")[1])

                if gen == 0:
                    first_gen_count += 1
                    first_gen.extend(interviewee_speech)
                else:  # gen == 1
                    second_gen_count += 1
                    second_gen.extend(interviewee_speech)

with open(outdir + 'first_gen.json', 'w') as file:
    json.dump(first_gen, file)
with open(outdir + 'second_gen.json', 'w') as file:
    json.dump(second_gen, file)

print(f"{first_gen_count} first-gen transcripts ({len(first_gen)} documents)\n{second_gen_count} second-gen transcripts ({len(second_gen)}) documents.")
