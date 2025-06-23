'''goemotions ("go_emotions"), goemotions + annotated ("go_annotated"), annotated ("annotated")'''

import pandas as pd
import torch
from transformers import pipeline, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, TrainerCallback
import time
import numpy as np

from util import go_emotion_to_index, go_emotions, save_model, tokenizer, fine_tune, go_eval, split_sentiment_data, eval, load_model

num_labels = 28
    
train, valid, test_sentences, test_labels = split_sentiment_data("sentiment_annotations.csv")
true_labels = [one_hot_label.index(1) for one_hot_label in test_labels]

#go_emotions
print("go_emotions model on test data...")
go_eval("go_emotions", pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", truncation=True, padding=True, top_k=None), test_sentences, true_labels)

#go_annotated
name = f"go_annotated"
print(f"{name} model...")
model = RobertaForSequenceClassification.from_pretrained('SamLowe/roberta-base-go_emotions', num_labels=num_labels)
fine_tune(train, valid, model, num_train_epochs=30)
go_eval(name, pipeline(task="text-classification", model=model, tokenizer=tokenizer, truncation=True, padding=True, top_k=None), test_sentences, true_labels)
save_model(model, name)
    
#annotated
name = f"annotated"
print(f"{name} model...")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
fine_tune(train, valid, model, num_train_epochs=250)
eval(name, pipeline(task="text-classification", model=model, tokenizer=tokenizer, top_k=None, truncation=True, padding=True), test_sentences, true_labels)
save_model(model, name)
    
#domain_annotated
name = f"rdomain_annotated"
print(f"{name} model...")
model = load_model(f"roberta_domain_adapted_10_v2", num_labels)[0]
fine_tune(train, valid, model, num_train_epochs=45)
eval(name, pipeline(task="text-classification", model=model, tokenizer=tokenizer, top_k=None, truncation=True, padding=True), test_sentences, true_labels)
save_model(model, name)
