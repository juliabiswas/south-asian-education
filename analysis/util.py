import time
import torch
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

models_dir = "../../models/"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

go_emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

go_emotion_to_index = {emotion: idx for idx, emotion in enumerate(go_emotions)}

def load_model(model_name, num_labels):
    model_path = f"{models_dir}{model_name}"
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    print(f"model loaded from {model_path}")
    return model, tokenizer

def save_model(model, model_name):
    model_path = f"{models_dir}{model_name}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"model saved to {model_path}")
    
def fine_tune(train_dataset, valid_dataset, model, num_train_epochs=3, warmup_steps=500, weight_decay=1e-3, logging_steps=10, learning_rate=3e-5):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=None,
            logging_dir=None,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="no",
            log_level='info',
            disable_tqdm=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"fine-tuning took {end_time - start_time:.2f} seconds.")

def label_eval(model_name, model, test_sentences, true_labels):
    predicted_labels = []
    for output in model(test_sentences):
        label = max(output, key=lambda x: x['score'])['label']  # e.g. "LABEL_0"
        predicted_labels.append(int(label.split("_")[1]))

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    print(f"macro f1 score for {model_name}: {f1_score(true_labels, predicted_labels, average='macro')}")
    print(f"accuracy for {model_name}: {accuracy_score(true_labels, predicted_labels)}")
    print(f"true positives: {tp}")
    print(f"false positives: {fp}")
    print(f"false negatives: {fn}")
    print(f"true negatives: {tn}")

    cm = confusion_matrix(true_labels, predicted_labels)
    true_labels_names = ['True First Gen', 'True Second Gen']
    predicted_labels_names = ['Predicted First Gen', 'Predicted Second Gen']

    plt.figure(figsize=(6,6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                     xticklabels=predicted_labels_names, yticklabels=true_labels_names,
                     annot_kws={"size": 16})
    ax.set_title("Generation Confusion Matrix", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(f"{model_name}-confusion_matrix.png")
    plt.close()

def eval(model_name, model, test_sentences, true_labels):
    predicted_labels = []
    for output in model(test_sentences):
        label = max(output, key=lambda x: x['score'])['label']  # e.g. "LABEL_4"
        predicted_labels.append(int(label.split("_")[1]))
    
    print(f"macro f1 score for {model_name}: {f1_score(true_labels, predicted_labels, average='macro')}")
    print(f"accuracy for {model_name}: {accuracy_score(true_labels, predicted_labels)}")
    
def go_eval(model_name, model, test_sentences, true_labels):
    predicted_labels = []
    for output in model(test_sentences):
        label = max(output, key=lambda x: x['score'])['label']
        predicted_labels.append(go_emotion_to_index[label])
    
    print(f"macro f1 score for {model_name}: {f1_score(true_labels, predicted_labels, average='macro')}")
    print(f"accuracy for {model_name}: {accuracy_score(true_labels, predicted_labels)}")
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
        
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def split_data(data):
    '''returns train_dataset, valid_dataset, and test sentences/labels given dataframe with "sentence" and "label" columns'''
    
    # 80% train, 10% valid, 10% test
    train_sentences, valid_test_sentences, train_labels, valid_test_labels = train_test_split(
        data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42
    )
    valid_sentences, test_sentences, valid_labels, test_labels = train_test_split(
        valid_test_sentences, valid_test_labels, test_size=0.5, random_state=42
    )

    return Dataset(tokenizer(train_sentences, truncation=True, padding=True), train_labels), Dataset(tokenizer(valid_sentences, truncation=True, padding=True), valid_labels), test_sentences, test_labels

def split_sentiment_data(filename):
    '''returns train_dataset, valid_dataset, and test sentences/labels'''
    data = pd.read_csv(filename, encoding="mac_roman")
    data = data.dropna(subset=['label'])
    
    one_hot_true_labels = []
    for label in data['label'].tolist():
        label_vector = [0] * len(go_emotion_to_index)
        label_vector[go_emotion_to_index[label.lower().strip()]] = 1
        one_hot_true_labels.append(label_vector)
    
    #80% train, 10% valid, 10% test
    train_sentences, valid_test_sentences, train_labels, valid_test_labels = train_test_split(data['document'].tolist(), one_hot_true_labels, test_size=0.2, random_state=42)
    valid_sentences, test_sentences, valid_labels, test_labels = train_test_split(valid_test_sentences, valid_test_labels, test_size=0.5, random_state=42)
    
    return SentimentDataset(tokenizer(train_sentences, truncation=True, padding=True), train_labels), SentimentDataset(tokenizer(valid_sentences, truncation=True, padding=True), valid_labels), test_sentences, test_labels

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def split_into_chunks(text, max_length=512):
    tokens = tokenizer.encode(text)
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

def print_chi2_results(counts_df):
    data = counts_df[["first_gen", "second_gen"]].values
    chi2, p, dof, expected = chi2_contingency(data.T)
    n = np.sum(data)
    min_dim = min(data.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    print(f"chi-square: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    print(f"cramér's v: {cramers_v:.4f}")
    if p < 0.05:
        print(f"the distribution of {description} is significantly different across groups")
    else:
        print(f"no significant difference in {description} distribution across groups")
