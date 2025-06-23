'''get LLR/KeyBERT keywords for each generation'''

import json
import os
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import nltk

# download stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords

data_dir = "../edu_data"
first_gen_file = "first_gen.json"
second_gen_file = "second_gen.json"

# load first-generation texts
with open(os.path.join(data_dir, first_gen_file), "r", encoding="utf-8") as f:
    first_gen_texts = json.load(f)

# load second-generation texts
with open(os.path.join(data_dir, second_gen_file), "r", encoding="utf-8") as f:
    second_gen_texts = json.load(f)

# combine all texts and labels
all_texts = first_gen_texts + second_gen_texts
labels = [0]*len(first_gen_texts) + [1]*len(second_gen_texts)

# get stopwords and vectorize
stop_words = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stop_words, min_df=2)
x = vectorizer.fit_transform(all_texts)
vocab = np.array(vectorizer.get_feature_names_out())

# split by group
x_first = x[:len(first_gen_texts), :]
x_second = x[len(first_gen_texts):, :]

first_counts = np.array(x_first.sum(axis=0)).flatten()
second_counts = np.array(x_second.sum(axis=0)).flatten()

def llr(k11, k12, k21, k22):
    # log-likelihood ratio for 2x2 table
    def safe_log(x):
        return np.log(x) if x > 0 else 0
    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    total = row1 + row2

    e11 = row1 * col1 / total if total else 0
    e12 = row1 * col2 / total if total else 0
    e21 = row2 * col1 / total if total else 0
    e22 = row2 * col2 / total if total else 0

    terms = []
    for k, e in zip([k11, k12, k21, k22], [e11, e12, e21, e22]):
        terms.append(k * safe_log(k / e) if k > 0 and e > 0 else 0)
    return 2 * sum(terms)

llr_scores = []
for i, word in enumerate(vocab):
    k11 = first_counts[i]
    k12 = second_counts[i]
    k21 = first_counts.sum() - k11
    k22 = second_counts.sum() - k12
    score = llr(k11, k12, k21, k22)
    llr_scores.append((word, score, k11, k12))

llr_scores = sorted(llr_scores, key=lambda x: -abs(x[1]))

first_unique = [w for w in llr_scores if w[2] > w[3]][:20]
second_unique = [w for w in llr_scores if w[3] > w[2]][:20]

print("top words unique to first-generation immigrants:")
for word, score, k11, k12 in first_unique:
    print(f"{word}: llr={score:.2f}, first_gen={k11}, second_gen={k12}")

print("\ntop words unique to second-generation immigrants:")
for word, score, k11, k12 in second_unique:
    print(f"{word}: llr={score:.2f}, first_gen={k11}, second_gen={k12}")

kw_model = KeyBERT()
first_keywords = kw_model.extract_keywords(" ".join(first_gen_texts), top_n=20)
second_keywords = kw_model.extract_keywords(" ".join(second_gen_texts), top_n=20)

print("\nKeyBERT keywords for first-generation:")
for kw, score in first_keywords:
    print(f"{kw}: {score:.2f}")

print("\nKeyBERT keywords for second-generation:")
for kw, score in second_keywords:
    print(f"{kw}: {score:.2f}")
