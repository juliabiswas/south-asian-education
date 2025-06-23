'''emotion analysis across entire generation'''

import json
import pandas as pd
import torch
from transformers import pipeline
import re
from collections import defaultdict, Counter

from util import load_model, tokenizer
from unique import llr

from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

num_labels = 28

model = pipeline(
    task="text-classification",
    model=load_model("go_annotated", num_labels)[0],
    tokenizer=tokenizer,
    top_k=None,
    truncation=True,
    padding=True,
)

indir = "../edu_data/"
outdir = "../results/"
infiles = ["first_gen", "second_gen"]

naturally_occuring_emotions = {"neutral", "approval", "confusion", "disapproval", "curiosity"}

emotion_examples = defaultdict(list)
infile_scores = {infile: defaultdict(float) for infile in infiles}
num_docs = {infile: 0 for infile in infiles}

kw_model = KeyBERT()

for infile in infiles:
    print(f"processing {infile}")
    with open(f"{indir}{infile}.json", "r") as f:
        curr_docs = json.load(f)

    overall_scores = defaultdict(float)
    doc_emotion_map = []
    for doc in curr_docs:
        
        for labels in model(doc):
            for entry in labels:
                overall_scores[entry["label"]] += entry["score"]
                infile_scores[infile][entry["label"]] += entry["score"]
                
            top_emotion = max(labels, key=lambda x: x['score'])['label']
            doc_emotion_map.append((doc, top_emotion))
    
            if len(emotion_examples[top_emotion]) < 10:
                emotion_examples[top_emotion].append(doc)

    num_docs[infile] = len(curr_docs)
    
    first_meaningful_emotion = next((emotion for emotion in sorted(overall_scores, key=lambda k: overall_scores[k], reverse=True) if emotion not in naturally_occuring_emotions), None)
    
    print(f"first meaningful for {infile}: {first_meaningful_emotion}")

    if first_meaningful_emotion is not None:
        docs_first_meaningful = [doc for doc, emo in doc_emotion_map if emo == first_meaningful_emotion]
        docs_other = [doc for doc, emo in doc_emotion_map if emo != first_meaningful_emotion]

        keywords_first_meaningful = kw_model.extract_keywords(" ".join(docs_first_meaningful), top_n=50, stop_words='english')
        keywords_other = kw_model.extract_keywords(" ".join(docs_other), top_n=50, stop_words='english')

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs_first_meaningful + docs_other)
        vocab = np.array(vectorizer.get_feature_names_out())
        n_most = len(docs_first_meaningful)
        n_other = len(docs_other)

        counts_most = np.asarray(X[:n_most].sum(axis=0)).flatten()
        counts_other = np.asarray(X[n_most:].sum(axis=0)).flatten()

        llr_scores = []
        for idx, word in enumerate(vocab):
            a = counts_most[idx]
            b = counts_other[idx]
            c = n_most - a
            d = n_other - b
            score = llr(a, b, c, d)
            llr_scores.append((word, score))

        llr_scores.sort(key=lambda x: x[1], reverse=True)
        top_llr_keywords = llr_scores[:20]

        print(f"top KeyBERT keywords for '{first_meaningful_emotion}' in {infile}:")
        print([kw for kw, _ in keywords_first_meaningful[:10]])
        print(f"top LLR keywords for '{first_meaningful_emotion}' in {infile}:")
        print([kw for kw, _ in top_llr_keywords[:10]])

        pd.DataFrame({
            "keyword": [kw for kw, _ in top_llr_keywords],
            "llr_score": [score for _, score in top_llr_keywords],
            "infile": infile,
            "emotion": first_meaningful_emotion
        }).to_csv(f"{outdir}{infile}_llr_keywords.csv", index=False)

        pd.DataFrame({
            "keyword": [kw for kw, _ in keywords_first_meaningful],
            "keybert_score": [score for _, score in keywords_first_meaningful],
            "infile": infile,
            "emotion": first_meaningful_emotion
        }).to_csv(f"{outdir}{infile}_keybert_keywords.csv", index=False)
    else:
        print(f"no meaningful emotion found for {infile}. skipping keyword analysis.")

summary_results = []
for infile in infiles:
    avg_scores = {emotion: (infile_scores[infile][emotion] / num_docs[infile]) if num_docs[infile] > 0 else 0 for emotion in infile_scores[infile]}
    avg_scores["infile"] = infile
    summary_results.append(avg_scores)

pd.DataFrame(summary_results).to_csv(f"{outdir}emotion_summary.csv", index=False)

pd.DataFrame(
    [{"emotion": key, "example": example}
     for key, examples in emotion_examples.items()
     for example in examples]
).to_csv(f"{outdir}emotion_examples.csv", index=False)
