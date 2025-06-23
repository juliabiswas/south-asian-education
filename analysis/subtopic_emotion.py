# main.py

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

dir = "../results/"
infiles = ["first_gen", "second_gen"]

naturally_occuring_emotions = {"neutral", "approval", "confusion", "disapproval", "curiosity"}

emotion_examples = defaultdict(list)
infile_scores = {infile: defaultdict(float) for infile in infiles}
infile_sentence_count = {infile: 0 for infile in infiles}

kw_model = KeyBERT()

for infile in infiles:
    print(f"processing {infile}")
    data = pd.read_csv(dir + infile + ".csv")

    results = []
    keybert_topic_rows = []
    llr_topic_rows = []

    for topic, group in sorted(data.groupby('topic'), key=lambda x: x[0]):
        if topic == -1:  # skip -1 topic (the catch-all topic)
            continue
        print(topic)
        curr_docs = group['document'].tolist()

        overall_scores = defaultdict(float)
        doc_emotion_map = []
        emotion_counts = Counter()

        for doc in curr_docs:
            doc_sentences = len(re.findall(r'[.?!]', doc))
            labels = model(doc)
            if isinstance(labels, list) and labels and isinstance(labels[0], list):
                labels = labels[0]

            for entry in labels:
                overall_scores[entry["label"]] += entry["score"] * doc_sentences
                infile_scores[infile][entry["label"]] += entry["score"] * doc_sentences

            top_emotion = max(labels, key=lambda x: x['score'])['label']
            emotion_counts[top_emotion] += 1
            doc_emotion_map.append((doc, top_emotion))

            if len(emotion_examples[top_emotion]) < 10:
                emotion_examples[top_emotion].append(doc)

        infile_sentence_count[infile] += sum(len(re.findall(r'[.?!]', doc)) for doc in curr_docs)

        meaningful_emotions = [(emo, count) for emo, count in emotion_counts.most_common() if emo not in naturally_occuring_emotions]
        if meaningful_emotions:
            most_common_emotion = meaningful_emotions[0][0]
        else:
            most_common_emotion = emotion_counts.most_common(1)[0][0]

        results.append({
            "topic": topic,
            "first meaningful": most_common_emotion
        })

        keybert_keywords = []
        if most_common_emotion:
            docs_most_common = [doc for doc, emo in doc_emotion_map if emo == most_common_emotion]
            docs_other = [doc for doc, emo in doc_emotion_map if emo != most_common_emotion]

            if docs_most_common:
                keywords_most_common = kw_model.extract_keywords(" ".join(docs_most_common), top_n=10, stop_words='english')
                keybert_keywords = [kw for kw, _ in keywords_most_common]

            keybert_topic_rows.append({
                "infile": infile,
                "topic": topic,
                "emotion": most_common_emotion,
                "keywords": keybert_keywords
            })

            llr_keywords = []
            if docs_most_common and docs_other:
                vectorizer = CountVectorizer(stop_words='english')
                X = vectorizer.fit_transform(docs_most_common + docs_other)
                vocab = np.array(vectorizer.get_feature_names_out())
                n_most = len(docs_most_common)
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
                llr_keywords = [kw for kw, _ in llr_scores[:10]]

            llr_topic_rows.append({
                "infile": infile,
                "topic": topic,
                "emotion": most_common_emotion,
                "keywords": llr_keywords
            })

    pd.DataFrame(results).to_csv(f"{dir}{infile}_sentiment.csv", index=False)
    if keybert_topic_rows:
        pd.DataFrame(keybert_topic_rows).to_csv(f"{dir}{infile}_subtopics_keybert_keywords.csv", index=False)
    if llr_topic_rows:
        pd.DataFrame(llr_topic_rows).to_csv(f"{dir}{infile}_subtopics_llr_keywords.csv", index=False)
