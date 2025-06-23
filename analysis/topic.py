from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

import string
import json
import re
import contractions
from nltk.corpus import stopwords, words as nltk_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

indir = "../edu_data/"
outdir = "../results/"
    
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
stopwords.update({"yeah", "yes", "absolutely", "sure", "no", "nah", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "oral", "interview", "lbk", "the", "th", "thousand", "los", "tom", "hav", "huh", "ummhmm", "los", "si", "tag", "yog", "ce", "ntwad", "thaum", "mmmhmm", "porque", "como", "que", "del", "solo", "tus", "yah", "mhmm", "yup", "mhm", "todo", "uh", "um", "la", "el", "yus", "tia", "lawv", "ntaub", "tsis", "mmmmhmmm", "mmmm", "hmmm", "thing", "really", "say", "pero", "le", "para", "aqui", "maybe" "hmm", "umhm", "ntawd", "thiab", "sgf"})

def process_document(doc):
    doc = re.sub(r'\b\w{1,2}\b', '', doc)
    doc = re.sub(r'\s+', ' ', doc).strip()
    
    doc = doc.lower()
    doc = ''.join([char for char in doc if not char.isdigit() and char not in string.punctuation])
    doc = contractions.fix(doc)
    
    processed_words = []
    for word in word_tokenize(doc):
        if word not in stopwords and lemmatizer.lemmatize(word) not in stopwords:
            processed_words.append(lemmatizer.lemmatize(word))
    
    return ' '.join(processed_words)
    
def visualize_keywords_for_topics(topic_model, outfile):
    """
    visualize the top 10 keywords for topics 0 - 9 with their relevance scores
    """
    colors = sns.color_palette("tab10", 10)
    
    for topic_num in range(10):
        topic_keywords = topic_model.get_topic(topic_num)
        if topic_keywords:
            words, scores = zip(*topic_keywords[:10])

            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(scores), y=list(words), color=colors[topic_num])
            plt.title(f'Top Keywords for Topic {topic_num}')
            plt.xlabel('Relevance Score')
            plt.ylabel('Keyword')
            plt.tight_layout()

            plt.savefig(f"{outfile}_keywords_topic_{topic_num}.png")
            plt.close()

def main():
    infiles = ["first_gen", "second_gen"]
    
    for infile in infiles:
        print(f"processing {infile}...")
        with open(indir + infile + ".json", "r", encoding="utf-8") as f:
            original_documents = json.load(f)
            documents = [process_document(doc) for doc in original_documents]
        
        outfile = outdir + f"processed_{infile}.txt"
        with open(outfile, 'w') as file:
            for doc in documents:
                file.write(doc + '\n')
        print(f"saved processed documents to {outfile}")
    
        vectorizer = CountVectorizer(min_df=5, max_features=10000)
        topic_model = BERTopic(nr_topics=50, min_topic_size=5, vectorizer_model=vectorizer)
        topics, probs = topic_model.fit_transform(documents)
        
        topic_info = topic_model.get_topic_info()
        topic_keywords = {
            topic: [word for word, _ in topic_model.get_topic(topic)[:10]]
            for topic in topic_info['Topic'].values if topic != -1
        }

        topic_counts = topic_info[['Topic', 'Count']]
        print("\ntopic counts (number of documents in each topic):")
        print(topic_counts)

        print("\ntopic info:\n", topic_info)
        
        print("\ntopic keywords (top 10):")
        for topic, words in topic_keywords.items():
            print(f"topic {topic}: {', '.join(words)}")

        outfile = outdir + infile + ".csv"
        pd.DataFrame({"document": original_documents, "topic": topics}).to_csv(outfile, index=False)
        print(f"\ndocuments with their topics saved to {outfile}")
        
        summary_outfile = outdir + f"topic_summary_{infile}.csv"
        pd.DataFrame({
            "topic number": topic_info["Topic"],
            "number of documents": topic_info["Count"],
            "top 10 keywords": [', '.join(topic_keywords.get(topic, [])) for topic in topic_info["Topic"]]
        }).to_csv(summary_outfile, index=False)
        print(f"topic summary saved to {summary_outfile}")
        
        visualize_keywords_for_topics(topic_model, outdir + infile)

if __name__ == "__main__":
    main()
