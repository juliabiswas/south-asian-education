import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency
import numpy as np

from util import print_chi2_results

RESULTS_DIR = "../results"

def load_data():
    first_gen_sentiment = pd.read_csv(os.path.join(RESULTS_DIR, "first_gen_sentiment.csv"))
    second_gen_sentiment = pd.read_csv(os.path.join(RESULTS_DIR, "second_gen_sentiment.csv"))
    topic_summary_first = pd.read_csv(os.path.join(RESULTS_DIR, "topic_summary_first_gen.csv"))
    topic_summary_second = pd.read_csv(os.path.join(RESULTS_DIR, "topic_summary_second_gen.csv"))
    topic_summary_first = topic_summary_first[topic_summary_first['topic number'] != -1]
    topic_summary_second = topic_summary_second[topic_summary_second['topic number'] != -1]
    return first_gen_sentiment, second_gen_sentiment, topic_summary_first, topic_summary_second

def map_topic_to_theme(themes):
    topic_to_theme = {}
    for theme, topics in themes.items():
        for topic in topics:
            topic_to_theme[int(topic)] = theme
    return topic_to_theme

def get_theme_doc_counts(topic_summary, topic_to_theme):
    theme_doc_counts = defaultdict(int)
    for _, row in topic_summary.iterrows():
        topic = int(row['topic number'])
        n_docs = int(row['number of documents'])
        theme = topic_to_theme.get(topic)
        if theme:
            theme_doc_counts[theme] += n_docs
    return theme_doc_counts

def plot_theme_distribution_combined(first_gen_theme_counts, second_gen_theme_counts, total_first, total_second, results_dir):
    all_themes = sorted(set(first_gen_theme_counts.keys()) | set(second_gen_theme_counts.keys()))

    data = []
    for theme in all_themes:
        fg_count = first_gen_theme_counts.get(theme, 0)
        sg_count = second_gen_theme_counts.get(theme, 0)
        data.append({
            "Theme": theme,
            "First Generation": fg_count / total_first if total_first else 0,
            "Second Generation": sg_count / total_second if total_second else 0
        })
    df = pd.DataFrame(data)
    df = df.melt(id_vars="Theme", var_name="Generation", value_name="Proportion")
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="Theme", y="Proportion", hue="Generation")
    plt.ylabel("Proportion of Documents")
    plt.xlabel("Theme")
    plt.title("Proportion of Documents per Theme (First vs Second Generation)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "theme_distribution_combined.png"))
    plt.close()

def get_theme_emotions(sentiment_df, topic_to_theme, topic_summary):
    sentiment_df = sentiment_df.copy()
    sentiment_df['theme'] = sentiment_df['topic'].map(topic_to_theme)
    topic_summary = topic_summary.copy()
    merged = sentiment_df.merge(
        topic_summary[['topic number', 'number of documents']],
        left_on='topic', right_on='topic number', how='left'
    )
    merged['number of documents'] = merged['number of documents'].fillna(1).astype(int)
    theme_emotion_counts = defaultdict(Counter)
    theme_total_docs = defaultdict(int)
    for _, row in merged.iterrows():
        theme = row['theme']
        emotion = row['first meaningful']
        n_docs = row['number of documents']
        if pd.notnull(theme):
            theme_emotion_counts[theme][emotion] += n_docs
            theme_total_docs[theme] += n_docs
    theme_emotion_labels = {}
    for theme, emotion_counts in theme_emotion_counts.items():
        total = theme_total_docs[theme]
        if total == 0:
            theme_emotion_labels[theme] = "n/a"
            continue
        for emotion, count in emotion_counts.items():
            if count > total / 2:
                theme_emotion_labels[theme] = emotion
                break
        else:
            avg = total / len(emotion_counts)
            above_avg = [emotion for emotion, count in emotion_counts.items() if count > avg]
            if above_avg:
                theme_emotion_labels[theme] = "; ".join(sorted(above_avg))
            else:
                theme_emotion_labels[theme] = "; ".join(sorted(emotion_counts.keys()))
    return theme_emotion_labels, theme_total_docs

def save_theme_csv(all_themes, first_gen_data, second_gen_data, filename, value_type="emotion"):
    rows = []
    for theme in all_themes:
        row = {
            "theme": theme,
            "first_gen": first_gen_data.get(theme, "n/a" if value_type == "emotion" else 0),
            "second_gen": second_gen_data.get(theme, "n/a" if value_type == "emotion" else 0)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)

def main(themes_first_gen, themes_second_gen):
    first_gen_sentiment, second_gen_sentiment, topic_summary_first, topic_summary_second = load_data()
    topic_to_theme_first = map_topic_to_theme(themes_first_gen)
    topic_to_theme_second = map_topic_to_theme(themes_second_gen)

    # get doc counts per theme
    first_gen_theme_counts = get_theme_doc_counts(topic_summary_first, topic_to_theme_first)
    second_gen_theme_counts = get_theme_doc_counts(topic_summary_second, topic_to_theme_second)
    total_first = sum(first_gen_theme_counts.values())
    total_second = sum(second_gen_theme_counts.values())
    
    plot_theme_distribution_combined(
        first_gen_theme_counts,
        second_gen_theme_counts,
        total_first,
        total_second,
        RESULTS_DIR
    )

    # get theme emotions
    first_gen_theme_emotions, first_gen_theme_docs = get_theme_emotions(first_gen_sentiment, topic_to_theme_first, topic_summary_first)
    second_gen_theme_emotions, second_gen_theme_docs = get_theme_emotions(second_gen_sentiment, topic_to_theme_second, topic_summary_second)
    all_themes = sorted(set(list(themes_first_gen.keys()) + list(themes_second_gen.keys())))

    # save emotion csv
    save_theme_csv(all_themes, first_gen_theme_emotions, second_gen_theme_emotions, "theme_emotions_by_generation.csv", value_type="emotion")

    # save doc count csv
    save_theme_csv(all_themes, first_gen_theme_docs, second_gen_theme_docs, "theme_doc_counts_by_generation.csv", value_type="count")

    # chi-square and cramér's v
    counts_df = pd.read_csv(os.path.join(RESULTS_DIR, "theme_doc_counts_by_generation.csv"))
    print_chi2_results(counts_df, description="themes")

if __name__ == "__main__":
    themes_first_gen = {
        "Community": [0, 16, 23, 25, 45],
        "Arts & Sports": [2, 21],
        "Schooling": [9, 13, 20, 29, 36],
        "Higher Education": [3, 4, 5, 8, 22, 27, 30, 38, 39, 48],
        "Immigration": [6, 42],
        "Culture": [7, 12, 18, 40],
        "Religion": [10, 24, 37],
        "Family": [11, 14, 15],
        "Finances": [17],
        "Violence": [19, 47],
        "Career": [26, 41, 46],
        "Government": [31, 33, 43],
        "Place": [1, 28, 32, 35, 44],
    }
    themes_second_gen = {
        "Community": [17, 34, 36, 41],
        "Arts & Sports": [16, 19],
        "Social Life": [7, 10, 11, 35, 47],
        "Schooling": [2, 14, 22, 25, 27, 28, 29, 30, 31, 40, 46],
        "Higher Education": [1, 24, 33, 39],
        "Religion": [15, 44],
        "Family": [0, 13, 20, 45],
        "Career": [9, 21, 37],
        "Place": [5, 38, 42],
        "Identity": [4, 6, 12, 18, 23, 48],
        "Responsibility": [8],
        "Activism": [26, 32, 43]
    }
    main(themes_first_gen, themes_second_gen)
