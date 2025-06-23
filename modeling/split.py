'''select a random subset of 100 interviewee turns of speech from across the transcripts & across the topics to label with emotion'''

import pandas as pd
import numpy as np

indir = "../results/"

first_gen = pd.read_csv(indir + 'first_gen.csv')
second_gen = pd.read_csv(indir + 'second_gen.csv')

data = pd.concat([first_gen, second_gen], ignore_index=True)

subset = data.sample(n=100, random_state=42)[['document']].copy()
subset['label'] = np.nan

subset.to_csv('sentiment_annotations.csv', index=False)
