import csv

import pandas as pd


def load_data():
    CHUNKS_FILE = 'data/quranic-treebank-0.4-chunks.tsv'
    return pd.read_csv(CHUNKS_FILE, sep='\t', quoting=csv.QUOTE_NONE)
