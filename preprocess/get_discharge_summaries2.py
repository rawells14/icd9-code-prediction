"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import csv

from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm

import pandas as pd

from caml-mimic.constants import MIMIC_3_DIR

#retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

def write_discharge_summaries(out_file):
    notes_file = '%s/NOTEEVENTS.csv.gz' % (MIMIC_3_DIR)
    print("processing notes file")
    df_notes = pd.read(notes_file)
    total = len(df_notes.index)
    with open(out_file, 'w') as outfile:
        print("writing to %s" % (out_file))
        outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
        with tqdm(total=total) as pbar:
            for index, row in tqdm(df_notes.iterrows()):
                pbar.update(1)
                category = row['Category']
                if category == "Discharge summary":
                    note = row['TEXT']
                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([row['SUBJECT_ID'], row['HADM_ID'], row['CHARTTIME'], text]) + '\n')
    return out_file


