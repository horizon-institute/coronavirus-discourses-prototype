import re as regex
from math import log2
from os.path import isfile, join
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from numpy import average
from datetime import datetime

import argparse


parser = argparse.ArgumentParser(description='Add some integers.')
parser.add_argument('input', help='Input dataset for the analysis (in text format)')
parser.add_argument('config', help='Configuration file for the analysis (in JSON format)')
parser.add_argument('-o', '--output', help='Output file for the analysis (default analysis.xlsx)', required=False)
parser.add_argument("-v", "--verbose", help="Print messages about current status of the analysis", action="store_true")


args = parser.parse_args()

if args.verbose:
    print("Verbosity turned on")

config_file = open(args.config)
config_data = json.load(config_file)
wsize = config_data['window']



skip_intermediate_if_exists = False

filter_on_words = True
save_window = config_data['save-concordance-lines']
filter_words = config_data['tokens']


def lookup_weight(t1_index, t2_index, weight_matrix):
    return weight_matrix[t1_index][t2_index]


def token_is_valid(token):
    for i in range(0, len(filter_patterns)):
        if filter_patterns[i].fullmatch(token) is not None:
            return i
    return False


inputfile = args.input
if args.output:
    if args.output.endswith("xlsx"):
        output_file_excel = args.output
    else:
        output_file_excel = args.output + ".xlsx"
else:
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_file_excel = "analysis_" + date_time + ".xlsx"

intermediate_file = "temp_file.csv"

lemmatizer = WordNetLemmatizer()

cleaned = []
with open(inputfile, "r", encoding="utf8", errors="ignore") as ifile:
    for item in ifile:
        no_tags = BeautifulSoup(item, 'html.parser').get_text()
        no_sc = ''.join(lemmatizer.lemmatize(e) for e in no_tags if e.isalnum() or e.isspace())
        if len(no_sc.strip()) > 0:
            cleaned.append(no_sc)
            done = len(cleaned)
            if args.verbose and done % 1000 == 0:
                print("[CLEANING HTML TAGS] " + str(done) + " items processed")

with open(intermediate_file, 'w', encoding="utf8", errors="ignore") as output_intermediate:
    output_intermediate.writelines(cleaned)

processed = []
lcounter = 0
for line in cleaned:
    sentences = sent_tokenize(line.lower())
    for sentence in sentences:
        processed.extend(word_tokenize(sentence))
    lcounter += 1
    if args.verbose and lcounter % 1000 == 0:
        print("[TOKENISATION] " + str(lcounter) + " items processed")

filter_patterns = []
for word in filter_words:
    p = regex.compile(word)
    filter_patterns.append(p)


tdict = {}
btd = {}
atd = {}
otd = {}


doc_length = len(processed)
if args.verbose:
    print("Document length: " + str(doc_length))
save_collocations = []
for i in range(0, doc_length):
    if args.verbose and i % 10000 == 0:
        progress = i / doc_length
        percentage = "{:.0%}".format(progress)
        print("[ANALYSIS] " + str(i) + "/" + str(doc_length) + " tokens processed (" + str(percentage) + ")")
    t1_index = token_is_valid(processed[i])
    if not t1_index:
        continue
    if processed[i] not in btd:
        btd[processed[i]] = {}
        btd[processed[i]]['collocates'] = {}
    if processed[i] not in atd:
        atd[processed[i]] = {}
        atd[processed[i]]['collocates'] = {}
    if processed[i] not in otd:
        otd[processed[i]] = {}
        otd[processed[i]]['collocates'] = {}

    if processed[i] not in tdict:
        tdict[processed[i]] = 1
    else:
        tdict[processed[i]] += 1

    b_tokens = []
    a_tokens = []
    o_tokens = []
    for j in range(max(0, i - wsize[0]), i - 1):
        b_tokens.append(processed[j])
    for j in range(i + 1, min(i + wsize[1], len(processed))):
        a_tokens.append(processed[j])
    for j in range(max(0, i - wsize[0]), min(i + wsize[1], len(processed))):
        if j != i:
            o_tokens.append(processed[j])
    for a in range(0, len(b_tokens)):
        token = b_tokens[a]
        if not token_is_valid(token):
            continue
        else:
            tdistance = len(b_tokens) - a + 1
            if token not in btd[processed[i]]['collocates']:
                btd[processed[i]]['collocates'][token] = {}
                btd[processed[i]]['collocates'][token]['collocate_frequency'] = 1
                btd[processed[i]]['collocates'][token]['collocate_distance'] = [tdistance]
            else:
                btd[processed[i]]['collocates'][token]['collocate_frequency'] = btd[processed[i]]['collocates'][token]['collocate_frequency'] + 1
                btd[processed[i]]['collocates'][token]['collocate_distance'].append(tdistance)
    for a in range(0, len(a_tokens)):
        token = a_tokens[a]
        t2_index = token_is_valid(token)
        if not t2_index:
            continue
        else:
            tdistance = a + 1
            if save_window:
                window = []
                lbound = i - wsize[0]
                ubound = i + wsize[1] + 1
                for j in range(lbound, ubound):
                    if j < 0 or j > doc_length:
                        window.append("[PADDING]")
                    else:
                        window.append(processed[j])
                window.append(tdistance)
                save_collocations.append(window)

            if token not in atd[processed[i]]['collocates']:
                atd[processed[i]]['collocates'][token] = {}
                atd[processed[i]]['collocates'][token]['collocate_frequency'] = 1
                atd[processed[i]]['collocates'][token]['collocate_distance'] = [tdistance]
            else:
                atd[processed[i]]['collocates'][token]['collocate_frequency'] = atd[processed[i]]['collocates'][token]['collocate_frequency'] + 1
                atd[processed[i]]['collocates'][token]['collocate_distance'].append(tdistance)
    for a in range(0, len(o_tokens)):
        token = o_tokens[a]
        if not token_is_valid(token):
            continue
        else:
            tdist = 1 + abs(a - wsize[0])
            if token not in otd[processed[i]]['collocates']:
                otd[processed[i]]['collocates'][token] = {}
                otd[processed[i]]['collocates'][token]['collocate_frequency'] = 1
                otd[processed[i]]['collocates'][token]['collocate_distance'] = [tdistance]
            else:
                otd[processed[i]]['collocates'][token]['collocate_frequency'] = otd[processed[i]]['collocates'][token]['collocate_frequency'] + 1
                otd[processed[i]]['collocates'][token]['collocate_distance'].append(tdistance)


for token in btd.keys():
    for ctoken in btd[token]['collocates']:
        cfreq = btd[token]['collocates'][ctoken]['collocate_frequency']
        t1_probability = tdict[ctoken] / doc_length
        t2_probability = tdict[token] / doc_length
        joint_probability = cfreq / doc_length
        pmi = (log2(joint_probability / (t1_probability * t2_probability)))
        npmi = pmi / (- log2(joint_probability))
        lmi = cfreq * pmi
        btd[token]['collocates'][ctoken]['collocate_pmi'] = pmi
        btd[token]['collocates'][ctoken]['collocate_npmi'] = npmi
        btd[token]['collocates'][ctoken]['collocate_lmi'] = lmi


for token in atd.keys():
    for ctoken in atd[token]['collocates']:
        cfreq = atd[token]['collocates'][ctoken]['collocate_frequency']
        t1_probability = tdict[ctoken] / doc_length
        t2_probability = tdict[token] / doc_length
        joint_probability = cfreq / doc_length
        pmi = (log2(joint_probability / (t1_probability * t2_probability)))
        npmi = pmi / (- log2(joint_probability))
        lmi = cfreq * pmi
        atd[token]['collocates'][ctoken]['collocate_pmi'] = pmi
        atd[token]['collocates'][ctoken]['collocate_npmi'] = npmi
        atd[token]['collocates'][ctoken]['collocate_lmi'] = lmi


for token in otd.keys():
    for ctoken in otd[token]['collocates']:
        cfreq = otd[token]['collocates'][ctoken]['collocate_frequency']
        t1_probability = tdict[ctoken] / doc_length
        t2_probability = tdict[token] / doc_length
        joint_probability = cfreq / doc_length
        pmi = (log2(joint_probability / (t1_probability * t2_probability)))
        npmi = pmi / (- log2(joint_probability))
        lmi = cfreq * pmi
        otd[token]['collocates'][ctoken]['collocate_pmi'] = pmi
        otd[token]['collocates'][ctoken]['collocate_npmi'] = npmi
        otd[token]['collocates'][ctoken]['collocate_lmi'] = lmi


def format_matrix(mat):
    all = []
    for token in mat.keys():
        for ctoken in mat[token]['collocates']:
            freq = mat[token]['collocates'][ctoken]['collocate_frequency']
            pmi = mat[token]['collocates'][ctoken]['collocate_pmi']
            npmi = mat[token]['collocates'][ctoken]['collocate_npmi']
            lmi = mat[token]['collocates'][ctoken]['collocate_lmi']
            avg_dist = average(mat[token]['collocates'][ctoken]['collocate_distance'])
            row = [token, ctoken, freq, pmi, npmi, lmi, avg_dist]
            all.append(row)
    return all


df_after = pd.DataFrame(format_matrix(atd), columns=['pivot', 'target', 'collocate_frequency', 'collocate_pmi', 'collocate_npmi', 'collocate_lmi', 'avg_distance']).sort_values(by=['pivot', 'target'], ascending=True)
df_before = pd.DataFrame(format_matrix(btd), columns=['pivot', 'target', 'collocate_frequency', 'collocate_pmi', 'collocate_npmi', 'collocate_lmi', 'avg_distance']).sort_values(by=['pivot', 'target'], ascending=True)
df_both = pd.DataFrame(format_matrix(otd), columns=['pivot', 'target', 'collocate_frequency', 'collocate_pmi', 'collocate_npmi', 'collocate_lmi', 'avg_distance']).sort_values(by=['pivot', 'target'], ascending=True)
if save_window:
    dcolumns = [str('Left-' + str(abs(t))) for t in range(-wsize[0], 0)]
    dcolumns.extend(['Pivot word'])
    dcolumns.extend([str('Right-' + str(t)) for t in range(1, wsize[1] + 1)])
    dcolumns.extend(['Distance'])
    df_collocs = pd.DataFrame(save_collocations, columns=dcolumns)

print("Saving output to " + str(output_file_excel))

with pd.ExcelWriter(output_file_excel) as writer:
    df_after.to_excel(writer, sheet_name="After")
    df_before.to_excel(writer, sheet_name="Before")
    df_both.to_excel(writer, sheet_name="Complete")
    if save_window:
        df_collocs.to_excel(writer, sheet_name="Collocations")

print("Deleting temp files")
if isfile(intermediate_file):
    os.remove(intermediate_file)

print("All done!")