import sys
import os
import argparse
import pandas as pd

import xml_utils

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="xlwsd_se13.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--src_gold", type=str, default="xlwsd_se13.key.txt",
                      help="Path to the gold sense tagging file.")
  parser.add_argument("--dictionary", type=str, default="wikpan-en-fr.tsv",
                      help="Use a dictionary for filtering. Available options: none, bn (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--alignment_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="File containing the output of step 2 (alignment).")
  parser.add_argument("--output_file", type=str, default="expandnet_step3_project.out.tsv")
  parser.add_argument("--join_char", type=str, default='_')
  return parser.parse_args()

args = parse_args()
src_data = args.src_data
src_gold = args.src_gold
dict_file = args.dictionary
alignment_file = args.alignment_file
output_file = args.output_file
join_char = args.join_char

# Load the dataset and alignment data.
df_src = xml_utils.process_dataset(src_data, src_gold)
#print('Dataset loaded')
#print(df_src.iloc[1], '\n')
df_sent = pd.read_csv(alignment_file, sep='\t')
#print('Alignment loaded')
#print(df_sent.iloc[0], '\n')


import csv
def load_dict(filepaths):
    """Load multiple TSV files into a dict: {english_word: set(french_words)}.
    All spaces are normalized to underscores.
    """
    dict_ = {}
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for line_num, row in enumerate(reader, start=1):
                if len(row) < 2:
                    print(f"Warning: Line {line_num} in {filepath} has fewer than 2 columns.")
                    continue
                eng_word = row[0].strip().lower().replace(' ', '_')  # Normalize English key
                fr_words = set(word.strip().lower().replace(' ', '_') for word in row[1].split())
                if eng_word in dict_:
                    dict_[eng_word].update(fr_words)  # Merge sets if key exists
                else:
                    dict_[eng_word] = fr_words
    return dict_


def is_valid_translation(eng_word, fr_word, dict_):
  """Check if (eng_word, fr_word) is a valid translation pair in the dict."""
  eng_word = eng_word.lower().strip().replace(' ', '_')
  fr_word = fr_word.lower().strip().replace(' ', '_')
  if eng_word.lower() not in dict_:
    return False
  return fr_word in dict_[eng_word]


# Load the dictionary.
dict_wik = load_dict([dict_file])

# group by sentence_id and aggregate bn_gold values into a list
bn_gold_lists = (
    df_src.groupby("sentence_id")["bn_gold"]
       .apply(lambda x: [v for v in x])  # drop NaN
       .reset_index(name="bn_gold_list")
)

lemma_gold_lists = (
    df_src.groupby("sentence_id")["lemma"]
       .apply(lambda x: [v for v in x])  # drop NaN
       .reset_index(name="lemma")
)

# merge back into df2


bn_gold_lists = bn_gold_lists.rename(columns={"bn_gold_list": "bn_gold_list"})
lemma_gold_lists = lemma_gold_lists.rename(columns={"lemma": "lemma_gold"})

df_sent = (
    df_sent.merge(bn_gold_lists, on="sentence_id", how="left").merge(lemma_gold_lists, on="sentence_id", how="left")
)

#print()
#print(df_sent.iloc[0], '\n')


def get_alignments(alignments, i):
  js = [link[1] for link in alignments if link[0] == i]
  return(js)


senses = set()
import ast
for _, row in df_sent.iterrows():
  sid = row['sentence_id']
  print(row)
  src = row['lemma_gold']
  tgt = row['translation_lemma'].split(' ')
  ali = ast.literal_eval(row['alignment'])
  bns = row['bn_gold_list']

  print('SID', sid)
  print('TXT', row['text'])
  print('SRC', src)
  print('TGT', tgt)
  print('ALI', ali)
  print('BNs', bns)
  if not (len(src) == len(bns)):
    print('SRC / BNs length mismatch.')
    continue

  for i, bn in enumerate(bns):
    if not str(bn)[:3] == 'bn:':
      continue
    alignment_indices = get_alignments(ali, i)
    if len(alignment_indices) > 1:
      candidates = [ join_char.join( [ tgt[j] for j in alignment_indices ] ) ] #+ [ tgt[j] for j in alignment_indices ]
    elif len(alignment_indices) == 1:
      candidates = [ tgt[alignment_indices[0]] ]
    else:
      candidates = []

    if candidates:
      for candidate in candidates:
        source = src[i]
        filter_pass = is_valid_translation(source, candidate, dict_wik)

        print('', i, bn, src[i], alignment_indices, candidate, filter_pass, sep='\t')
        if bool(filter_pass):
          print('SENSE', bn, candidate, sep='\t')
          senses.add((bn, candidate))
    else:
      print('', i, bn, src[i], alignment_indices, 'NO_CANDIDATES', False, sep='\t')

  
  print()

with open(output_file, 'w') as f:
  for (bn, lemma) in sorted(senses):
    print(bn,lemma,sep='\t',file=f)

print('DONE')
