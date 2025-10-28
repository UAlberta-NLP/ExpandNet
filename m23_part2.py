import sys
import os
import argparse
import pandas as pd

import xml_utils

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  # parser.add_argument("--src_data", type=str, default="xlwsd_se13.xml",
  #                     help="Path to the XLWSD XML corpus file.")
  # parser.add_argument("--src_gold", type=str, default="xlwsd_se13.key.txt",
  #                     help="Path to the gold sense tagging file.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--full_df_file", type=str, default="xlwsd_se13_en_fr_full.pkl",
                      help="File to load sentences, translations, alignments, and golds from.")
  return parser.parse_args()


args = parse_args()

# src_data = args.src_data
# src_gold = args.src_gold
lang_src = args.lang_src
lang_tgt = args.lang_tgt
full_df_file = args.full_df_file

# Print argument details.
print(f"Running on: {lang_src} -> {lang_tgt}")
#print(f"Corpus: {src_data}")
#print(f"Gold tags: {src_gold}")
print(f"Data file: {full_df_file}")

# df_src = xml_utils.process_dataset(src_data, src_gold)
# print(df_src.head(30), '\n')

if os.path.exists(full_df_file):
  print(f'The file "{full_df_file}" exists. Loading...')
  df_sent = pd.read_pickle(full_df_file)
  print('Loading complete.')
else:
  print(f'The file "{full_df_file}" does not exist. Exiting...')
  sys.exit()


print()
print(df_sent.head(5), '\n')
print(df_sent.iloc[1], '\n')


### CHANGE THIS PART TO SWAP TOKENIZERS.
import spacy
pipelines = {}
def tokenize_sentence(sentence: str, lang: str, lemmatize: bool = False) -> list:
  # Check if pipeline is already loaded
  if lang not in pipelines:
    model_map = {
      'en': 'en_core_web_lg',
      'fr': 'fr_core_news_lg',
    }

    model_name = model_map.get(lang, f"{lang}_core_news_lg")

    try:
      nlp = spacy.load(model_name)
    except:
      print(f'Could not load spacy model {model_name}. Exiting...')
      sys.exit()

    # Cache the pipeline
    pipelines[lang] = nlp

  # Use cached pipeline
  doc = pipelines[lang](sentence)
  if lemmatize:
    return([token.lemma_ for token in doc])
  else:
    return([token.text   for token in doc])
### END OF TOKENIZER CODE.


def get_alignments(alignments, i):
  js = [link[1] for link in alignments if link[0] == i]
  return(js)

print()  
for _, row in df_sent.iterrows():
  sid = row['sentence_id']
  src = row['lemma'].split(' ')
  tgt = tokenize_sentence(row['translation'], lang_tgt, lemmatize=True)
  ali = row['alignment']
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
      candidates = [ '_'.join( [ tgt[j] for j in alignment_indices ] ) ]
    elif len(alignment_indices) == 1:
      candidates = [ tgt[alignment_indices[0]] ]
    else:
      candidates = []

    if candidates:
      for candidate in candidates:
        print('CANDIDATE', src[i], bn, candidate, sep='\t')

  print()
