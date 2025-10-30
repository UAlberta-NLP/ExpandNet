import sys
import os
import argparse
import pandas as pd

import xml_utils

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="xlwsd_se13.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--output_file", type=str, default="expandnet_step1_translate.out.tsv",
                      help="File to store sentences and translations.")
  return parser.parse_args()

# Parse the arguments.
args = parse_args()
src_data = args.src_data
lang_src = args.lang_src
lang_tgt = args.lang_tgt
output_file = args.output_file

# Print argument details.
print(f"Languages:   {lang_src} -> {lang_tgt}")
print(f"Corpus:      {src_data}")
print(f"Output file: {output_file}")

# Load the data.
df_src = xml_utils.process_xml(src_data)
print('Data loaded:')
print(df_src.head(10), '\n')
df_sent = xml_utils.extract_sentences(df_src)
print('Sentences assembled:')
print(df_sent.head(3), '\n')

# Translate.
tr_model = f"Helsinki-NLP/opus-mt-{lang_src}-{lang_tgt}"
from transformers import pipeline
pipe = pipeline("translation", model=tr_model, device=0)

def translate(sentence):
    output = pipe(sentence)[0]['translation_text']
    return(output)

df_sent['translation'] = df_sent['text'].apply(
    lambda s: translate(s)
)
print(df_sent.head(5), '\n')

print(f'Creation complete. Saving to "{output_file}"...')
cols = ['sentence_id', 'text', 'translation']
df_sent[cols].to_csv(output_file, sep='\t', index=False)
print(f'Saving complete.')

