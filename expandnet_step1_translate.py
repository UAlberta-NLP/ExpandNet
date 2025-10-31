import sys
import os
import argparse
import pandas as pd
import spacy
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
try:
  pipe = pipeline("translation", model=tr_model, device=0)
except OSError:
  raise RuntimeError("unsupported sentence pair:" + str(lang_src) + ' ' + str(lang_tgt))


model_map = {
      'en': 'en_core_web_lg',
      'zh': 'zh_core_web_lg',
      'fr': 'fr_core_news_lg',
      'es': 'es_core_news_lg'
    }

# Keep hold of lemmatization and tokenization pipelines
pipelines = {}

try:
  pipelines[lang_src] = spacy.load(model_map.get(lang_src, f"{lang_src}_core_news_lg"))
except KeyError:
  print("No spacy pipeline found for source language", lang_src, "no lemmatization will be done on source language")
  
try:
  pipelines[lang_tgt] = spacy.load(model_map.get(lang_tgt, f"{lang_tgt}_core_news_lg"))
except KeyError:
  print("No spacy pipeline found for target language", lang_tgt, "no lemmatization will be done on target language")


def tokenize_sentence(sentence: str, lang: str, lemmatize: bool = False):
  # Check if pipeline is already loaded
  # Use cached pipeline
  doc = pipelines[lang](sentence)
  if lemmatize:
    return ' '.join([a.replace(' ', '_') for a in [token.lemma_ for token in doc]])
  else:
    return ' '.join([a.replace(' ', '_') for a in [token.text   for token in doc]])
  
def translate(sentence):
    output = pipe(sentence)[0]['translation_text']
    return(output)

df_sent['translation'] = df_sent['text'].apply(
    lambda s: translate(s)
)

#df_sent['lemma'] = df_sent['text'].apply(
#    lambda s: tokenize_sentence(s, lang_src, True)
#)

df_sent['translation_token'] = df_sent['translation'].apply(
    lambda s: tokenize_sentence(s, lang_tgt)
)

df_sent['translation_lemma'] = df_sent['translation'].apply(
    lambda s: tokenize_sentence(s, lang_tgt, True)
)

print(df_sent.head(5), '\n')

print(f'Creation complete. Saving to "{output_file}"...')
cols = ['sentence_id', 'text', 'translation', 'lemma', 'translation_token', 'translation_lemma']
df_sent[cols].to_csv(output_file, sep='\t', index=False)
print(f'Saving complete.')

