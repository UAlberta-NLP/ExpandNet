import argparse
import pandas as pd
import spacy
from transformers import pipeline
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
  parser.add_argument("--join_char", type=str, default='_')
  return parser.parse_args()

# Parse the arguments.
args = parse_args()

# Print argument details.
print(f"Languages:   {args.lang_src} -> {args.lang_tgt}")
print(f"Corpus:      {args.src_data}")
print(f"Output file: {args.output_file}")

# Load the data.
df_src = xml_utils.process_xml(args.src_data)
print(f'Data loaded: {len(df_src)} rows')

df_sent = xml_utils.extract_sentences(df_src)
print(f'Sentences assembled: {len(df_sent)} rows')

# Translate.
tr_model = f"Helsinki-NLP/opus-mt-{args.lang_src}-{args.lang_tgt}"
try:
  pipe = pipeline("translation", model=tr_model, device=0)
except OSError:
  raise RuntimeError(f"Unsupported language pair: {args.lang_src} -> {args.lang_tgt}")

model_map = {
  'en': 'en_core_web_lg',
  'zh': 'zh_core_web_lg',
  'fr': 'fr_core_news_lg',
  'es': 'es_core_news_lg'
}

# Chinese doesn't use lemmatization
lemmatize = False if args.lang_tgt in ['zh'] else True

# Load spacy pipelines
pipelines = {}

try:
  pipelines[args.lang_src] = spacy.load(model_map.get(args.lang_src, f"{args.lang_src}_core_news_lg"))
except OSError:
  print(f"No spacy pipeline found for source language {args.lang_src}")

try:
  pipelines[args.lang_tgt] = spacy.load(model_map.get(args.lang_tgt, f"{args.lang_tgt}_core_news_lg"))
except OSError:
  print(f"No spacy pipeline found for target language {args.lang_tgt}")


def tokenize_sentence(sentence: str, lang: str, join_char: str, lemmatize: bool = False):
  doc = pipelines[lang](sentence)
  if lemmatize:
    return ' '.join(token.lemma_.replace(' ', join_char) for token in doc)
  else:
    return ' '.join(token.text.replace(' ', join_char) for token in doc)

translations = pipe(df_sent['text'].tolist(), batch_size=16)
df_sent['translation'] = [t['translation_text'] for t in translations]

df_sent['translation_token'] = df_sent['translation'].apply(
    lambda s: tokenize_sentence(s, args.lang_tgt, args.join_char, False)
)

df_sent['translation_lemma'] = df_sent['translation'].apply(
    lambda s: tokenize_sentence(s, args.lang_tgt, args.join_char, lemmatize)
)

print(f'Translation complete: {len(df_sent)} sentences processed\n')

print(f'Saving to "{args.output_file}"...')
cols = ['sentence_id', 'text', 'translation', 'lemma', 'translation_token', 'translation_lemma']
df_sent[cols].to_csv(args.output_file, sep='\t', index=False)
print('Complete!')