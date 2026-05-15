import argparse
import pandas as pd
import spacy
from transformers import pipeline
import xml_utils
from torch import cuda
from tqdm import tqdm

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--output_file", type=str, default="expandnet_step1_translate.out.tsv",
                      help="File to store sentences and translations.")
  parser.add_argument("--translator", type=str, default="gpt",
                      help="What translator to use? Enter 'gpt' or 'helsinki'.")
  parser.add_argument(
    "--no_pos",
    action="store_false",
    dest="pos_tag",
    help="Optionally turn OFF the part-of-speech tagging which can be used downstream for filtering. (Default: tagging is ON)."
)
  parser.add_argument("--target_join_char", type=str, default='_')
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

translation_system = args.translator.lower()

assert translation_system in ['gpt', 'helsinki']


# Translate.

if translation_system == 'helsinki':
 tr_model = f"Helsinki-NLP/opus-mt-{args.lang_src}-{args.lang_tgt}"
 try:
  pipe = pipeline("translation", model=tr_model, device=0 if cuda.is_available() else -1)
 except OSError:
  raise RuntimeError(f"Unsupported language pair: {args.lang_src} -> {args.lang_tgt}")
 except ValueError:
  pipe = pipeline("translation", model=tr_model)
  
 translations = pipe(df_sent['text'].tolist(), batch_size=16)
 
else:
  from gpt_translate import translate_gpt
  translations = [{'translation_text': translate_gpt(x, args.lang_src, args.lang_tgt)} for x in tqdm(df_sent['text'].tolist())]
  

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

CACHE = {}

def tokenize_sentence(sentence: str, lang: str, join_char: str, lemmatize: bool = False):
  key = (sentence, lang)
  
  if key not in CACHE:
    CACHE[key] = pipelines[lang](sentence)
  else:
    pass
  doc = CACHE[key]
    
  if lemmatize:
    return ' '.join(token.lemma_.replace(' ', join_char) for token in doc)
  else:
    return ' '.join(token.text.replace(' ', join_char) for token in doc)
  
def pos_tag_sentence(sentence: str, lang: str, join_char: str):
  doc = pipelines[lang](sentence)
  return ' '.join(token.pos_.replace(' ', join_char) for token in doc)



df_sent['translation'] = [t['translation_text'] for t in translations]

print("Translation complete!")

print("Tokenizing...")

df_sent['translation_token'] = df_sent['translation'].apply(
    lambda s: tokenize_sentence(s, args.lang_tgt, args.target_join_char, False)
)

df_sent['translation_lemma'] = df_sent['translation'].apply(
    lambda s: tokenize_sentence(s, args.lang_tgt, args.target_join_char, lemmatize)
)

if args.pos_tag:
  df_sent['translation_pos'] = df_sent['translation'].apply(
    lambda s: pos_tag_sentence(s, args.lang_tgt, args.target_join_char)
  )
  cols = ['sentence_id', 'text', 'translation', 'lemma', 'translation_token', 'translation_lemma', 'translation_pos']
else:
  cols = ['sentence_id', 'text', 'translation', 'lemma', 'translation_token', 'translation_lemma']

print(f'Tokenization complete: {len(df_sent)} sentences processed\n')

print(f'Saving to "{args.output_file}"...')

df_sent[cols].to_csv(args.output_file, sep='\t', index=False)
print('Complete!')