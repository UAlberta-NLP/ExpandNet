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
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--translation_df_file", type=str, default="xlwsd_se13_en_fr",
                      help="File to load sentences and translations from.")
  parser.add_argument("--full_df_file", type=str, default="xlwsd_se13_en_fr_full.pkl",
                      help="File to save alignment and gold data to.")
  return parser.parse_args()


args = parse_args()

src_data = args.src_data
src_gold = args.src_gold
lang_src = args.lang_src
lang_tgt = args.lang_tgt
translation_df_file = args.translation_df_file
full_df_file = args.full_df_file

# Print argument details.
print(f"Running on: {lang_src} -> {lang_tgt}")
print(f"Corpus: {src_data}")
print(f"Gold tags: {src_gold}")

df_src = xml_utils.process_dataset(src_data, src_gold)
print(df_src.head(30), '\n')

if os.path.exists(translation_df_file):
  print(f'The file "{translation_df_file}" exists. Loading...')
  df_sent = pd.read_csv(translation_df_file, sep='\t')
  print('Loading complete.')
else:
  print(f'The file "{translation_df_file}" does not exist. Exiting...')
  sys.exit()


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


### Set up aligner.
from simalign import SentenceAligner
ali = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="mai")
def align(lang_src, lang_tgt, tokens_src, tokens_tgt):
  alignment_links = ali.get_word_aligns(tokens_src, tokens_tgt)['itermax']
  return(alignment_links)


from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=8)
from tqdm import tqdm

# Create your own progress bar that writes to stderr
def apply_with_progress(df, func, axis=1):
    results = []
    with tqdm(total=len(df), file=sys.stderr, desc="Processing") as pbar:
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            pbar.update(1)
            return result
        
        # Apply with your wrapped function
        results = df.apply(wrapped_func, axis=axis)
    return results

df_sent['alignment'] = apply_with_progress(
    df_sent,
    lambda row: align(lang_src, 
                      lang_tgt,
                      row['lemma'].split(' '),
                      tokenize_sentence(row['translation'], lang_tgt, lemmatize=True)),
    axis=1
)


print()
print(df_sent.head(5), '\n')

# group by sentence_id and aggregate bn_gold values into a list
bn_gold_lists = (
    df_src.groupby("sentence_id")["bn_gold"]
       .apply(lambda x: [v for v in x])  # drop NaN
       .reset_index(name="bn_gold_list")
)


df_sent = df_sent.merge(bn_gold_lists, on="sentence_id", how="left")

print()
print(df_sent.head(5), '\n')
print(df_sent.iloc[1], '\n')

print(f'Writing corpus information to {full_df_file}...')
df_sent.to_pickle(full_df_file)
print('Done.')

print('END PART 1.')


