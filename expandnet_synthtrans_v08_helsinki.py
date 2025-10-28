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
  parser.add_argument("--usedict", type=str, default="none",
                      help="Use a dictionary for filtering. Available options: none, bn (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--translation_df_file", type=str, default="xlwsd_se13_en_fr",
                      help="File to either create or load sentences and translations to or from.")
  parser.add_argument("--aligner", type=str, default="simalign",
                      help="Aligner to use ('simalign' or 'dbalign').")
  return parser.parse_args()


args = parse_args()

src_data = args.src_data
src_gold = args.src_gold
lang_src = args.lang_src
lang_tgt = args.lang_tgt
usedict  = args.usedict
translation_df_file = args.translation_df_file
aligner = args.aligner


# Check that the dictionary argument is a valid option.
# Define valid options
valid_dicts = {"none", "bn", "wik", "wikpan"}
# Validate
if usedict not in valid_dicts:
    print(f"Error: 'usedict' must be one of: {', '.join(valid_dicts)}")
    print("  - 'none': no dictionary filtering")
    print("  - 'bn': use BabelNet as dictionary")
    print("  - 'wik': use WiktExtract as dictionary")
    print("  - 'wikpan': use WiktExtract and PanLex as dictionaries")
    sys.exit(1)


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


if usedict == 'wik':
  dict_wik = load_dict(['wiktextract-en-fr.tsv'])
elif usedict == 'wikpan':
  dict_wik = load_dict(['wiktextract-en-fr.tsv','panlex-en-fr.tsv'])


# Print argument details.
print(f"Running on: {lang_src} -> {lang_tgt}")
print(f"Dictionary filter: {usedict}")
print(f"Corpus: {src_data}")
print(f"Gold tags: {src_gold}")

# Make sure the aligner is a valid choice
def validate_aligner(aligner):
  valid_options = {"simalign", "dbalign"}
  if aligner not in valid_options:
    print(f"Error: Invalid aligner '{aligner}'. Must be one of: {', '.join(valid_options)}")
    sys.exit(1)
  print(f"Aligner: {aligner}")
validate_aligner(aligner)


df_src = xml_utils.process_dataset(src_data, src_gold)
#df_src = df_src.loc[ df_src['sentence_id'] == 'semeval2013.d000.s017' ]
print(df_src.head(30), '\n')

if os.path.exists(translation_df_file):
  print(f'The file "{translation_df_file}" exists. Loading...')
  df_sent = pd.read_csv(translation_df_file, sep='\t')
  print('Loading complete.')

else:
  print(f'The file "{translation_df_file}" does not exist. Creating...')

  df_sent = xml_utils.extract_sentences(df_src)
  print(df_sent.head(5), '\n')

  ### CHANGE THIS PART TO SWAP TRANSLATORS.
  from transformers import pipeline
  pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr", device=0)
  def translate(sentence, lang_src, lang_tgt):
    if lang_src == 'en' and lang_tgt == 'fr':
      output = pipe(sentence)[0]['translation_text']
      return(output)
    else:
      print('As of this version, only English-to-French translation is supported.')
      sys.exit()
  ### END OF TRANSLATION MODULE.

  df_sent['translation'] = df_sent['text'].apply(
    lambda sentence: translate(sentence, lang_src, lang_tgt)
  )
  print(df_sent.head(5), '\n')

  print(f'Creation complete. Saving to "{translation_df_file}"...')
  df_sent.to_csv(translation_df_file, sep='\t', index=False)
  print(f'Saving complete.')

#df_sent = df_sent.loc[ df_sent['sentence_id'] == 'semeval2013.d000.s017' ]


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
    except OSError:
      # Fallback to small model
      fallback_model = model_name.replace('_lg', '_sm')
      nlp = spacy.load(fallback_model)

    # Cache the pipeline
    pipelines[lang] = nlp

  # Use cached pipeline
  doc = pipelines[lang](sentence)
  if lemmatize:
    return([token.lemma_ for token in doc])
  else:
    return([token.text   for token in doc])
### END OF TOKENIZER CODE.

### CHANGE THIS PART TO SWAP ALIGNERS.
if aligner == 'simalign':
  from simalign import SentenceAligner
  ali = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
  def align(lang_src, lang_tgt, tokens_src, tokens_tgt):
    alignment_links = ali.get_word_aligns(tokens_src, tokens_tgt)['itermax']
    return(alignment_links)

elif aligner == 'dbalign':
  from babelnetalign import DBAligner
  if usedict == 'bn':
    print("Initializing DBAlign with BabelNet.")
    ali = DBAligner(lang_src, lang_tgt)
  elif usedict == 'wik':
    print("Initializing DBAlign with WiktExtract.")
    ali = DBAligner(lang_src, lang_tgt, 'custom', 'wiktextract-en-fr.tsv')
  elif usedict == 'wikpan':
    print("Initializing DBAlign with Wik+Pan.")
    ali = DBAligner(lang_src, lang_tgt, 'custom', 'wikpan-en-fr.tsv')
  else:
    print("Initializing DBAlign with Wik+Pan.")
    ali = DBAligner(lang_src, lang_tgt, 'custom', 'wikpan-en-fr.tsv')

  def spans_to_links(span_string):
    span_string = span_string.strip()
    span_list = span_string.split(' ')
    links = set()
    for s in span_list:
      try:
        (x_start, x_end, y_start, y_end) = s.split('-')
        for x in range(int(x_start), int(x_end)+1):
          for y in range(int(y_start), int(y_end)+1):
            links.add((x,y))
      except:
        pass
    return(sorted(links))

  def align(lang_src, lang_tgt, tokens_src, tokens_tgt):
    alignment_spans = ali.new_align(tokens_src, tokens_tgt)
    return(spans_to_links(alignment_spans))
### END OF ALIGNMENT MODULE.

from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=False, nb_workers=16)
pandarallel.initialize(progress_bar=False, nb_workers=5)
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

#df_sent.to_pickle('expandnet_synthtrans_v07a_trans-and-ali.pkl')

# group by sentence_id and aggregate bn_gold values into a list
bn_gold_lists = (
    df_src.groupby("sentence_id")["bn_gold"]
       .apply(lambda x: [v for v in x])  # drop NaN
       .reset_index(name="bn_gold_list")
)

# merge back into df2
df_sent = df_sent.merge(bn_gold_lists, on="sentence_id", how="left")

print()
print(df_sent.head(5), '\n')

def get_alignments(alignments, i):
  js = [link[1] for link in alignments if link[0] == i]
  return(js)

from bn_utils import bnsyn, synonyms, bnsyn_save
bnsyn_save()
  
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
      candidates = [ '_'.join( [ tgt[j] for j in alignment_indices ] ) ] #+ [ tgt[j] for j in alignment_indices ]
    elif len(alignment_indices) == 1:
      candidates = [ tgt[alignment_indices[0]] ]
    else:
      candidates = []

    if candidates:
      for candidate in candidates:
        source = src[i]
        if usedict == 'bn':
          # Filter with BN dictionary.
          filter_pass = False
          if source.lower() == candidate.lower():
            filter_pass = True
          elif bnsyn(lang_src, lang_tgt, source, candidate):
            filter_pass = True
        elif usedict in ['wik', 'wikpan']:
          filter_pass = is_valid_translation(source, candidate, dict_wik)
        else:
          # Do not filter.
          filter_pass = 'n/a'

        print('', i, bn, src[i], alignment_indices, candidate, filter_pass, sep='\t')
        if bool(filter_pass):
          print('SENSE', bn, candidate, sep='\t')
    else:
      print('', i, bn, src[i], alignment_indices, 'NO_CANDIDATES', False, sep='\t')

  
  print()


# Save BN synonymy checks.
bnsyn_save()
print('DONE')
