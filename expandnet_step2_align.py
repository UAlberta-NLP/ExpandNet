import argparse
import pandas as pd
import sys

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--translation_df_file", type=str, default="expandnet_step1_translate.out.tsv",
                      help="Path to the TSV file containing tokenized translated sentences.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--dict", type=str, default="wikpan-en-es.tsv",
                      help="Use a dictionary with DBAlign. This argument should be a path, the string 'bn' if you are using babelnet, or can be none if you are using simalign.")
  parser.add_argument("--aligner", type=str, default="dbalign",
                      help="Aligner to use ('simalign' or 'dbalign').")
  parser.add_argument("--output_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="Output file to save the file with alignments to.")
  
  return parser.parse_args()

args = parse_args()


lang_src = args.lang_src
lang_tgt = args.lang_tgt
usedict  = args.dict
translation_df_file = args.translation_df_file
aligner = args.aligner
output_file = args.output_file

if aligner == 'simalign':
  from simalign import SentenceAligner
  ali = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
  def align(lang_src, lang_tgt, tokens_src, tokens_tgt):
    alignment_links = ali.get_word_aligns(tokens_src, tokens_tgt)['itermax']
    return(alignment_links)

elif aligner == 'dbalign':
  from align_utils import DBAligner
  if usedict == 'bn':
    print("Initializing DBAlign with BabelNet.")
    ali = DBAligner(lang_src, lang_tgt)
  else:
    print("Initializing DBAlign with Provided Dictionary.")
    ali = DBAligner(lang_src, lang_tgt, 'custom', usedict)

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
df_sent = pd.read_csv(translation_df_file, sep='\t')

df_sent['alignment'] = apply_with_progress(
    df_sent,
    lambda row: align(lang_src, 
                      lang_tgt,
                      row['lemma'].split(' '),
                      row['translation_lemma']),
    axis=1
)

df_sent.to_csv(output_file, sep='\t', index=False)