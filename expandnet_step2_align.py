import argparse
import pandas as pd
import xml_utils

def replace_lemma_with_gold(in_df, gold_file_path):
    
    # Process the gold XML and extract sentences
    df_src = xml_utils.process_xml(gold_file_path)
    df_sent = xml_utils.extract_sentences(df_src)
  
    # Make sure the DataFrames have the same length
    if len(in_df) != len(df_sent):
        raise ValueError(
            f"Row mismatch: in_df has {len(in_df)} rows, but gold df has {len(df_sent)} rows"
        )

    # 1. Create a "dictionary-like" mapping from the gold dataframe
    lemma_map = df_sent.set_index('sentence_id')['lemma']
    tokens_map = df_sent.set_index('sentence_id')['tokens']

    # 2. Safely apply the mappings based on the sentence_id
    in_df['lemma'] = in_df['sentence_id'].map(lemma_map)
    in_df['tokens'] = in_df['sentence_id'].map(tokens_map)
    
    return in_df
  
def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--translation_df_file", type=str, default="expandnet_step1_translate.out.tsv",
                      help="Path to the TSV file containing tokenized translated sentences.")
  parser.add_argument("--src_data", type=str, default="semcor_en.data.dev.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--dictionary", type=str, default="res/dicts/wikpan-en-fr.tsv",
                      help="Use a dictionary with DBAlign. This argument should be a path, the string 'bn' if you are using babelnet, or can be none if you are using simalign.")
  parser.add_argument("--aligner", type=str, default="dbalign",
                      help="Aligner to use ('simalign' or 'dbalign').")
  parser.add_argument("--output_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="Output file to save the file with alignments to.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers to paralellize the alignment computation over. More than one is not recommended on Windows or less powerful machines. (Default: 1)")
  parser.add_argument("--source_join_char", type=str, default='_')
  parser.add_argument("--target_join_char", type=str, default='_')
  
  return parser.parse_args()


def safe_replace(s, old, new):
    if old == "":
        return s
    return s.replace(old, new)

args = parse_args()

SOURCE_JOIN_CHAR = args.source_join_char
TARGET_JOIN_CHAR = args.target_join_char

print(f"Languages:   {args.lang_src} -> {args.lang_tgt}")
print(f"Aligner:     {args.aligner}")
print(f"Input file:  {args.translation_df_file}")
print(f"Output file: {args.output_file}")

if args.aligner == 'simalign':
  from simalign import SentenceAligner
  ali = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
  def align(lang_src, lang_tgt, tokens_src, tokens_tgt, lemmas_src=None, lemmas_tgt=None):
    alignment_links = ali.get_word_aligns(tokens_src, tokens_tgt)['itermax']
    return(alignment_links)

elif args.aligner == 'dbalign':
  from align_utils import DBAligner
  if args.dictionary == 'bn':
    print("Initializing DBAlign with BabelNet.")
    ali = DBAligner(args.lang_src, args.lang_tgt)
  else:
    print("Initializing DBAlign with Provided Dictionary.")
    ali = DBAligner(args.lang_src, args.lang_tgt, 'custom', args.dictionary, args.source_join_char, args.target_join_char)

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

  def align(lang_src, lang_tgt, tokens_src, tokens_tgt, lemmas_src, lemmas_tgt):
    tokens_tgt = [safe_replace(a, TARGET_JOIN_CHAR, " ") for a in tokens_tgt]
    alignment_spans = ali.new_align(tokens_src, tokens_tgt, lemmas_src, lemmas_tgt)
   
    return(spans_to_links(alignment_spans))

if args.num_workers > 1:
    from pandarallel import pandarallel
    pandarallel.initialize(
        progress_bar=True,
        nb_workers=args.num_workers
    )
    apply_fn = lambda df, fn: df.parallel_apply(fn, axis=1)
else:
    from tqdm import tqdm
    tqdm.pandas()
    apply_fn = lambda df, fn: df.progress_apply(fn, axis=1)

print(f"Loading data from {args.translation_df_file}...")
df_sent = pd.read_csv(args.translation_df_file, sep='\t')

replace_lemma_with_gold(df_sent, args.src_data)

print(f"Loaded {len(df_sent)} sentences\n")

print("Aligning sentences...")
df_sent['alignment'] = apply_fn(
    df_sent,
    lambda row: align(
        args.lang_src,
        args.lang_tgt,
        row['tokens'].split(),
        row['translation_token'].split(),
        row['lemma'].split(' '),
        row['translation_lemma'].split(' ')
    )
)

print(f"\nSaving results to {args.output_file}...")
df_sent.to_csv(args.output_file, sep='\t', index=False)
print("Complete!")