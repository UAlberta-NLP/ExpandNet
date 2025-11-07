import argparse
import pandas as pd

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

print(f"Languages:   {args.lang_src} -> {args.lang_tgt}")
print(f"Aligner:     {args.aligner}")
print(f"Input file:  {args.translation_df_file}")
print(f"Output file: {args.output_file}")

if args.aligner == 'simalign':
  from simalign import SentenceAligner
  ali = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
  def align(lang_src, lang_tgt, tokens_src, tokens_tgt):
    alignment_links = ali.get_word_aligns(tokens_src, tokens_tgt)['itermax']
    return(alignment_links)

elif args.aligner == 'dbalign':
  from align_utils import DBAligner
  if args.dict == 'bn':
    print("Initializing DBAlign with BabelNet.")
    ali = DBAligner(args.lang_src, args.lang_tgt)
  else:
    print("Initializing DBAlign with Provided Dictionary.")
    ali = DBAligner(args.lang_src, args.lang_tgt, 'custom', args.dict)

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

pandarallel.initialize(progress_bar=True, nb_workers=5)

print(f"Loading data from {args.translation_df_file}...")
df_sent = pd.read_csv(args.translation_df_file, sep='\t')
print(f"Loaded {len(df_sent)} sentences\n")

print("Aligning sentences...")
df_sent['alignment'] = df_sent.parallel_apply(
    lambda row: align(args.lang_src,
                      args.lang_tgt,
                      row['lemma'].split(' '),
                      row['translation_lemma'].split(' ')),
    axis=1
)

print(f"\nSaving results to {args.output_file}...")
df_sent.to_csv(args.output_file, sep='\t', index=False)
print("Complete!")